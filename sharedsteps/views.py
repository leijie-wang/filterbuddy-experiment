from calendar import c
from cgi import test
from distutils.command import build
from functools import partial
from math import log
from re import T
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from openai import chat
from datasets.dataset import Dataset
import sharedsteps.utils as utils
import logging
import json, random
from sharedsteps.models import SYSTEMS, Participant

logger = logging.getLogger(__name__)


BuildDataSet = Dataset("old.csv")
UpdateDataSet = Dataset("new.csv")

def onboarding(request):
    """
        the starting point of the experiment
    """
    participant_id = request.GET.get('participant_id', default=None)
    participant = None
    if participant_id is None:
        system = request.GET.get('system', default=None)
        participant = Participant.create_participant(system=system)
        logger.info(f"participant {participant.participant_id} is created with the system {participant.system}")
    else:
        logger.info(f"participant {participant_id} is moving to the update stage")
        participant = Participant.objects.get(participant_id=participant_id)
        participant.stage = "update"
        participant.save()
    return redirect(f"/groundtruth/?participant_id={participant.participant_id}&stage={participant.stage}")

def label_ground_truth(request):
    """
        for both build and update stages, the participant will label the ground truth as the first step
    """
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default=None)
    
    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)

    dataset = []
    if stage == "build":
        dataset = BuildDataSet.load_test_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_test_dataset(participant_id)

    return render(request, 'groundtruth.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "stage": stage
        })

def store_groundtruth(request):
    """
        store the ground truth labeled by the participant at the stage of build or update
    """
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    dataset = request_data.get('dataset')
    logs = request_data.get('logs')

    utils.save_logs(participant_id, stage, logs)
    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    from sharedsteps.models import GroundTruth
    GroundTruth.objects.filter(participant_id=participant_id, stage=stage).delete()
    for datum in dataset:
        GroundTruth(participant_id=participant_id, text=datum["text"], label=datum["label"], stage=stage).save()
    return JsonResponse({"status": True, "message": "Participants' ground truth are stored successfully"}, safe=False)

def load_system(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default=None)

    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    system = participant.system
    logging.debug(f"participant_id: {participant_id}, system: {system}, stage: {stage}")

    if system == SYSTEMS.EXAMPLES_ML.value:
        return redirect(f'/examplelabel/?participant_id={participant_id}&system={system}&stage={stage}')
    elif system == SYSTEMS.RULES_TREES.value:
        return redirect(f'/ruleconfigure/?participant_id={participant_id}&system={system}&stage={stage}')
    elif system == SYSTEMS.PROMPTS_LLM.value:
        return redirect(f'/promptwrite/?participant_id={participant_id}&system={system}&stage={stage}')
    else:
        logging.error("System unsupported yet: {}".format(system))

# def load_more_data(request):
#     participant_id = request.GET.get('participant_id', default=None)
#     new_batch = TrainDataSet.load_batch(participant_id)
#     logger.info(f"participant {participant_id} loaded a new batch of size {len(new_batch)}")
#     return JsonResponse(json.dumps(new_batch), safe=False)

def examplelabel(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system = request.GET.get('system')

    error_message = utils.check_parameters(participant_id, stage, system)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    if stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id)

    return render(request, 'examplelabel.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "stage": stage,
            "system": system,
        })

def store_labels(request):
    """
        request.body
        {
            "dataset": [
                {
                    "text": "text1",
                    "label": 1
                },
                {
                    "text": "text2",
                    "label": 0
                }
            ],
            "participant_id": "123456"
            "stage": "build" or "update"
        }
    """
    request_data = json.loads(request.body)
    dataset = request_data.get('dataset')
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    logs = request_data.get('logs')
    utils.save_logs(participant_id, stage, logs)

    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    from sharedsteps.models import ExampleLabel
    # delete the labels of the participant from the database first, for the testing purposes
    ExampleLabel.objects.filter(participant_id=participant_id, stage=stage).delete()

    for item in dataset:
        example_label = ExampleLabel(participant_id=participant_id, stage=stage, text=item["text"], label=item["label"])
        example_label.save()
    return JsonResponse(
                    {
                        "status": True, 
                        "message": "Participants' labeled examples are stored successfully"
                    }, 
                safe=False
            )

def promptwrite(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system = request.GET.get('system')

    error_message = utils.check_parameters(participant_id, stage, system)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    if stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id)

    prompts = utils.read_prompts_from_database(participant_id, "build")
    return render(request, 'promptwrite.html', {
        "participant_id": participant_id,
        "stage": stage,
        "system": system,
        "dataset": json.dumps(dataset),
        "prompts": json.dumps(prompts)
    })

def store_prompts(request):
    """
        request.body
        {
            "prompts": [
                {
                    "rubric": "Catch all texts that involve race-related hate speech",
                    "positives": ["You are just some mad browskin because Europeans styled on your ass"],
                    "negatives": ["Your daily reminder that the vast majority of athletes are hilariously stupid"],
                },
            ],
            "participant_id": "123456"
        }
    """
    request_data = json.loads(request.body)
    prompts = request_data.get("prompts")
    participant_id = request_data.get('participant_id')
    stage = request_data.get("stage")

    from sharedsteps.models import PromptWrite
    # delete the labels of the participant from the database first, for the testing purposes
    PromptWrite.objects.filter(participant_id=participant_id, stage=stage).delete()

    counter = 0
    for item in prompts:
        prompt = PromptWrite(
            name=item["name"], 
            prompt_id=counter, 
            rubric=item["rubric"], 
            priority=item["priority"],  
            action=item["action"],
            participant_id=participant_id, 
            stage=stage
        )
        prompt.set_positives(item["positives"])
        prompt.set_negatives(item["negatives"])
        prompt.save()
        counter = counter + 1
    
    return JsonResponse(
                {
                    "status": True, 
                    "message": "Participants' prompts are stored successfully"
                }, 
                safe=False
            )

def ruleconfigure(request):
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default="build")
    system = request.GET.get('system')

    error_message = utils.check_parameters(participant_id, stage, system)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = []
    if stage == "build":
        dataset = BuildDataSet.load_train_dataset(participant_id)
    elif stage == "update":
        dataset = UpdateDataSet.load_train_dataset(participant_id)

    # when the participant just starts the updating stage, we should load the rules from the build stage
    rules = utils.read_rules_from_database(participant_id, "build")
    return render(request, 'ruleconfigure.html', {
            "participant_id": participant_id,
            "stage": stage,
            "system": system,
            "dataset": json.dumps(dataset),
            "rules": json.dumps(rules)
        })

def store_rules(request):
    """
    request.body
    {
        "rules": [
            {
                "name": "Catch all texts that involve race-related hate speech",
                "action": 1 for "remove" and 0 for "approve",
                "units": [
                    {
                        "type": "include" or "exclude",
                        "words": list[str]
                    }
                ]
            },
        ],
        "participant_id": "123456",
        "stage": "build" or "update"
    }
    """
    request_data = json.loads(request.body)
    rules = request_data.get("rules")
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')

    from sharedsteps.models import RuleConfigure, RuleUnit
    # delete the existing rules of the participant from the database, note we only deleted rules of the corresponding stage
    RuleConfigure.objects.filter(participant_id=participant_id, stage=stage).delete()

    counter = 0
    for item in rules:
        
        rule = RuleConfigure(
            name=item["name"], 
            action=item["action"], 
            rule_id=counter, 
            priority=item["priority"], 
            variants=item["variants"],
            participant_id=participant_id,
            stage=stage
        )
        rule.save()
        for unit in item["units"]:
            rule_unit = RuleUnit(type=unit["type"], rule=rule)
            rule_unit.set_words(unit["words"])
            rule_unit.save()

        counter = counter + 1

    return JsonResponse(
        {
            "status": True,
            "message": f"Participants' {participant_id} rules are stored successfully for {stage}"
        },
        safe=False
    )

def get_similar_phrases(request):
    print("get_similar_phrases")
    request_data = json.loads(request.body)
    phrases = request_data.get("phrases")
    chatbot = utils.ChatCompletion()
        
    response = chatbot.chat_completion(
        system_prompt="""
            You are expected to suggest 3 the most similar phrases to a given list of phrases. 
            These phrases should also be commonly used in social media. You should also take into consideration common typos. 
            RETURN YOUR RESULTS in the JSON format {"results": [a list of phrases]}
        """,
        user_prompt=f"Given the following phrases: {', '.join(phrases)}",
    )
    response = json.loads(response or "{}")
    if "results" in response:
        logger.info(response["results"])
        return JsonResponse(
            {
                "status": True,
                "message": "Successfully got the similar phrases",
                "data": {
                    "phrases": response["results"]
                }
            },
            safe=False
        )



def validate_page(request):
    
    # parse out the participant id fr dom the request GET parameters
    participant_id = request.GET.get('participant_id', default=None)
    system = request.GET.get('system', default=None)
    stage = request.GET.get('stage', default=None)

    error_message = utils.check_parameters(participant_id, stage, system)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = utils.get_groundtruth_dataset(participant_id, stage)
    status, response = test_system(participant_id, stage, dataset)
    if status == 0:
        return HttpResponse(f"Error: {response}")
    elif status == 1:
        task_id = response
        status, build_task_id = test_system(participant_id, "build", dataset) if stage == "update" else (2, "")
        return render(request, 'validate.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "system": system,
            "stage": stage,
            "build_task_id": build_task_id,
            "task_id": task_id,
        })
    else:
        test_results = response
        utils.save_test_results(participant_id, stage, test_results["prediction"])
        # we want to show the performance of the old system on the new dataset
        status, old_results = test_system(participant_id, "build", dataset) if stage == "update" else (2, {})
        return render(request, 'validate.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "system": system,
            "stage": stage,
            "test_results": json.dumps(test_results),
            "old_results": json.dumps(old_results),
            "task_id": ""
        })
        
def test_filter(request):
    request_data = json.loads(request.body)
    dataset = request_data.get('dataset')
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')
    if stage == "update":
        status, response = test_system(participant_id, "build", dataset)
        old_performance = utils.calculate_stage_performance(participant_id, "build")
        if status == 0:
            return JsonResponse({"status": False, "message": response}, safe=False)
        elif status == 1:
            task_id = response
            return JsonResponse({
                "status": True,
                "message": "Successfully started the testing",
                "data": {
                    "old_performance": old_performance,
                    "task_id": task_id
                }
            })
        else:
            test_results = response
            utils.save_test_results(participant_id, stage, test_results["prediction"], old=True)
            return JsonResponse({
                "status": True,
                "message": "Successfully tested the system",
                "data": {
                    "old_performance": old_performance,
                    "test_results": test_results
                }
            })

def test_system(participant_id, stage, test_dataset):
    """
        test the system of the given stage for a participant
        @param participant_id: the id of the participant
        @param stage: the stage of the system we want to test
        @param test_dataset: the dataset we want to test the system on
    """

    X_test = [item["text"] for item in test_dataset]
    y_test = [item["label"] for item in test_dataset]
    
    participant = Participant.objects.get(participant_id=participant_id)
    system = participant.system
    logger.info(f"participant {participant_id} tests system: {system} of the {stage} stage")
    
    classifier_class = None
    if system == SYSTEMS.EXAMPLES_ML.value:
        from systems.ml_filter import MLFilter
        classifier_class = MLFilter
    elif system == SYSTEMS.PROMPTS_LLM.value:
        from systems.llm_filter import LLMFilter
        classifier_class = LLMFilter
    elif system == SYSTEMS.PROMPTS_ML.value:
        from systems.llm_ml_filter import LLM_ML_MixedFilter
        classifier_class = LLM_ML_MixedFilter
    elif system == SYSTEMS.RULES_TREES.value:
        from systems.trees_filter import TreesFilter
        classifier_class = TreesFilter
    elif system == SYSTEMS.RULES_ML.value:
        from systems.trees_ml_filter import Trees_ML_MixedFilter
        classifier_class = Trees_ML_MixedFilter

    training_dataset = BuildDataSet if stage == "build" else UpdateDataSet # used in the llm_ml_filter and trees_ml_filter
    status, classifier = classifier_class.train(participant_id, dataset=training_dataset, stage=stage)
    if not status:
        return 0, classifier # the error message
    
    logger.info(f"starting testing for participant {participant_id} at stage {stage} using system {system}")
    if system == SYSTEMS.PROMPTS_LLM.value:
        from sharedsteps.tasks import test_llm_classifier
        task = test_llm_classifier.delay(classifier.prompts, X_test, y_test)
        return 1, task.id
    else:
        test_results = classifier.test_model(X=X_test, y=y_test)
        return 2, test_results

    
def train_LLM(request):
    from sharedsteps.tasks import train_llm_task
    logger.info(f"starting training the LLM model")
    request_data = json.loads(request.body)
    
    prompts = request_data.get('prompts')
    dataset = request_data.get('dataset')
    
    dataset = [item["text"] for item in dataset]
    
    task = train_llm_task.delay(prompts, dataset)
    logger.info(f"task {task.id} is started to train the LLM model")
    return JsonResponse({
                    "status": True,
                    "message": f"Successfully started the LLM model training",
                    "data": {
                        "task_id": task.id
                    }
                }, safe=False
            )

def get_LLM_results(request, task_id):
    from celery.result import AsyncResult
    
    task_result = AsyncResult(task_id)
    if task_result.ready():
        logger.info(f"task result {task_result.get()}")
        return JsonResponse({
            "status": True,
            "message": f"Task {task_id} is completed",
            "data": {
                "results": task_result.get()
            }
        })
    else:
        return JsonResponse({
            "status": False,
            "message": "Task is still processing"
        })
    
def get_validate_results(request):
    
    from celery.result import AsyncResult
    # get the parameter of the GET request
    participant_id = request.GET.get('participant_id')
    stage = request.GET.get('stage')
    task_id = request.GET.get('task_id')

    
    task_result = AsyncResult(task_id)
    if task_result.ready():
        results = task_result.get()
        
        utils.save_test_results(participant_id, stage, results["prediction"])
        build_performance = utils.calculate_stage_performance(participant_id, "build") if stage == "update" else {}

        return JsonResponse({
            "status": True,
            "message": f"Task {task_id} is completed",
            "data": {
                "test_results": results,
                "build_performance": build_performance
            }
        })
    else:
        return JsonResponse({
            "status": False,
            "message": "Task is still processing"
        })

def train_trees(request):
    from systems.trees_filter import TreesFilter

    request_data = json.loads(request.body)
    rules = request_data.get('rules')
    dataset = request_data.get('dataset')

    dataset = [item["text"] for item in dataset]
    tree_filter = TreesFilter(rules)
    results = tree_filter.test_model(X=dataset, y=None)
    return JsonResponse({
                    "status": True,
                    "message": f"Successfully tested the Trees model",
                    "data": {
                        "results": results
                    }
                }, safe=False
            )