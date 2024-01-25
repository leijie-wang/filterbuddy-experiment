from cgi import test
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from datasets.dataset import Dataset
import sharedsteps.utils as utils
import logging
import json, random
from sharedsteps.models import SYSTEMS, Participant

logger = logging.getLogger(__name__)


BuildDataSet = Dataset("train")
UpdateDataSet = Dataset("train")

def onboarding(request):
    """
        the starting point of the experiment
    """
    system = request.GET.get('system', default=None)

    new_participant = Participant.create_participant(system=system)
    return redirect(f"/groundtruth?participant_id={new_participant.participant_id}&stage={new_participant.stage}")

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

    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    from sharedsteps.models import GroundTruth
    GroundTruth.objects.filter(participant_id=participant_id, stage=stage).delete()
    for datum in dataset:
        GroundTruth(participant_id=participant_id, text=datum["text"], label=datum["label"], stage=stage).save()
    return JsonResponse({"status": True, "message": "Participants' ground truth are stored successfully"}, safe=False)

def load_system(request):
    print(request.body)
    participant_id = request.GET.get('participant_id', default=None)
    stage = request.GET.get('stage', default=None)

    error_message = utils.check_parameters(participant_id, stage)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    participant = Participant.objects.get(participant_id=participant_id)
    system = participant.system
    logging.debug(f"participant_id: {participant_id}, system: {system}, stage: {stage}")

    if system == SYSTEMS.EXAMPLES_ML.value:
        return redirect(f'/examplelabel?participant_id={participant_id}&system={system}&stage={stage}')
    elif system == SYSTEMS.RULES_TREES.value:
        return redirect(f'/ruleconfigure?participant_id={participant_id}&system={system}&stage={stage}')
    elif system == SYSTEMS.PROMPTS_LLM.value:
        return redirect(f'/promptwrite?participant_id={participant_id}&system={system}&stage={stage}')
    else:
        logging.error("System unsupported yet: {}".format(system))

# def load_more_data(request):
#     participant_id = request.GET.get('participant_id', default=None)
#     new_batch = TrainDataSet.load_batch(participant_id)
#     print(f"participant {participant_id} loaded a new batch of size {len(new_batch)}")
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
        }
    """
    request_data = json.loads(request.body)
    dataset = request_data.get('dataset')
    participant_id = request_data.get('participant_id')
    stage = request_data.get('stage')

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
    if participant_id is None:
        participant_id = utils.generate_userid()

    system = request.GET.get('system', default=SYSTEMS.PROMPTS_LLM.value)

    
    dataset = TrainDataSet.load_batch(participant_id, start=True)
    return render(request, 'promptwrite.html', {
        "dataset": json.dumps(dataset),
        "participant_id": participant_id,
        "system": system,
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
        prompt = PromptWrite(rubric=item["rubric"], participant_id=participant_id, prompt_id=counter, stage=stage)
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
    if participant_id is None:
        participant_id = utils.generate_userid()
    
    system = request.GET.get('system', default=SYSTEMS.RULES_TREES.value)
    if system not in [SYSTEMS.RULES_TREES.value, SYSTEMS.RULES_ML.value]:
        logging.error(f"Unknown system: {system}")
        return redirect(f'/onboarding')
    
    stage = request.GET.get('stage', default="build")
    if stage == "build":
        dataset = TrainDataSet.load_batch(participant_id, start=True)
        return render(request, 'ruleconfigure.html', {
            "participant_id": participant_id,
            "system": system,
            "stage": stage,
            "rules": json.dumps([]),
            "dataset": json.dumps(dataset),
        })
    elif stage == "update":
        from sharedsteps.utils import read_rules_from_database
        rules = read_rules_from_database(participant_id)

        from sharedsteps.models import GroundTruth
        dataset = TrainDataSet.load_previous_batches(participant_id)
        return render(request, 'ruleconfigure.html', {
            "participant_id": participant_id,
            "system": system,
            "stage": stage,
            "rules": json.dumps(rules),
            "dataset": json.dumps(dataset)     
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

def validate_page(request):
    from sharedsteps.models import GroundTruth
    # parse out the participant id fr dom the request GET parameters
    participant_id = request.GET.get('participant_id', default=None)
    system = request.GET.get('system', default=None)
    stage = request.GET.get('stage', default=None)

    error_message = utils.check_parameters(participant_id, stage, system)
    if error_message is not None:
        return JsonResponse({"status": False, "message": error_message}, safe=False)
    
    dataset = utils.get_groundtruth_dataset(participant_id, stage)
    test_results = test_system(participant_id, dataset)
    return render(request, 'validate.html', {
        "dataset": json.dumps(dataset),
        "participant_id": participant_id,
        "system": system,
        "stage": stage,
        "test_results": json.dumps(test_results),
    })

def test_system(participant_id, test_dataset):
    
    X_test = [item["text"] for item in test_dataset]
    y_test = [item["label"] for item in test_dataset]
    
    participant = Participant.objects.get(participant_id=participant_id)
    system = participant.system
    stage = participant.stage
    print(f"participant {participant_id} tests system: {system} at the stage {stage}")
    
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

    training_dataset = BuildDataSet if stage == "build" else UpdateDataSet
    status, classifier = classifier_class.train(participant_id, training_dataset)
    if not status:
        return JsonResponse({"status": False, "message": classifier}, safe=False)
    
    print(f"starting testing for participant {participant_id} at stage {stage} using system {system}")
    test_results = classifier.test_model(X=X_test, y=y_test)
    return test_results
    
def trainLLM(request):
    from systems.llm_filter import LLMFilter
    request_data = json.loads(request.body)
    
    prompts = request_data.get('prompts')
    dataset = request_data.get('dataset')
    
    dataset = [item["text"] for item in dataset]
    llm_filter = LLMFilter(prompts, debug=False)
    results = llm_filter.test_model(X=dataset, y=None)
    return JsonResponse({
                    "status": True,
                    "message": f"Successfully tested the LLM model",
                    "data": {
                        "results": results
                    }
                }, safe=False
            )

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

def update_page_redirect(request):
    participant_id = request.GET.get('participant_id', default=None)
    system = request.GET.get('system', default=None)
    if participant_id is not None:
        if system in [SYSTEMS.RULES_TREES.value, SYSTEMS.RULES_ML.value]:
            return redirect(f'/ruleconfigure?participant_id={participant_id}&system={system}&stage=update')
        else:
            raise Exception(f"Unsupported system: {system}")