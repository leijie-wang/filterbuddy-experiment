from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from datasets.dataset import Dataset
import sharedsteps.utils as utils
import logging
import json, sys, os, re
from sharedsteps.models import SYSTEMS

logger = logging.getLogger(__name__)


TrainDataSet = Dataset("train")
ValidateDataSet = Dataset("validate")

# Create your views here.
def onboarding(request):
    return render(request, 'onboarding.html')

def load_system(request):
    # parse get parameters of the request
    participant_id = request.GET.get('participant_id', default=None)
    condition = request.GET.get('condition', default=None)
    system, task = condition.split('.')
    # parameters = {
    #         "task": task,
    #         "participant_id": participant_id,
    #     }
    logging.debug(f"participant_id: {participant_id}, system: {system}, task: {task}")
    # redirect participants to the assigned system
    if system == "basic-filter":
        return redirect(f'/wordfilter?task={task}&participant_id={participant_id}')
    elif system == "basic-ML":
        return redirect(f'/examplelabel?task={task}&participant_id={participant_id}')
    elif system == "advanced-ML":
        pass
    elif system == "forest-filter":
        pass
    else:
        logging.error("Unknown system: {}".format(system))

def load_more_data(request):
    participant_id = request.GET.get('participant_id', default=None)
    new_batch = TrainDataSet.load_batch(participant_id)
    print(f"participant {participant_id} loaded a new batch of size {len(new_batch)}")
    return JsonResponse(json.dumps(new_batch), safe=False)

def examplelabel(request):
    participant_id = request.GET.get('participant_id', default=None)
    if participant_id is None:
        participant_id = utils.generate_userid()
    
    dataset = TrainDataSet.load_batch(participant_id, start=True)
    return render(request, 'examplelabel.html', {
         "dataset": json.dumps(dataset),
         "participant_id": participant_id,
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

    from sharedsteps.models import ExampleLabel
    # delete the labels of the participant from the database first, for the testing purposes
    ExampleLabel.objects.filter(participant_id=participant_id).delete()

    for item in dataset:
        example_label = ExampleLabel(participant_id=participant_id, text=item["text"], label=item["label"])
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
    # parse out the participant id from the request GET parameters
    participant_id = request.GET.get('participant_id', default=None)
    system = request.GET.get('system', default=None)
    if participant_id is not None and system is not None:
        dataset = ValidateDataSet.load_all()
        return render(request, 'validate.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
            "system": system,
        })

def validate_system(request):
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    system = request_data.get('system')
    validate_dataset = request_data.get('dataset')

    from sharedsteps.models import GroundTruth
    GroundTruth.objects.filter(participant_id=participant_id, type="validation").delete()
    for item in validate_dataset:
        GroundTruth(participant_id=participant_id, type="validation", text=item["text"], label=item["label"]).save()

    X_test = [item["text"] for item in validate_dataset]
    y_test = [item["label"] for item in validate_dataset]
    print(f"participant {participant_id} validates system: {system}")
    if system == SYSTEMS.EXAMPLES_ML.value:
        # retrieve the labels of the examples labeled by the participant from the database
        from sharedsteps.models import ExampleLabel
        from systems.ml_filter import MLFilter

        training_dataset = list(ExampleLabel.objects.filter(participant_id=participant_id).values("text", "label"))
        if len(training_dataset) == 0:
            return JsonResponse({"status": False, "message": "No labels found for the participant"}, safe=False)
        

        ml_filter = MLFilter("Bayes")
        X_train = [item["text"] for item in training_dataset] 
        y_train = [item["label"] for item in training_dataset]

        print(f"starting training with {len(X_train)} examples labeled by the participant")
        train_results = ml_filter.train_model(X=X_train, y=y_train)

        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = ml_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "status": True,
                    "message": f"Successfully trained and tested the {system} model",
                    "data": {
                        "train_results": train_results,
                        "test_results": test_results
                    }
                }, safe=False
            )
    elif system == SYSTEMS.PROMPTS_LLM.value:
        from systems.llm_filter import LLMFilter
        from sharedsteps.models import PromptWrite

        prompts = read_prompts_from_database(participant_id)
        if len(prompts) == 0:
            return JsonResponse({"status": False, "message": "No prompts found for the participant"}, safe=False)
        
        llm_filter = LLMFilter(prompts, debug=False)
        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = llm_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "status": True,
                    "message": f"Successfully tested the {system} model",
                    "data": {
                        "test_results": test_results
                    }
                }, safe=False
            )
    elif system == SYSTEMS.PROMPTS_ML.value:
        from systems.llm_ml_filter import LLM_ML_MixedFilter
        from sharedsteps.models import PromptWrite

        prompts = list(PromptWrite.objects.filter(participant_id=participant_id).values("rubric", "positives", "negatives", "prompt_id"))
        if len(prompts) == 0:
            return JsonResponse({"status": False, "message": "No prompts found for the participant"}, safe=False)
        
        llm_ml_filter = LLM_ML_MixedFilter(prompts, "Bayes")
        
        X_train = TrainDataSet.load_batch_by_size(participant_id, size=360)
        X_train = [item["text"] for item in X_train]
        train_results = llm_ml_filter.train_model(X=X_train)

        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = llm_ml_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "status": True,
                    "message": f"Successfully trained and tested the {system} model",
                    "data": {
                        "train_results": train_results,
                        "test_results": test_results
                    }
                }, safe=False
            )
    elif system == SYSTEMS.RULES_TREES.value:
        from systems.trees_filter import TreesFilter
        from sharedsteps.utils import read_rules_from_database

        rules = read_rules_from_database(participant_id)
        if len(rules) == 0:
            return JsonResponse({"status": False, "message": "No rules found for the participant"}, safe=False)
        
        tree_filter = TreesFilter(rules)
        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = tree_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "status": True,
                    "message": f"Successfully tested the {system} model",
                    "data": {
                        "test_results": test_results
                    }
                }, safe=False
            )
    elif system == SYSTEMS.RULES_ML.value:
        from systems.trees_ml_filter import Trees_ML_MixedFilter
        from sharedsteps.utils import read_rules_from_database

        rules = read_rules_from_database(participant_id)
        if len(rules) == 0:
            return JsonResponse({"status": False, "message": "No rules found for the participant"}, safe=False)
        
        trees_ml_filter = Trees_ML_MixedFilter(rules, "Bayes")
        
        X_train = TrainDataSet.load_batch_by_size(participant_id, size=360)
        X_train = [item["text"] for item in X_train]
        train_results = trees_ml_filter.train_model(X=X_train)

        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = trees_ml_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "status": True,
                    "message": f"Successfully trained and tested the {system} model",
                    "data": {
                        "train_results": train_results,
                        "test_results": test_results
                    }
                }, safe=False
            )

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