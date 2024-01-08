from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from datasets.dataset import Dataset
import sharedsteps.utils as utils
import logging
import json, sys, os, re

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

    system = request.GET.get('system', default="promptsLLM")
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

    from sharedsteps.models import PromptWrite
    # delete the labels of the participant from the database first, for the testing purposes
    PromptWrite.objects.filter(participant_id=participant_id).delete()

    counter = 0
    for item in prompts:
        prompt = PromptWrite(rubric=item["rubric"], participant_id=participant_id, prompt_id=counter)
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
    from systems.rule_templates import RuleTemplate
    participant_id = request.GET.get('participant_id', default=None)
    if participant_id is None:
        participant_id = utils.generate_userid()
    
    system = request.GET.get('system', default="rulesTrees")
    dataset = TrainDataSet.load_batch(participant_id, start=True)
    rule_templates = RuleTemplate.get_all_schemas()
    return render(request, 'ruleconfigure.html', {
        "dataset": json.dumps(dataset),
        "participant_id": participant_id,
        "system": system,
        "rule_templates": json.dumps(rule_templates),
    })

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

    X_test = [item["text"] for item in validate_dataset]
    y_test = [item["label"] for item in validate_dataset]
    
    print(f"participant {participant_id} validates system: {system}")
    if system == "examplesML":
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
    elif system == "promptsLLM":
        from systems.llm_filter import LLMFilter
        from sharedsteps.models import PromptWrite

        prompts = list(PromptWrite.objects.filter(participant_id=participant_id).values("rubric", "positives", "negatives", "prompt_id"))
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
    elif system == "promptsML":
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


def trainLLM(request):
    from systems.llm_filter import LLMFilter
    request_data = json.loads(request.body)
    
    prompts = request_data.get('prompts')
    prompts = list(prompts.values()) # it is a dict in the frontend
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
    rules = list(rules.values()) # it is a dict in the frontend
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