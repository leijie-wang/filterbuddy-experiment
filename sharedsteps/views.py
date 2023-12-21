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
    return JsonResponse(json.dumps(TrainDataSet.load_batch()), safe=False)

def examplelabel(request):
    participant_id = utils.generate_userid() 
    # used as the identifier of the participant; it should be passed as a get parameter but now we just generate it randomly
    dataset = TrainDataSet.load_batch(start=True, reshuffle=True)
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
    return JsonResponse({"response": 200}, safe=False)

   
def validate_page(request):
    # parse out the participant id from the request GET parameters
    # participant_id = request.GET.get('participant_id', default=None)
    participant_id = utils.generate_userid()
    if participant_id is not None:
        dataset = ValidateDataSet.load_all()
        return render(request, 'validate.html', {
            "dataset": json.dumps(dataset),
            "participant_id": participant_id,
        })

def validate_system(request):
    request_data = json.loads(request.body)
    participant_id = request_data.get('participant_id')
    system = request_data.get('system')
    validate_dataset = request_data.get('dataset')
    print(f"participant {participant_id} validates system: {system}")
    if system == "examples+ML":
        # retrieve the labels of the examples labeled by the participant from the database
        from sharedsteps.models import ExampleLabel
        from sharedsteps.ml_filter import MLFilter

        training_dataset = list(ExampleLabel.objects.filter(participant_id=participant_id).values("text", "label"))
        if len(training_dataset) == 0:
            return JsonResponse({"response": 400, "message": "No labels found for the participant"}, safe=False)
        

        ml_filter = MLFilter("Bayes")
        X_train = [item["text"] for item in training_dataset] 
        y_train = [item["label"] for item in training_dataset]

        X_test = [item["text"] for item in validate_dataset]
        y_test = [item["label"] for item in validate_dataset]

        print(f"starting training with {len(X_train)} examples labeled by the participant")
        train_results = ml_filter.train_model(X=X_train, y=y_train)

        print(f"starting testing with {len(X_test)} examples labeled by the participant")
        test_results = ml_filter.test_model(X=X_test, y=y_test)
        return JsonResponse({
                    "response": 200, 
                    "train_results": train_results,
                    "test_results": test_results
                }, safe=False
            )






def promptwrite(request):
    dataset = TrainDataSet.load_batch(start=True)
    return render(request, 'promptwrite.html', {
         "dataset": json.dumps(dataset),
    })

def trainLLM(request):
    # parse POST data
    request_data = json.loads(request.body)
    
    prompts = request_data.get('prompts')
    dataset = request_data.get('dataset')
    print(f"prompts: {prompts}")
    
    predictions = [{"total_prediction": None, "prompt_predictions": {}} for _ in range(len(dataset))]
    # concatenate datasets in the format of 1. data1\n2.data2\n escape the double quotes for each text
    dataset_str = []
    for index in range(len(dataset)):
        text = dataset[index]["text"].replace('"', '\\"')
        dataset_str.append(f'DATA<{index}>: <{text}>')    
    dataset_str = "\n".join(dataset_str)

    system_prompt = f"""
        For each text in the dataset, you task is to give a 1 (True) or 0 (False) prediction that represents whether the text satisfies the description in the overview and the rubrics.
        Each text starts with "DATA" and a number. Both the number and the text are enclosed by "<" and ">".

        In the following, the user will provide one rubric to help you make your decision. 
        It might be associated with some examples that should be caught and some examples that should not be caught for you to better understand the rubric.
        As long as the given rubric is satisfied, you should give a True prediction. Otherwise, give a False prediction.

        RETURN YOUR ANSWER in the json format {{"results": [(index, prediction), ...]}} where index is the index of the text in the dataset and prediction is either 1 or 0.
    """

    for index in range(len(prompts)):
        prompt = prompts[index]
        rubric = f"Rubric: <{prompt['rubric']}>\n"
        if len(prompt["positives"]) > 0:
            rubric += f"\tExamples that should be caught: <{prompt['positives'][0]}>\n"
        if len(prompt["negatives"]) > 0:
            rubric += f"\tExamples that should not be caught: <{prompt['negatives'][0]}>\n"
        
        user_prompt = f"""\t### RUBRIC\n\t{rubric}"""

        print(f"prompt: {user_prompt}")
        user_prompt += f"""\n\n\t### DATASETS: "{dataset_str}","""
        # response = json.loads(utils.chat_completion(system_prompt, user_prompt))
        # generate a random number either 0 or 1
        
        import random
        response = {"results": [(index, random.randint(0, 1)) for index in range(len(dataset))]}
    
        parsed_response = {}
        """
            we use a dict rather than a list to store predictions of individual prompts
            because users may remove prompts in their process and the index of the prompt may thus change
            refering the prediction of a prompt based on its index may lead to wrong predictions
        """
        if "results" in response:
            for item in response["results"]:
                parsed_response[item[0]] = item[1]
            for index in range(len(dataset)):
                predictions[index]["prompt_predictions"][prompt["id"]] = parsed_response[index] if index in parsed_response else None

    for prediction in predictions:
        # aggregate individual predictions using or operation, do not consider None
        valid_prediction = [value for value in prediction["prompt_predictions"].values() if value is not None]
        prediction["total_prediction"] = any(valid_prediction) if len(valid_prediction) > 0 else None
    print(f"response: {predictions}")
    return JsonResponse({"response": 200, "answer": predictions}, safe=False)

def ruleconfigure(request):
    dataset = TrainDataSet.load_batch(start=True)
    return render(request, 'ruleconfigure.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })