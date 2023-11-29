from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import logging
import json, sys, os

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))
BATCH_SIZE = 20

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


def load_data(number, start):
    # read json datasets from dataset folder
    from datasets import toxicity
    mydataset = toxicity.load_dataset(number, start)
    return mydataset

def load_more_data(request):
    last_length = request.GET.get('last_length', default=None)
    if last_length is not None:
        return JsonResponse(load_data(BATCH_SIZE, int(last_length)), safe=False)

def wordfilter(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'wordfilter.html', {
        "dataset": json.dumps(dataset),
    })

def examplelabel(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'examplelabel.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })

def trainML(request):
    #todo: train ML model based on user labels in the format of [(text1, 1), (text2, 0), ....]
    pass

def promptwrite(request):
    dataset = load_data(BATCH_SIZE/2, 0)
    return render(request, 'promptwrite.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })

def ruleconfigure(request):
    dataset = load_data(BATCH_SIZE, 0)
    return render(request, 'ruleconfigure.html', {
         "dataset": json.dumps(dataset),
         # use json.dumps to ensure it can be read in js
    })