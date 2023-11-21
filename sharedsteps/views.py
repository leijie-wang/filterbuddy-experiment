from django.shortcuts import render, redirect
from django.conf import settings
import logging
import json, sys, os

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))

# Create your views here.
def onboarding(request):
    return render(request, 'sharedsteps/onboarding.html')

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
        return redirect(f'/wordfilter/main?task={task}&participant_id={participant_id}')
    elif system == "basic-ML":
        pass
    elif system == "advanced-ML":
        pass
    elif system == "forest-filter":
        pass
    else:
        logging.error("Unknown system: {}".format(system))



def wordfilter(request):
    # read json datasets from dataset folder
    from datasets import toxicity
    mydataset = toxicity.load_dataset(10)
    logger.info("Loaded dataset: {}".format(mydataset))
    return render(request, 'wordfilter.html', {
        "dataset": mydataset,
    })
