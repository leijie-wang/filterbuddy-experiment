from django.shortcuts import render
from django.conf import settings
import logging
import json, sys, os

logger = logging.getLogger(__name__)
# Create your views here.

sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))

def main(request):
    # read json datasets from dataset folder
    from datasets import toxicity
    mydataset = toxicity.load_dataset(10)
    logger.info("Loaded dataset: {}".format(mydataset))
    return render(request, 'wordfilter/main.html', {
        "dataset": mydataset,
    })