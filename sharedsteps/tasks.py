from celery import shared_task
from systems.llm_filter import LLMFilter
import json
import logging
from sharedsteps.models import SYSTEMS

logger = logging.getLogger(__name__)

@shared_task
def train_llm_task(prompts, dataset_text_list):
    llm_filter = LLMFilter(prompts, debug=True)
    results = llm_filter.test_model(X=dataset_text_list, y=None)
    return results

@shared_task
def test_llm_classifier(prompts, X, y):

    llm_filter = LLMFilter(prompts, debug=False)
    return llm_filter.test_model(X, y)
