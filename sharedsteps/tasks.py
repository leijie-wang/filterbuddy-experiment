from celery import shared_task
from systems.llm_filter import LLMFilter
import json
import logging

logger = logging.getLogger(__name__)

@shared_task
def train_llm_task(prompts, dataset_text_list):
    llm_filter = LLMFilter(prompts, debug=False)
    results = llm_filter.test_model(X=dataset_text_list, y=None)
    return results