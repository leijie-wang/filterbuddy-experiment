from celery import shared_task
from systems.llm_filter import LLMFilter
from django.conf import settings
import json
import logging


logger = logging.getLogger(__name__)

@shared_task
def train_llm_task(prompts, dataset_text_list):
    llm_filter = LLMFilter(prompts, debug=settings.LLM_DEBUG)
    results = llm_filter.test_model(X=dataset_text_list, y=None)
    return results

@shared_task
def test_llm_classifier(prompts, X, y):
    try:
        logger.info(f"Initializing LLM model with {len(prompts)} prompts.")
        llm_filter = LLMFilter(prompts, debug=settings.LLM_DEBUG)
        logger.info(f"Testing LLM model with {len(X)} examples.")
        return llm_filter.test_model(X, y)
    except Exception as e:
        logger.warning(f"Error in test_llm_classifier: {e}")
        return {}
