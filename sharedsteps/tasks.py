import re
from unittest import result
from celery import shared_task
import prompt_toolkit
from systems.llm_filter import LLMFilter
from django.conf import settings
import json
import logging


logger = logging.getLogger(__name__)

@shared_task
def train_llm_task(prompts, dataset):
    data_new = [(datum.get("predictions", None) is None) for datum in dataset] # if the data does not have the predictions key, we will treat it as new data
    old_dataset = [datum for datum, is_new in zip(dataset, data_new) if not is_new]
    new_dataset = [datum for datum, is_new in zip(dataset, data_new) if is_new]
    logger.info(f"There are {len(old_dataset)} old data and {len(new_dataset)} new data.")

    changed_prompts = [prompt for prompt in prompts if prompt["changed"]]
    unchanged_prompts = [prompt for prompt in prompts if not prompt["changed"]]
    logger.info(f"There are {len(changed_prompts)} changed prompts versus {len(unchanged_prompts)} unchanged prompts.")

    # when we do not have changed prompts, it will return a empty list
    llm_filter = LLMFilter(changed_prompts, debug=settings.LLM_DEBUG)

    results = llm_filter.test_model(X=[item["text"] for item in old_dataset], y=None)
    logger.info(f"results from testing LLM model: {results}")
    if len(unchanged_prompts) > 0:
        logger.info(f"Reading cached predictions for {len(unchanged_prompts)} unchanged prompts.")
        texts_predictions = results["texts_predictions"]
        for prompt in unchanged_prompts:
            prompt_id = prompt["id"]
            for datum_index in range(len(old_dataset)):
                datum_predictions = old_dataset[datum_index]["predictions"]
                cached_prediction = next((pred for pred in datum_predictions if pred["id"] == prompt_id), None)
                cached_prediction = cached_prediction["prediction"] if cached_prediction is not None else None
                texts_predictions[datum_index].append({
                    "id": prompt_id,
                    "prediction": cached_prediction,
                })
        results["texts_predictions"] = texts_predictions

        prediction = [None for _ in range(len(old_dataset))] # overall predictions
        for index in range(len(old_dataset)):
            text_pred = texts_predictions[index]
            for pred in text_pred:
                if pred["prediction"] is not None:
                    prediction[index] = pred["prediction"]
                    break
        
        # we treat None (not affected texts) as approved texts, which is 0
        prediction = [(0 if pred is None else pred) for pred in prediction]
        results["prediction"] = prediction


    if len(new_dataset) > 0:
        logger.info(f"Testing LLM model with {len(new_dataset)} new examples.")
        llm_filter = LLMFilter(prompts, debug=settings.LLM_DEBUG)
        new_results = llm_filter.test_model(X=[item["text"] for item in new_dataset], y=None)
        results["texts_predictions"] = results["texts_predictions"] + new_results["texts_predictions"]
        results["prediction"] = results["prediction"] + new_results["prediction"]
        
    
    return results

@shared_task
def test_llm_classifier(prompts, X, y):
    try:
        logger.info(f"Initializing LLM model with {len(prompts)} prompts.")
        llm_filter = LLMFilter(prompts, debug=settings.LLM_DEBUG, retry=True)
        logger.info(f"Testing LLM model with {len(X)} examples.")
        return llm_filter.test_model(X, y)
    except Exception as e:
        logger.warning(f"Error in test_llm_classifier: {e}")
        return {}
