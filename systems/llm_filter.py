from sharedsteps.utils import calculate_algorithm_metrics
from openai import OpenAI
from django.conf import settings
import logging
import random
import json
import threading
import time
from sharedsteps.utils import ChatCompletion

logger = logging.getLogger(__name__)

class LLMFilter:

    @classmethod
    def train(cls, system, **kwargs):
        prompts = system.read_prompts()
        if len(prompts) == 0:
            return False, "No prompts found for the participant"
        return True, LLMFilter(prompts, debug=False)

    def __init__(self, prompts, batch_size=10, debug=False, retry=False):
        """
            debug: if debug is True, we will not prompt the model to get predictions
        """
        self.debug = debug 
        self.prompts = prompts
        self.prompts = sorted(self.prompts, key=lambda x: x["priority"])
        self.BATCH_SIZE = batch_size
        self.retry = retry
        self.system_prompt = f"""
            For each text in the dataset, give a 1 (True) or 0 (False) prediction to represent whether the text satisfies the description in the rubrics. 
            Each text starts with “DATA” and a number. Both the number and the text are enclosed by “<” and “>”. 
            
            In the following, the user will provide one rubric to help you make your decision. 
            If the given rubric is completely satisfied, give a True prediction. Otherwise, give a False prediction. 
            RETURN YOUR ANSWER in the json format {{“results”: [(index, prediction), ...]}} where (index, prediction) is a tuple, index is the number of the text in the dataset, and prediction is either 1 or 0.
        """
        self.llm_client = ChatCompletion()
    
    def _generate_dataset_list(self, dataset):
        """
            Concatenate datasets in the format of 1. data1\n2.data2\n escape the double quotes for each text
        """
        dataset_list = []
        for index in range(len(dataset)):
            text = dataset[index] # we have already cleaned the text in the prepare.py by removing the double quotes
            dataset_list.append(f'DATA<{index}>: <{text}>')    
        return dataset_list
    
    def _handle_batch_prompt(self, batch, predictions, user_prompt):
        """
            @param batch: a list of texts
            @param predictions: a list of predictions
            @param user_prompt: the prompt that will be used to prompt the model
        """
        batch_str = "\n".join(batch)
        now_prompt = user_prompt + f"""\n\n\t### DATASETS: "{batch_str}","""
        while True:
            response = self.llm_client.chat_completion(self.system_prompt, now_prompt)
            response = json.loads(response or "{}")
            if "results" in response:
                results = response["results"]
                batch_preds = []
                try:
                    if type(results[0]) == dict and "index" in results[0] and "prediction" in results[0]:
                        for item in results:
                            batch_preds.append((
                                    int(item["index"]), 
                                    int(item["prediction"])
                                ))
                    elif type(results[0]) == dict and len(results[0]) == 1:
                        for item in results:
                            (index, prediction), = item.items()
                            batch_preds.append((
                                int(index), 
                                int(prediction)
                            ))
                    elif type(results[0]) == list and len(results[0]) == 2:
                        for item in results:
                            batch_preds.append((
                                int(item[0]), 
                                int(item[1])
                            ))
                    elif type(results[0]) == tuple and len(results[0]) == 2:
                        batch_preds = results
                    else:
                        logger.warning(f"response {results} is not in any expected format")
                except Exception as e:
                    logger.warning(f"reading individual predictions from results {results} raise an error: {e}")
            else:
                logger.warning(f"batch_response does not have 'results' key: {response}")

            if len(batch_preds) != len(batch):
                logger.warning(f"response length {len(results)} does not match batch length {len(batch)}\nresults: {results}")
            
            if self.retry and len(batch_preds) != len(batch):
                logger.warning(f"retrying to prompt the model with the batch again")
                continue
            elif len(batch) > 0 and len(batch_preds) / len(batch) < 0.9:
                logger.warning(f"response length {len(batch_preds)} is less than half of the batch length {len(batch)}")
                continue
            else:
                predictions += batch_preds
                return predictions
    
    def _test_prompt(self, prompt, dataset_list):
        rubric = f"Rubric: <{prompt['rubric']}>\n"

        # we assume there is only one positive and one negative example for each prompt
        if len(prompt["positives"]) > 0:
            rubric += f"\tExamples that should be marked as True: <{prompt['positives'][0]}>\n"
        if len(prompt["negatives"]) > 0:
            rubric += f"\tExamples that should be marked as False: <{prompt['negatives'][0]}>\n"
        
        user_prompt = f"""\t### RUBRIC\n\t{rubric}"""
        logger.info(f"prompt: {user_prompt}")

        predictions = []
        threads = []
        if not self.debug:
            for index in range(0, len(dataset_list), self.BATCH_SIZE):
                batch = dataset_list[index: index + self.BATCH_SIZE]
                logger.info(f"now predicting batch from index {index} to {len(batch) + index}")
                # self._handle_batch_prompt(batch, predictions, user_prompt)
                thread = threading.Thread(target=self._handle_batch_prompt, args=(batch, predictions, user_prompt))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()  
        else:
            # generate a random number either 0 or 1
            predictions = [(index, random.randint(0, 1)) for index in range(len(dataset_list))]

            
                
        if len(predictions) != len(dataset_list):
            logger.warning(f"response length {len(predictions)} does not match dataset length {len(dataset_list)}")
        
        predictions_in_dict = {}
        """
            we use a dict rather than a list to store predictions of individual prompts
            because users may remove prompts in their process and the index of the prompt may thus change
            refering the prediction of a prompt based on its index may lead to wrong predictions
        """
        
        for item in predictions:
            try:
                predictions_in_dict[item[0]] = item[1]
            except:
                logger.warning(f"item {item} is ill-formated")
    
        return predictions_in_dict
    
    def test_model(self, X, y=None):
        """
            There is no training stage for LLM. We only test the model against X, y
            Here, in order to generate explanations for each prediction, 
            instead of prompting the model with all prompts once, we prompt the model with each prompt individually and aggregate the results for the final prediction.

            Besides, it might also be possible that feeding the model with all data might overwhelm the model and lead to some idiosyncratic behavior.
            Therefore, we also feed the model with a small batch of data each time.

            @param X: a list of texts
            @param y: a list of 0, 1 representing the labels of the texts

        """
        X_test, y_test = X, y
        
        
        texts_predictions = [[] for _ in range(len(X_test))]
        
        dataset_list = self._generate_dataset_list(X_test)

        for prompt in self.prompts:
            
            prompt_id = prompt["id"]

            start_time = time.time()
            prompt_pred = self._test_prompt(prompt, dataset_list)
            end_time = time.time()
            logger.info(f"LLM model testing time for the {prompt_id}-th prompt: {end_time - start_time} seconds")

            if prompt_pred is not None:
                print(f"prompt_pred: {prompt_pred}")
                for datum_index in range(len(X_test)):
                    texts_predictions[datum_index].append({
                        "id": prompt_id,
                        "prediction": prompt_pred.get(datum_index, 0) # if no prediction is given, we use 0 as the default value
                    })

                    

        prediction = [None for _ in range(len(X_test))] # overall predictions
        for index in range(len(X_test)):
            text_pred = texts_predictions[index]
            """ 
                the final prediction is 1 if at least one prompt gives a positive prediction
            """
            prediction[index] = 1 if any([pred["prediction"] == 1 for pred in text_pred]) else 0
        
        # if the user builds the model interactively, then y_test will be None
        if y_test is not None:            
            performance = calculate_algorithm_metrics(y_test, prediction)
        else:
            performance = {}

        return {
            "prediction": prediction,
            "texts_predictions": texts_predictions,
            "performance": performance
        }