from django.conf import settings
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sympy import loggamma
from openai import OpenAI
from django.conf import settings
from sharedsteps.models import Participant, GroundTruth
import logging

logger = logging.getLogger(__name__)

class ChatCompletion:

    def __init__(self):
        self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def chat_completion(self, system_prompt, user_prompt):
        response = self.llm_client.chat.completions.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        answer = response.choices[0].message.content
    
        return answer

def check_parameters(participant_id, stage=None, system=None):
    if participant_id is None:
        return "No participant id provided"
    participant = Participant.objects.get(participant_id=participant_id)
    if participant is None:
        return "No participant found"
    if stage is not None and participant.stage != stage:
        return f"Participant is not in the correct stage {stage}"
    if system is not None and participant.system != system:
        return f"Participant is not using the assigned system {system}"
    
    return None

def get_groundtruth_dataset(participant_id, stage):

    groundtruths = GroundTruth.objects.filter(
            participant_id=participant_id, 
            stage=stage
        ).order_by('id').values('text', 'label')
    return list(groundtruths)

def save_test_results(participant_id, stage, predictions, old=False):
    """
        save the predictions of the system on the ground truth dataset from the given stage
        If the old is False, the predictions are made by the system from the same stage.
        IF the old is True and the stage is update, the predictions are made by the system from the build stage
    """
    # Retrieve the ground truth entries in the same order as the predictions
    groundtruths = GroundTruth.objects.filter(
        participant_id=participant_id, 
        stage=stage
    ).order_by('id')

    # Check if the lengths of groundtruths and results match
    if len(groundtruths) != len(predictions):
        logger.error("The number of ground truths does not match the number of predictions.")
        return

    if old:
        if stage != "update":
            logger.warning("Old predictions are only saved in the update stage.")
            return
        
        # save the predictions of the old filter on the new set of ground truths
        for groundtruth, prediction in zip(groundtruths, predictions):
            groundtruth.old_prediction = prediction
            groundtruth.save()
    else:
        # Update each ground truth with its corresponding prediction
        for groundtruth, prediction in zip(groundtruths, predictions):
            groundtruth.prediction = prediction
            groundtruth.save()

def calculate_algorithm_metrics(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # Calculate FNR and FPR
    fnr = fn / (fn + tp)  # False Negative Rate
    fpr = fp / (fp + tn)  # False Positive Rate

    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall,
        "fnr": fnr,
        "fpr": fpr
    }

def calculate_stage_performance(participant_id, stage):
    """
        This is for testing the performance of the system in a stage on the groundtruth dataset from the same stage.
    """
    groundtruths = GroundTruth.objects.filter(
        participant_id=participant_id, 
        stage=stage
    ).order_by('id').values('label', 'prediction')
    y = [groundtruth['label'] for groundtruth in groundtruths]
    y_pred = [groundtruth['prediction'] for groundtruth in groundtruths]
    return calculate_algorithm_metrics(y, y_pred)



def read_rules_from_database(participant_id, stage):
    from sharedsteps.models import RuleConfigure, RuleUnit
    
    rule_objects = RuleConfigure.objects.filter(participant_id=participant_id, stage=stage)
    if len(rule_objects) == 0:
        return []
    
    rules = []
    for rule in rule_objects:
        units = rule.units.all()
        rules.append({
            "id": rule.rule_id, # different from the id field of a django model
            "name": rule.name,
            "action": rule.action,
            "priority": rule.priority,
            "variants": rule.variants,
            "units": [
                {
                    "type": unit.type,
                    "words": unit.get_words()
                } for unit in units
            ]
        })
    return rules

def read_prompts_from_database(participant_id, stage):
    from sharedsteps.models import PromptWrite
    
    prompt_objects = PromptWrite.objects.filter(participant_id=participant_id, stage=stage)
    if len(prompt_objects) == 0:
        return []
    
    prompts = []
    for prompt in prompt_objects:
        prompts.append({
            "id": prompt.prompt_id,
            "name": prompt.name,
            "action": prompt.action,
            "priority": prompt.priority,
            "rubric": prompt.rubric,
            "positives": prompt.get_positives(),
            "negatives": prompt.get_negatives()
        })
    return prompts

def save_logs(participant_id, stage, logs):
    
    from sharedsteps.models import ExperimentLog
    for log in logs:
        try:
            log = ExperimentLog.objects.create(
                participant_id = participant_id,
                stage = stage,
                timestamp = log['timestamp'],
                time_left = log["time_left"],
                system = log["system"],
                codename = log["codename"],
                description=log["description"]
            )
            if not log.codename == "LabelExamples":
                print(str(log))
        except:
            print("Error saving log", log)