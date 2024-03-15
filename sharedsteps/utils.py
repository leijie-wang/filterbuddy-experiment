from django.conf import settings
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from openai import OpenAI
from django.conf import settings
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
    from sharedsteps.models import Participant, SYSTEMS
    if participant_id is None:
        return False, "No participant id provided"
    
    participant = Participant.objects.get(participant_id=participant_id)
    if participant is None:
        return False, "No participant found"
    
    if stage is not None or system is not None:
        now_stage, now_system = participant.get_stage_system()
        if stage is not None:
            if stage not in ["build", "update"]:
                return False, f"Invalid stage: {stage}"
            if stage != now_stage:
                return False, f"The participant is at the {now_stage} stage, not the {stage} stage"
        if system is not None:
            if system not in [SYSTEMS.EXAMPLES_ML.value, SYSTEMS.RULES_TREES.value, SYSTEMS.PROMPTS_LLM.value, "FinalSurvey"]:
                return False, f"Invalid system: {system}"
            if system != now_system:
                return False, f"The participant is using the {now_system} system, not the {system} system"
    return True, ""


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

# def calculate_stage_performance(participant_id, stage):
#     """
#         This is for testing the performance of the system in a stage on the groundtruth dataset from the same stage.
#     """
#     groundtruths = GroundTruth.objects.filter(
#         participant_id=participant_id, 
#         stage=stage
#     ).order_by('id').values('label', 'prediction')
#     y = [groundtruth['label'] for groundtruth in groundtruths]
#     y_pred = [groundtruth['prediction'] for groundtruth in groundtruths]
#     return calculate_algorithm_metrics(y, y_pred)
