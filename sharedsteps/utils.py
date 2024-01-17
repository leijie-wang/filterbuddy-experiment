from django.conf import settings
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

def generate_userid():
    # generate a random user id of length 10
    import random
    import string

    # for debugging purposes, set the seed for the local generator, still keep other random generators random
    # local_random = random.Random()
    # local_random.seed(42)  

    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

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

def read_rules_from_database(participant_id):
    from sharedsteps.models import RuleConfigure, RuleUnit
    
    rule_objects = RuleConfigure.objects.filter(participant_id=participant_id)
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

def read_prompts_from_database(participant_id):
    from sharedsteps.models import PromptWrite
    
    prompt_objects = PromptWrite.objects.filter(participant_id=participant_id)
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
    return
