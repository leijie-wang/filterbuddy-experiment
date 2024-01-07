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
