from openai import OpenAI
from django.conf import settings
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
def chat_completion(system_prompt, user_prompt):
    response = llm_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
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

def generate_userid():
    # generate a random user id of length 10
    import random
    import string
    # for debugging purposes, set the seed for the local generator, still keep other random generators random
    local_random = random.Random()
    local_random.seed(42)  

    return ''.join(local_random.choices(string.ascii_uppercase + string.digits, k=10))

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
