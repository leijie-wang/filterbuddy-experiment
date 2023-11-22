import json
import os
from django.conf import settings

def clean_comemnt(comment, text_only=True):
    if text_only:
        return {"text": f"{comment['comment']}"}
    else:
        return {
            "text": f"{comment['comment']}", # make sure that the enclosing quotes are double quotes
            "perspective_score": comment["perspective_score"],
            "ratings": [rating["toxic_score"] for rating in comment["ratings"]]
        }

def load_dataset(number, start=0, text_only=True):
    dataset_path = os.path.join(settings.BASE_DIR, 'datasets/toxicity_ratings.json')
    dataset = []
    counter = 0
    start_pos = 0
    with open(dataset_path) as f:    
        comment = f.readline().strip()
        while comment:
            # only start reading new commments after the start position
            if start_pos < start:
                start_pos += 1
                comment = f.readline().strip()
                continue

            comment = clean_comemnt(json.loads(comment), text_only=text_only)
            if len(comment["text"].strip()) > 0:
                dataset.append(comment)
            
            counter += 1
            if counter >= number:
                break
            comment = f.readline().strip()

    return dataset
