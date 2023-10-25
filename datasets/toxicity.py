import json
import os
from django.conf import settings

def clean_comemnt(comment):
   return {
      "text": comment["comment"],
      "perspective_score": comment["perspective_score"],
      "ratings": [rating["toxic_score"] for rating in comment["ratings"]]
   }

def load_dataset(number):
    dataset_path = os.path.join(settings.BASE_DIR, 'datasets/toxicity_ratings.json')
    dataset = []
    counter = 0
    with open(dataset_path) as f:    
        comment = f.readline().strip()
        while comment:
            comment = json.loads(comment)
            dataset.append(clean_comemnt(comment))
            
            counter += 1
            if counter >= number:
                break
            comment = f.readline().strip()
    return dataset
