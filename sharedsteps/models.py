from django.db import models
from enum import Enum
import json 

class SYSTEMS(Enum):
    PROMPTS_LLM = "promptsLLM"
    PROMPTS_ML = "promptsML"
    EXAMPLES_ML = "examplesML"
    
class Participant(models.Model):
    participant_id = models.CharField(max_length=100)
    random_prime = models.IntegerField(null=True) # for shuffling the dataset
    current_batch = models.IntegerField(default=0) # documenting the current batch a participant is viewing
    system = models.CharField(
        max_length=100, 
        choices=[(system.name, system.value) for system in SYSTEMS],
        null=True
    )

class ExampleLabel(models.Model):
    participant_id = models.CharField(max_length=100)
    text = models.TextField()
    label = models.IntegerField()

    def __str__(self):
        return f"Participant {self.participant_id} labeled {self.text} as {self.label}"
    
class PromptWrite(models.Model):
    participant_id = models.CharField(max_length=100)
    prompt_id = models.IntegerField()
    rubric = models.TextField()
    positives = models.TextField()
    negatives = models.TextField()

    def set_positives(self, positives):
        self.positives = json.dumps(positives)

    def get_positives(self):
        return json.loads(self.positives) if self.positives else []
    
    def set_negatives(self, negatives):
        self.negatives = json.dumps(negatives)
    
    def get_negatives(self):
        return json.loads(self.negatives) if self.negatives else []

    def __str__(self):
        return f"Participant {self.participant_id} wrote the prompt {self.rubric}"