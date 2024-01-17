from django.db import models
from enum import Enum
import json 

class SYSTEMS(Enum):
    PROMPTS_LLM = "promptsLLM"
    PROMPTS_ML = "promptsML"
    EXAMPLES_ML = "examplesML"
    RULES_TREES = "rulesTrees"
    RULES_ML = "rulesML"

ACTION_CHOICES = (
    (0, 'approve'),
    (1, 'remove'),
)

STAGES = (
    ("build", "build"),
    ("update", "update"),
)

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
    stage = models.CharField(max_length=10, choices=STAGES)
    
    text = models.TextField()
    label = models.IntegerField()

    def __str__(self):
        return f"Participant {self.participant_id} labeled {self.text} as {self.label}"

class RuleConfigure(models.Model):
    participant_id = models.CharField(max_length=100)
    stage = models.CharField(max_length=10, choices=STAGES)

    name = models.TextField()
    rule_id = models.IntegerField()
    priority = models.IntegerField()
    variants = models.BooleanField()

    action = models.IntegerField(choices=ACTION_CHOICES)
    

    def __str__(self):
        return f"Participant {self.participant_id} configured a rule named {self.name} that {self.action} texts"
    
class RuleUnit(models.Model):
    rule = models.ForeignKey(RuleConfigure, related_name='units', on_delete=models.CASCADE)
    TYPE_CHOICES = (
        ('include', 'include'),
        ('exclude', 'exclude'),
    )
    
    type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    words = models.TextField()

    def __str__(self):
        return f'Units that targets texts that {self.type} {self.words}'

    def set_words(self, words_list):
        """Set words from a Python list."""
        self.words = json.dumps(words_list)
    
    def get_words(self):
        """Get words as a Python list."""
        return json.loads(self.words) if self.words else []

 
class PromptWrite(models.Model):
    participant_id = models.CharField(max_length=100)
    stage = models.CharField(max_length=10, choices=STAGES)

    prompt_id = models.IntegerField()
    rubric = models.TextField()
    priority = models.IntegerField()

    positives = models.TextField()
    negatives = models.TextField()

    ACTION_CHOICES = (
        (0, 'approve'),
        (1, 'remove'),
    )
    action = models.IntegerField(choices=ACTION_CHOICES)
    
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
    
class GroundTruth(models.Model):
    participant_id = models.CharField(max_length=100)
    text = models.TextField()
    label = models.IntegerField()
    
    DATA_TYPE = (
        ('validation', 'validation'),
        ('test', 'test'),
    )
    
    type = models.CharField(max_length=20, choices=DATA_TYPE)

    def __str__(self):
        return f"Participant {self.participant_id} labeled {self.text} as {self.label}"