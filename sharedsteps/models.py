from os import name
from django.db import models
from enum import Enum
import json, random, sympy, string, logging
from datasets.dataset import DATASET_SIZE
from systems.ml_filter import MLFilter
from systems.trees_filter import TreesFilter
from systems.llm_filter import LLMFilter

logger = logging.getLogger(__name__)

class SYSTEMS(Enum):
    PROMPTS_LLM = "promptsLLM"
    PROMPTS_ML = "promptsML"
    EXAMPLES_ML = "examplesML"
    RULES_TREES = "rulesTrees"
    RULES_ML = "rulesML"

ACTION_CHOICES = [
    (0, 'approve'),
    (1, 'remove'),
]

STAGES = [
    ("build", "build"),
    ("update", "update"),
]

code2name_dict = {"A": "examplesML", "B": "rulesTrees",  "C": "promptsLLM"}
name2code_dict = {"examplesML": "A", "rulesTrees": "B", "promptsLLM": "C"}
def code2name(code):
    return code2name_dict[code]

def name2code(name):
    return name2code_dict[name]

class Participant(models.Model):
    participant_id = models.CharField(max_length=100)
    random_seed = models.IntegerField(null=True)
    group = models.CharField(max_length=100, null=True)
    """
        which group the participant is in, this determines the order of the conditions; 
        we represent the example group as A, the rule group as B, and the prompt group as C
        the oder of the ABC is then the order of the conditions
        for example, if the participant is in the group CBA, then the order of the conditions is promptsLLM, rulesTrees, examplesML
    """

    progress = models.IntegerField(null=True)
    # regardless of the group, which stage of the 8 stages the participant is in right now

    @classmethod
    def generate_userid(cls):
        """
            generate a random user id of length 10
        """

        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    
    @classmethod
    def create_participant(cls, group):
        participant = Participant.objects.create(
            participant_id = Participant.generate_userid(),
            random_seed = sympy.randprime(1, DATASET_SIZE),
            group = group,
            progress = 1
        )
        participant.save()
        participant.create_conditions()
        return participant
    
    def create_conditions(self):
        for index in range(3):
            system_code = self.group[index]
            system_name = code2name(system_code)
            condition = Condition.objects.create(
                participant = self,
                system_name = system_name,
                order = index,
                stage = "build"
            )
            condition.save()

    def update_conditions(self):
        conditions = self.conditions.all()
        for condition in conditions:
            condition.stage = "update"
            condition.save()
        

    def get_stage_system(self):
        stage = "build" if self.progress < 5 else "update"
        if self.progress == 1 or self.progress == 5:
            system = "GroundTruth"
        else:
            system_index = self.progress - 2 if self.progress < 5 else self.progress - 6 # turn to 0, 1, 2
            system_code = self.group[system_index]
            system = code2name(system_code)
        return stage, system
    
    def get_progress(self, stage, system):
        if stage == "build":
            return 1 if system == "GroundTruth" else 2 + self.group.find(name2code(system))
        else:
            return 5 if system == "GroundTruth" else 6 + self.group.find(name2code(system))
        
    def validate_stage_system(self, stage, system):
        now_stage, now_system = self.get_stage_system()
        return now_stage == stage and now_system == system
    
    def update_progress(self, stage=None, system=None):
        if stage is not None and system is not None:
            requested_progress = self.get_progress(stage, system)
            if requested_progress != self.progress:
                # users are either refreshing this page again or trying to access a page they are not supposed to
                logger.warning(f"Participant {self.participant_id} is trying to access the {requested_progress}th stage, but they are at the {self.progress}th stage")
                return
            else:
                self.progress += 1
                if self.progress >= 5:
                    self.update_conditions()
                self.save()

    def get_condition(self, system):
        return self.conditions.get(system_name=system)
    
    def __str__(self):
        return f"{str(self.condition)}: {self.timestamp}\n{self.codename}\t{self.description[:100]}\n"
    
class Condition(models.Model):
    participant = models.ForeignKey(Participant, related_name='conditions', on_delete=models.CASCADE)
    system_name = models.CharField(max_length=100, choices=[(system.name, system.value) for system in SYSTEMS], null=True)

    order = models.IntegerField() # in a within-subject design, the order of the condition
    stage = models.CharField(max_length=10, choices=STAGES) # which stage the participant is in right now
    
    def __str__(self):
        return f"{self.system}"
    
    def save_logs(self, logs):
        # save the log will only be called once, so we can delete all the logs related to this condition
        experiment_logs = ExperimentLog.objects.filter(condition=self)
        log_count = experiment_logs.count()
        if log_count > 0:
            logger.info(f"Deleting {log_count} logs related to condition {self.participant.participant_id}-{self.stage}-{self.system_name}")
            experiment_logs.delete()

        for log in logs:
            try:
                log = ExperimentLog.objects.create(
                    condition = self,
                    timestamp = log['timestamp'],
                    time_left = log["time_left"],
                    codename = log["codename"],
                    description=log["description"]
                )
            except Exception as e:
                logger.warning("Error saving log", log)
                logger.warning(e)
    
    def create_system(self, spent_time, stage):
        repeated_system = System.objects.filter(condition=self, spent_time=spent_time, stage=stage)
        if repeated_system.count() > 0:
            logger.warning(f"System {self.system_name} has already been created at {spent_time} seconds in the {stage} stage")
            return repeated_system.first()
        
        logger.info(f"Success creating system {self.system_name} at {spent_time} seconds in the {stage} stage")
        system = System.objects.create(
            condition = self,
            spent_time = spent_time,
            stage = stage
        )
        system.save()
        return system

    def get_groundtruth_dataset(self, stage):
        return list(GroundTruth.objects.filter(condition=self, stage=stage).order_by('id').values('text', 'label', 'datum_id'))
    
    def get_latest_system(self, stage):
        # there might not be a system in the given stage
        system = self.systems.filter(stage=stage).first()
        if system is None and stage == "update":
            system = self.systems.filter(stage="build").first()
        return system
    
    def save_test_results(self, stage, predictions, old=False):
        """
            save the predictions of the system on the ground truth dataset from the given stage
            If the old is False, the predictions are made by the system from the same stage.
            IF the old is True and the stage is update, the predictions are made by the system from the build stage
        """
        # Retrieve the ground truth entries in the same order as the predictions
        groundtruths = GroundTruth.objects.filter(condition=self, stage=stage).order_by('id')

        # Check if the lengths of groundtruths and results match
        if len(groundtruths) != len(predictions):
            logger.error("The number of ground truths does not match the number of predictions.")
            return

        if old:
            if stage != "update":
                logger.error("Old predictions are only saved in the update stage.")
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
    
    def get_time_spent(self, stage):
        system = self.systems.filter(stage=stage).first()
        if system is not None:
            return system.spent_time
        else:
            return 0

class ExperimentLog(models.Model):
    condition = models.ForeignKey(Condition, related_name='logs', on_delete=models.CASCADE)
    
    # use GMT-0 time
    timestamp = models.DateTimeField()
    time_left = models.IntegerField()
    codename = models.CharField(max_length=100)
    description = models.TextField()
    
class System(models.Model):
    condition = models.ForeignKey(Condition, related_name='systems', on_delete=models.CASCADE)
    spent_time = models.IntegerField() # how much time users spend creating this system
    stage = models.CharField(max_length=10, choices=STAGES) # which stage when the system is created

    class Meta:
        # when querying the system related to a condition,  they are returned with the most recent one first
        ordering = ['-spent_time'] # in increasing order of created_at
    
    def __str__(self):
        return f"created at {self.spent_time} seconds in the {self.stage} stage"
    
    def read_examples(self, **kwargs):
        """
            read labeled examples from the database related to this system
        """
        if self.condition.system_name == SYSTEMS.EXAMPLES_ML.value:
            return list(ExampleLabel.objects.filter(system=self).values("text", "label", "datum_id"))
        else:
            logger.error(f"System {self.condition.system_name} does not have examples")
            return []
    
    def read_rules(self, **kwargs):
        """
            read rules from the database related to this system
        """
        if self.condition.system_name == SYSTEMS.RULES_TREES.value:
            rule_objects = RuleConfigure.objects.filter(system=self)
            if len(rule_objects) == 0:
                return []
            
            rules = []
            for rule in rule_objects:
                units = rule.units.all()
                rules.append({
                    "id": rule.rule_id, # different from the id field of a django model
                    "name": rule.name, "action": rule.action,
                    "priority": rule.priority, "variants": rule.variants,
                    "units": [
                        {"type": unit.type, "words": unit.get_words()} for unit in units
                    ]
                })
            return rules
        else:
            logger.error(f"System {self.condition.system_name} does not have rules")
            return []

    def read_prompts(self, **kwargs):
        """
            read prompts from the database related to this system
        """
        if self.condition.system_name == SYSTEMS.PROMPTS_LLM.value:
            prompt_objects = PromptWrite.objects.filter(system=self)
            if len(prompt_objects) == 0:
                return []
            
            prompts = []
            for prompt in prompt_objects:
                prompts.append({
                    "id": prompt.prompt_id,
                    "name": prompt.name, "action": prompt.action,
                    "priority": prompt.priority, "rubric": prompt.rubric,
                    "positives": prompt.get_positives(), "negatives": prompt.get_negatives()
                })
            return prompts
        else:
            logger.error(f"System {self.condition.system_name} does not have prompts")
            return []
    
    def save_examples(self, **kwargs):
        dataset = kwargs["dataset"]
        for item in dataset:
            ExampleLabel(system=self, datum_id=item["datum_id"], text=item["text"], label=item["label"]).save()
    
    def save_rules(self, **kwargs):
        rules = kwargs["rules"]

        counter = 0
        for item in rules:
            rule = RuleConfigure(system=self, name=item["name"], action=item["action"], 
                                 rule_id=counter, priority=item["priority"], variants=item["variants"])
            rule.save()
            for unit in item["units"]:
                rule_unit = RuleUnit(type=unit["type"], rule=rule)
                rule_unit.set_words(unit["words"])
                rule_unit.save()

            counter = counter + 1
    
    def save_prompts(self, **kwargs):
        prompts = kwargs["prompts"]

        counter = 0
        for item in prompts:
            prompt = PromptWrite(system=self, name=item["name"], action=item["action"], 
                                 prompt_id=counter, priority=item["priority"], rubric=item["rubric"])
            prompt.set_positives(item["positives"])
            prompt.set_negatives(item["negatives"])
            prompt.save()
            counter = counter + 1

    def save_system(self, **kwargs):
        if self.condition.system_name == SYSTEMS.EXAMPLES_ML.value:
            self.save_examples(**kwargs)
        elif self.condition.system_name == SYSTEMS.RULES_TREES.value:
            self.save_rules(**kwargs)
        elif self.condition.system_name == SYSTEMS.PROMPTS_LLM.value:
            self.save_prompts(**kwargs)

    def train(self, **kwargs):
        classifier_class = None
        system_name = self.condition.system_name
        if system_name == SYSTEMS.EXAMPLES_ML.value:
            classifier_class = MLFilter
        elif system_name == SYSTEMS.PROMPTS_LLM.value:
            classifier_class = LLMFilter
        elif system_name == SYSTEMS.RULES_TREES.value:
            classifier_class = TreesFilter
       
        return classifier_class.train(self, **kwargs)
        
class ExampleLabel(models.Model):
    system = models.ForeignKey(System, related_name='examples', on_delete=models.CASCADE)
    datum_id = models.IntegerField()
    text = models.TextField()
    label = models.IntegerField()

    def __str__(self):
        return f"{self.text} is labeled as {self.label}"

class RuleConfigure(models.Model):
    system = models.ForeignKey(System, related_name='rules', on_delete=models.CASCADE)
    name = models.TextField()
    rule_id = models.IntegerField()
    priority = models.IntegerField()
    variants = models.BooleanField() # whether spelling variants are considered
    action = models.IntegerField(choices=ACTION_CHOICES) # whether the rule is to approve or remove texts
    

    def __str__(self):
        return f"A rule named {self.name} that {self.action} texts"
    
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
    system = models.ForeignKey(System, related_name='prompts', on_delete=models.CASCADE)
    
    name = models.TextField()
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
    condition = models.ForeignKey(Condition, related_name='groundtruths', on_delete=models.CASCADE)
    stage = models.CharField(max_length=10, choices=STAGES)

    datum_id = models.IntegerField()
    text = models.TextField()
    label = models.IntegerField()
    prediction = models.IntegerField(null=True)
    old_prediction = models.IntegerField(null=True)

    def __str__(self):
        return f"The text {self.text} is labled as {self.label}"
    
    @classmethod
    def save_groundtruth(cls, participant, stage, dataset):
        conditions = participant.conditions.all()
        for condition in conditions:
            GroundTruth.objects.filter(condition=condition, stage=stage).delete()
            for datum in dataset:
                GroundTruth(condition=condition, stage=stage, datum_id=datum["datum_id"] , text=datum["text"], label=datum["label"]).save()