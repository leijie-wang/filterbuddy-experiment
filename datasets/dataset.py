import json, logging, os, sys, csv, random
from django.conf import settings
import sympy
sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))

logger = logging.getLogger(__name__)
BATCH_SIZE = 30
DATASET_SIZE = 1200
TEST_SIZE = 120

class Dataset:

    
    def __init__(self, type):
        if type == "train":
            self.dataset_path = os.path.join(settings.BASE_DIR, 'datasets/train.csv')
        elif type == "validate":
            self.dataset_path = os.path.join(settings.BASE_DIR, 'datasets/validation.csv')
        elif type == "test":
            self.dataset_path = os.path.join(settings.BASE_DIR, 'datasets/test.csv')

        self.dataset = []
        # load the dataset from a csv file
        with open(self.dataset_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                self.dataset.append(row)
        
        self.size = len(self.dataset)
        
    def load_all(self):
        return self.dataset
    
    # def load_previous_batches(self, participant_id):
    #     """
    #         load all previous batches of data for a participant
    #         @param participant_id: the id of the participant
    #     """
    #     participant, created = Participant.objects.get_or_create(participant_id=participant_id)
    #     if created:
    #         raise Exception(f"Participant {participant_id} haven't built a classifier yet")
        
            
        
    #     limit = participant.current_batch * BATCH_SIZE
    #     limit = min(limit, self.size)
    #     # map this batch to another random batch so that different participants do not see the same examples in each batch
    #     new_batch = [(participant.random_prime * i % self.size) for i in range(0, limit)]
    #     dataset = [self.dataset[i] for i in new_batch]
    #     return dataset
    def _get_random_seed(self, participant_id):
        from sharedsteps.models import Participant
        
        participant = Participant.objects.get(participant_id=participant_id)
        # checks are already done in the view
        return participant.random_seed
    
    def load_train_dataset(self, participant_id, size=DATASET_SIZE-TEST_SIZE):
        random_seed = self._get_random_seed(participant_id)
        train_ids = [(random_seed * id % self.size) for id in range(0, size)]
        train_dataset = [self.dataset[id] for id in train_ids]
        return train_dataset

    def load_test_dataset(self, participant_id):
        random_seed = self._get_random_seed(participant_id)
        test_ids = [(random_seed * id % self.size) for id in range(self.size - TEST_SIZE, self.size)]
        test_dataset = [self.dataset[id] for id in test_ids]
        return test_dataset
    
    # def load_batch(self, participant_id, start=False):
    #     """
    #         load a batch of data for a participant
    #         @param participant_id: the id of the participant
    #         @param start: whether this is the first time the participant is loading data
    #     """
    #     participant, created = Participant.objects.get_or_create(participant_id=participant_id)
        
    #     if start:
    #         participant.random_prime = sympy.randprime(1, self.size)
    #         participant.current_batch = 0
    #         participant.save()
    #         print(f"participant {participant_id} created with random prime {participant.random_prime}")

    #     limit = (participant.current_batch + 1) * BATCH_SIZE
    #     limit = min(limit, self.size)
    #     # map this batch to another random batch so that different participants do not see the same examples in each batch
    #     new_batch = [(participant.random_prime * i % self.size) for i in range(participant.current_batch * BATCH_SIZE, limit)]
    #     dataset = [self.dataset[i] for i in new_batch]
    #     participant.current_batch += 1
    #     participant.save()
    #     return dataset
    

    
