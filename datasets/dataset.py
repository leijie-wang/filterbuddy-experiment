import json
import os, sys
import csv
import sympy
from django.conf import settings
from sharedsteps.models import Participant
sys.path.append(os.path.join(settings.BASE_DIR, 'datasets'))

BATCH_SIZE = 30
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
    
    def load_previous_batches(self, participant_id):
        """
            load all previous batches of data for a participant
            @param participant_id: the id of the participant
        """
        participant, created = Participant.objects.get_or_create(participant_id=participant_id)
        if created:
            raise Exception(f"Participant {participant_id} haven't built a classifier yet")
        
            
        
        limit = participant.current_batch * BATCH_SIZE
        limit = min(limit, self.size)
        # map this batch to another random batch so that different participants do not see the same examples in each batch
        new_batch = [(participant.random_prime * i % self.size) for i in range(0, limit)]
        dataset = [self.dataset[i] for i in new_batch]
        return dataset
    
    def load_batch(self, participant_id, start=False):
        """
            load a batch of data for a participant
            @param participant_id: the id of the participant
            @param start: whether this is the first time the participant is loading data
        """
        participant, created = Participant.objects.get_or_create(participant_id=participant_id)
        
        if start:
            participant.random_prime = sympy.randprime(1, self.size)
            participant.current_batch = 0
            participant.save()
            print(f"participant {participant_id} created with random prime {participant.random_prime}")

        limit = (participant.current_batch + 1) * BATCH_SIZE
        limit = min(limit, self.size)
        # map this batch to another random batch so that different participants do not see the same examples in each batch
        new_batch = [(participant.random_prime * i % self.size) for i in range(participant.current_batch * BATCH_SIZE, limit)]
        dataset = [self.dataset[i] for i in new_batch]
        participant.current_batch += 1
        participant.save()
        return dataset
    
    def load_batch_by_size(self, participant_id, size):
        participant = Participant.objects.get(participant_id=participant_id)
        if participant is None:
            raise Exception(f"Participant {participant_id} does not exist")
        
        limit = min(size, self.size)
        new_batch = [(participant.random_prime * i % self.size) for i in range(0, limit)]
        dataset = [self.dataset[i] for i in new_batch]
        return dataset
    
