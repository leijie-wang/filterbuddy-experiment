import json
import os, sys
import csv
import sympy
from django.conf import settings
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
        
        self.current_batch = 0
        self.size = len(self.dataset)
        self.random_prime = sympy.randprime(1, self.size)
        print(f"now using random prime {self.random_prime}")
        

    def load_all(self, reshuffle=False):
        return self.dataset
    
    def load_batch(self, start=False, reshuffle=False):
        if start:
            self.current_batch = 0

        if reshuffle:
            self.random_prime = sympy.randprime(1, self.size)
            print(f"now using random prime {self.random_prime}")

        limit = (self.current_batch + 1) * BATCH_SIZE
        limit = min(limit, self.size)
        # map this batch to another random batch so that different participants do not see the same examples in each batch
        new_batch = [(self.random_prime * i % self.size) for i in range(self.current_batch * BATCH_SIZE, limit)]
        dataset = [self.dataset[i] for i in new_batch]
        self.current_batch += 1
        return dataset