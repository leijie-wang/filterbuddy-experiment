import json
import os, sys
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import StratifiedShuffleSplit

DATASET_NAME = "toxicity_ratings.json"

TRAIN_DATASET_SIZE = 1200
VALIDATION_DATASET_SIZE = 240
TEST_DATASET_SIZE = 240
ALL_DATASET_SIZE = TRAIN_DATASET_SIZE + VALIDATION_DATASET_SIZE + TEST_DATASET_SIZE
SAMPLE_BUCKETS = 8 # we should ensure every bucket has the same number of samples

def clean_toxicity_comment(comment):
    text = comment["comment"].strip()
    # escape double quotes
    text = text.replace('"', '').replace("\\", "")
    if len(text) == 0:
        return None
    
    scores = [rating["toxic_score"] for rating in comment["ratings"]]
    return {"text": text, "score": sum(scores) / len(scores)}

def determine_split_index(bucket_size):
    if ALL_DATASET_SIZE / SAMPLE_BUCKETS > bucket_size:
        print("Warning: the bucket size is too small to split the dataset; we use ratio instead")
        train_end = int(bucket_size * TRAIN_DATASET_SIZE / ALL_DATASET_SIZE)
        valid_end = int(bucket_size * VALIDATION_DATASET_SIZE / ALL_DATASET_SIZE) + train_end
        return train_end, valid_end, bucket_size
    else:
        train_end = int(TRAIN_DATASET_SIZE/SAMPLE_BUCKETS)
        valid_end = train_end + int(VALIDATION_DATASET_SIZE/SAMPLE_BUCKETS)
        test_end = valid_end + int(TEST_DATASET_SIZE/SAMPLE_BUCKETS)
        return train_end, valid_end, test_end

def main():
    """
        We split the dataset into three parts: train, validation, and test
        We ensure that each dataset have a balanced distribution of toxicity scores, i.e., each dataset has the same number of samples in each toxicity score bucket
    """
    dataset_path = DATASET_NAME

    dataset = []
    with open(dataset_path) as f:
        for line in f:
            comment = line.strip()
            comment = clean_toxicity_comment(json.loads(comment))
            if comment is not None:
                dataset.append(comment)

    dataset = pd.DataFrame(dataset)
    print("describe the dataset:", dataset.describe())

    min_score = dataset['score'].min()
    max_score = dataset['score'].max()
    bin_width = (max_score - min_score) / SAMPLE_BUCKETS
    dataset['score_bucket'] = pd.cut(dataset['score'], bins=np.arange(min_score, max_score + bin_width, bin_width), right=False)
    
    train_set, validation_set, test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for _, group in dataset.groupby('score_bucket', observed=False):
        group = group.sample(frac=1).reset_index(drop=True) # shuffle the group

        bucket_size =  len(group)
        train_end, valid_end, test_end = determine_split_index(bucket_size)
        train_set = pd.concat([train_set, group.iloc[:train_end]])
        validation_set = pd.concat([validation_set, group.iloc[train_end:valid_end]])
        test_set = pd.concat([test_set, group.iloc[valid_end:test_end]])

    train_set = train_set.sample(frac=1).reset_index(drop=True)
    validation_set = validation_set.sample(frac=1).reset_index(drop=True)
    test_set = test_set.sample(frac=1).reset_index(drop=True)

    # remove the score_bucket column and then store the dataset into csv files
    train_set.drop(columns=['score_bucket']).to_csv("train.csv", index=False)
    validation_set.drop(columns=['score_bucket']).to_csv("validation.csv", index=False)
    test_set.drop(columns=['score_bucket']).to_csv("test.csv", index=False)
    
if __name__ == "__main__":
    main()