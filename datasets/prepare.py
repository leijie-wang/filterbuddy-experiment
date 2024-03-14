from calendar import c
from email import header
from email.quoprimime import header_decode
import json
import os, sys
import csv
import comm
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import StratifiedShuffleSplit

# DATASET_NAME = "toxicity_ratings.json"
# DATASET_NAME = "jigsaw_sampled.json"
TRAIN_DATASET_SIZE = 1200
VALIDATION_DATASET_SIZE = 240
TEST_DATASET_SIZE = 240
ALL_DATASET_SIZE = TRAIN_DATASET_SIZE + VALIDATION_DATASET_SIZE + TEST_DATASET_SIZE
SAMPLE_BUCKETS = 2 # we should ensure every bucket has the same number of samples

def clean_toxicity_comment(comment):
    text = comment["comment"].strip()
    # escape double quotes
    text = text.replace('"', '').replace("\\", "")
    if len(text) == 0:
        return None
    # TODO map the toxicity score to 0 or 1
    scores = [rating["toxic_score"] for rating in comment["ratings"]]
    return {"text": text, "score": sum(scores) / len(scores)}

def clean_jigsaw_comment(comment):
    text = comment["comment"].strip()
    # escape double quotes
    text = text.replace('"', '').replace("\\", "").replace("`", "")
    if len(text) == 0:
        return None
    return {"text": text, "score": comment["toxicity"]}

def clean_youtube_comment(comment):
    text = comment[0].strip()
    # escape double quotes
    text = text.replace('"', '').replace("\\", "").replace("`", "").replace("\n", " ")
    if len(text) == 0:
        return None
    try:
        score = int(float(comment[1]) >= 0.7)
        return {"text": text, "score": score, "title": comment[2]}
    except:
        print("Error in parsing the score:", comment)
        return None

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

def main(dataset_path, clean_function, format="json"):
    """
        We split the dataset into three parts: train, validation, and test
        We ensure that each dataset have a balanced distribution of toxicity scores, i.e., each dataset has the same number of samples in each toxicity score bucket
    """

    print("I am reading from dataset:", dataset_path)
    dataset = []
    if format == "json":
        with open(dataset_path) as f:
            for line in f:
                comment = line.strip()
                comment = clean_function(json.loads(comment))
                if comment is not None:
                    dataset.append(comment)
        
    elif format == "csv":
        with open(dataset_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count < 5:
                    print(row)
                comment = clean_function(row)
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

def create_balanced_dataset(filename, clean_function, new_filename, format="csv"):
    dataset_path = f"collected_datasets/{filename}.{format}"
    dataset = []
    if format == "json":
        with open(dataset_path) as f:
            for line in f:
                comment = line.strip()
                comment = clean_function(json.loads(comment))
                if comment is not None:
                    dataset.append(comment)
        
    elif format == "csv":
        with open(dataset_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = True
            for row in csv_reader:
                if header:
                    print(row)
                    header = False
                else:
                    comment = clean_function(row)
                    if comment is not None:
                        dataset.append(comment)
                
    dataset = pd.DataFrame(dataset)
    # print("describe the dataset:", dataset["score"].describe())
    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # sample the same number of toxic and non-toxic comments, note that the number of toxic comments is less than the number of non-toxic comments

    toxic_comments = dataset[dataset["score"] > 0.5]
    non_toxic_comments = dataset[dataset["score"] <= 0.5]
    
    print(f"the length of the non-toxic comments: {len(non_toxic_comments)}; the length of the toxic comments: {len(toxic_comments)}")
    non_toxic_comments = non_toxic_comments.sample(n=len(toxic_comments)).reset_index(drop=True)
    balanced_dataset = pd.concat([toxic_comments, non_toxic_comments]).sample(frac=1).reset_index(drop=True)
    print("describe the balanced dataset:", balanced_dataset["score"].describe())

    # balanced_dataset["comment_word_len"] = balanced_dataset["text"].str.split().str.len()
    
    balanced_dataset.to_csv(f"{new_filename}.csv", index=False)



if __name__ == "__main__":
    create_balanced_dataset("youtube_gun_toxicity_clean", clean_youtube_comment, "old")
    # create_balanced_dataset("youtube_border_toxicity_clean", clean_youtube_comment, "new")
    #create_balanced_dataset("jigsaw_sampled", clean_jigsaw_comment, "tutorial", format="json")