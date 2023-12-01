### JSON METHODS

import json
from os.path import dirname, join
import statistics
import numpy as np

def load(filename = join(dirname(__file__), 'toxicity_ratings.json')):
  with open(filename, 'r') as f:
    lc = 0
    for line in f:
      if len(line.strip()) == 0:
        continue
      record = json.loads(line)     
      yield f'toxc-{lc}', {
        'text': record['comment'],
        'ratings': record['ratings']
      }
      
      lc += 1

def load_toxicity_ratings(record, group):

  disagreement_scores = {}
  means = {}

  # Assign every comment a disagreement score, and a label
  for key, value in record.items():
    ratings = []
    for rating in value['ratings']:
        ratings.append(rating['toxic_score'])
    score = statistics.stdev(ratings)
    disagreement_scores[key] = score

    mean = statistics.mean(ratings)
    bool_mean = int(mean > 2) # If average ratings > 2, label as 1 (toxic). Otherwise, label as 0 (non-toxic)
    means[key] = bool_mean

  # Sort keys by their disagreement scores (lowest to highest)
  keys = list(disagreement_scores.keys())
  values = list(disagreement_scores.values())
  sorted_index = np.argsort(values)
  sorted_dict = {keys[i]: values[i] for i in sorted_index}

  new_record = {}
  num = 0

  # Each entry's key is comment ID, each entry's value is [comment, label] 
  if group == "least-disagree":
      for key in sorted_dict:
          new_record['toxc-' + str(num)] = [(record[key]['text'].encode('utf-8')).decode('utf-8'), means[key]]
          num += 1
       
  else:
      for key in reversed(sorted_dict):
          new_record['toxc-' + str(num)] = [(record[key]['text'].encode('utf-8')).decode('utf-8'), means[key]]
          num += 1

  return new_record


### DATASET CLASS

import random
from collections import Counter

class Dataset:
  def __init__(self, data, index = None, rng = random.Random()):
    self.rng = rng
    self.data = data
    self.index = index

  def make_index(self, mode = 'all'):
    if mode == 'all':
      self.index = sorted(self.data.keys())

  def get(self, i):
    return self.data[self.index[i]]

  def get_comment(self, i):
    return self.data[self.index[i]]
  
  def size(self):
    return len(self.index)

  def __iter__(self):
    self._iter_temp = [(i, key) for i, key in enumerate(self.index)]
    return self

  def __next__(self):
    try:
      i, key = self._iter_temp.pop(0)
      return i, self.data[key]
    except Exception:
      raise StopIteration

def load_dataset(type = 'toxicity', size = '100'):
  """Load a dataset from a datasource

  @param type - can be toxicity
  @param task - can be one of tutorial, task-1, task-2, or task-3
  """
  if type == 'toxicity':

    rng = random.Random(1337)
    temp = {}
    i = 0
    for key, value in load():
      temp[key] = value
    data = load_toxicity_ratings(temp, 'least-disagree') # Try the comments with the least annotator disagreement
    #data = load_toxicity_ratings(temp, 'most-disagree') # Try the comments with the most annotator disagreement
    # create dataset
    shuffled = sorted(data.keys())
    rng.shuffle(shuffled)
    if size == '100':
      return Dataset(data, shuffled[:200], random.Random(3141))
    elif size == '500':
      return Dataset(data, shuffled[:1000], random.Random(3141))
    elif size == '1000':
      return Dataset(data, shuffled[:2000], random.Random(3141))
    elif size == '5000':
      return Dataset(data, shuffled[:10000], random.Random(3141))
    else:
      raise Exception(f'Unrecognized size {size}')

### ML FILTER TRAINING AND TESTING

import random
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

##from zeugma.embeddings import EmbeddingTransformer # Uncomment for embeddings

class MLFilter:

  def __init__(self, dataset, rng = random.Random(3141)):
    self.dataset = dataset
    self.train_ids = []
    self.pipe = None
    ##glove = EmbeddingTransformer('glove') # Uncomment for embeddings

  def train_model(self, force_retrain = True):
    """Train a model with the labels and print the test results on the training set

    @param force_retrain - Force the model to be replaced even if it is already trained.
    """
    # Create the training set
    X_train, y_train = [], []
    count = 0
    for item in self.dataset:
      if count < 0.5*(self.dataset.size()):
        self.train_ids.append(item[0])
        X_train.append(item[1][0]) # The comment
        y_train.append(item[1][1]) # The label, based on the average of the annotators' ratings
      count += 1

    ##X_train = glove.transform(X_train) # Uncomment for embeddings
      
    if force_retrain or self.pipe is None:
      self.pipe = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)),
        ('scaler', StandardScaler(with_mean = False)),
        ('gbc', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)) # Uncomment to use gradient boosting
        #('rfc', RandomForestClassifier(n_estimators = 100, max_depth=1, random_state=0)) # Uncomment to use random forest
      ])
      self.pipe.fit(X_train, y_train)

    acc = self.pipe.score(X_train, y_train)
    acc = acc * 100
    print(f'The model achieved an accuracy of {acc}% on the training examples')

  def test_model(self, show_pos = 10, show_neg = 10):
    """Test the model against a random subset of k

    @param show_passed - How many items that the filter lets pass are shown
    @param show_failed - How many items that the filter catches are shown
    """

    if self.pipe is None:
      print('Classifier not trained or stale! Please retrain via .train_model()')
      return
    # Shuffle randomly but the same random
    rng = random.Random(3141)

    # Find test options
    non_train_ids = [id for id, _ in self.dataset if not id in self.train_ids]
    # Shuffle
    rng.shuffle(non_train_ids)

    # Sample some
    X_test = []
    for i in non_train_ids:
      X_test.append(self.dataset.get(i)[0])
    y_pred = self.pipe.predict(X_test)

    ##X_test = glove.transform(X_test) # Uncomment for embeddings

    # Collect the predictions
    pos, neg = [], []
    for i, y in zip(non_train_ids, y_pred):
      if y > 0:
        pos.append(i)
      else:
        neg.append(i)
    print(f'A total of {len(pos)} ({len(pos) / len(non_train_ids) * 100}%) were marked [positive]')
    print(f'           {len(neg)} ({len(neg) / len(non_train_ids) * 100}%) were marked [negative]\n')
    print('Below is a sample of the top ' + str(show_pos) + ' positive items:')
    for i in pos[:show_pos]:
      print(f'   {self.dataset.get(i)[0]}')
    if len(pos[:show_pos]) == 0:
      print('N/A')

    print('\nBelow is a sample of the top ' + str(show_neg) + ' negative items:')
    for i in neg[:show_neg]:
      print(f'   {self.dataset.get(i)[0]}')
    if len(pos[:show_neg]) == 0:
      print('N/A')

    # Calculate and print performance metrics
    
    pos_pred = []
    pos_actual = []
    for i in pos:
      pos_pred.append(1)
      pos_actual.append(self.dataset.get(i)[1])

    neg_pred = []
    neg_actual = []
    for i in neg:
      neg_pred.append(0)
      neg_actual.append(self.dataset.get(i)[1])

    pred = pos_pred + neg_pred
    actual = pos_actual + neg_actual

    tp, tn, fp, fn = 0, 0, 0, 0
    accuracy, precision, recall = 0, 0, 0

    for actual, pred in zip(actual, pred):

        if actual == 1 and pred == 1:
            tp += 1

        if actual == 0 and pred == 0:
            tn += 1

        if actual == 0 and pred == 1:
            fp += 1

        if actual == 1 and pred == 0:
            fn += 1

    if tp != 0 or tn != 0 or fp != 0 or fn != 0:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy = "N/A"

    print('\n Accuracy: ' + str(round(accuracy*100, 2)) + '%')

if __name__ == '__main__':
  def test():
    dataset = load_dataset('toxicity', '5000') # Can try with 100, 500, 1000, 5000 training examples
    system = MLFilter(dataset)
    system.train_model()
    system.test_model()

  test()
