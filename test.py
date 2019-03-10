import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
test_df = pd.read_csv(args.datafile)

# drop the first column
test_df.drop(test_df.columns[0], axis=1, inplace=True)

Y = list(test_df[test_df.columns[-1]])
X = test_df.drop(test_df.columns[-1], axis=1)

print("Loading model from file {}.".format(args.modelfile))
clf = pickle.load(open(args.modelfile, 'rb'))

print("Testing model.")
predictions = clf.predict(X)

print("Accuracy is ...")
print(accuracy_score(Y, predictions))

print("Perplexity is...")
pred_probs = []
for index in range(len(X)):
    elem = X.iloc[index]
    prediction_probs = clf.predict_proba([elem])[0]
    max_prob = max(prediction_probs)
    pred_probs.append(max_prob)
entropy = sum(pred_probs) / len(pred_probs)
perplexity = 2**entropy
print(perplexity)
# this could be the prediction probabilities of all elements in the test set, summed up and 
# divided by total number of samples. This gives entropy. 2^entropy = perplexity. Not sure though