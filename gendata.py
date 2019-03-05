import re
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# this function removes the POS tags from text, returning only words
def get_words_from_line(line):
    # regex to remove POS tags or punctuation marks followed by a slash
    rg = r'\/([A-Z]+|\W)'
    line = re.sub(rg, '', line)
    # replace multiple spaces with a single space
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()
    return line


# this function loads the text from file, removing POS tags
def load_text_from_file(filename):
    text_lines = []
    with open(filename, 'r') as rfile:
        for line in rfile:
            # get only words from this line; remove POS tags
            line = get_words_from_line(line)
            text_lines.append(line)

    return text_lines


# this function gets all unique words from a list of strings, for the vocabulary
def get_vocabulary(text_lines):
    vocabulary = set()
    for line in text_lines:
        words = line.split()
        vocabulary.update(words)
    
    return vocabulary


# this function generates the one-hot encoded vectors for each word in each line
def generate_one_hot_vectors(vocabulary):
    word_vectors = {}
    vector_len = len(vocabulary) + 1
    index_set_true = 0
    
    word_vec = [0] * vector_len
    word_vec[index_set_true] = 1
    index_set_true += 1
    word_vectors['init_empty'] = word_vec

    for word in vocabulary:
        word_vec = [0] * vector_len
        word_vec[index_set_true] = 1
        word_vectors[word] = word_vec
        index_set_true += 1

    return word_vectors


def generate_ngram_model(text_lines, encoded_word_vectors, n=3):
    ngram_model = []
    for line in text_lines:
        words = line.split()

        # add starting empty vector to start of list of words, as many as number of ngrams-1
        for i in range(0, n-1):
            words.insert(0, 'init_empty')
        # for i in range(len(words), n-1):
        #     words.insert(i, 'terminate_empty')
        
        start = 0
        for index in range(n-1, len(words)):
            feature_vector = []
            for i in range(start, index):
                word = words[i]
                word_vec = encoded_word_vectors[word]
                feature_vector.extend(word_vec)

            output_label = words[index]
            feature_vector.append(output_label)
            ngram_model.append(feature_vector)
            start += 1

    ngram_model = pd.DataFrame(ngram_model)
    return ngram_model


# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
text_lines = load_text_from_file(args.inputfile)

print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
    text_lines = text_lines[args.startline : args.endline]
else:
    print("Ending at last line of file.")
    text_lines = text_lines[args.startline : ]

print("Constructing {}-gram model.".format(args.ngram))
# text_lines = text_lines[:1]
print("Generating vocabulary...")
vocabulary = get_vocabulary(text_lines)
print("Generating one hot encoded vectors...")
encoded_word_vectors = generate_one_hot_vectors(vocabulary)
print("Generating ngram model...")
ngram_model = generate_ngram_model(text_lines, encoded_word_vectors, n=3)
print(ngram_model.head())

print("Writing table to {}.".format(args.outputfile))
ngram_model.to_csv(args.outputfile)
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.