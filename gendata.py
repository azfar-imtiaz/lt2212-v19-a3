import re
import os, sys
import glob
import argparse
import numpy as np
import pandas as pd

# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# this function removes the POS tags from text, returning only words
def get_words_from_line(line, use_pos):
    # get the word and remove anything following the slash
    if use_pos is False:
        rg = r'\/[^\s]+'
    # get the POS tag and remove anything before the slash
    else:
        rg = r'[^\s]+\/'
    line = re.sub(rg, '', line)

    # replace multiple spaces with a single space
    line = re.sub(r'\s+', ' ', line)
    line = line.strip()
    return line


# this function loads the text from file, removing POS tags
def load_text_from_file(filename, use_pos=False):
    text_lines = []
    with open(filename, 'r') as rfile:
        for line in rfile:
            # get only words from this line; remove POS tags
            line = get_words_from_line(line, use_pos)
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
import random

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("-T", "--test-lines", metavar="T", dest="testlines",
                    type=int, default=5,
                    help="How many lines in the selected number of lines should be used for testing.")
# parser.add_argument("-P", "--use-pos", metavar="P", dest='usepos', type=int, default=0,
#                     help="Decide whether the POS tags should be used for model or the text, or both. Default is False, which means text will be used. Specify 0 to use only text, specify 1 to use only POS tags, and specify 2 to use both.")
parser.add_argument("-P" "--use-pos", action="store_true", default=False, dest='usepos',
                    help="Should the POS tags be used for building and training the model, instead of the words? Default is False, which means words will be used.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

args = parser.parse_args()

# if args.usepos == 1:
#     print("Using POS tags for the model!")
# elif args.usepos == 2:
#     print("Using words and POS tags for the model!")
# else:
#     print("Using only words for the model!")
if args.usepos is True:
    print("Using POS tags for the model!")
else:
    print("Using words for the model!")

print("Loading data from file {}.".format(args.inputfile))
text_lines = load_text_from_file(args.inputfile, args.usepos)

print("Starting from line {}.".format(args.startline))
if args.endline:
    if args.startline < 0 or args.endline < 0:
        print("Please enter positive numbers for line indices")
        exit(1)
    if args.startline >= args.endline:
        print("Please ensure that the ending line is greater than the starting line")
        exit(1)
    if abs(args.startline - args.endline) <= 1:
        print("Please ensure that the difference between starting and ending line numbers if greater than 1")
        exit(1)
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

print("Creating train test split...")
print("%d lines will be used for testing" % args.testlines)
if args.testlines > len(text_lines):
    print("Please ensure that the number of lines selected for start and end is greater than the number of testing lines")
    exit(1)

# shuffle the data, then use the first -T lines for testing, and the remaining for training
random.shuffle(text_lines)
test_lines = text_lines[: args.testlines]
train_lines = text_lines[args.testlines + 1 : ]

print("Generating ngram model for training data...")
ngram_model_train = generate_ngram_model(train_lines, encoded_word_vectors, n=args.ngram)
print(ngram_model_train.head())

print("Generating ngram model for testing data...")
ngram_model_test = generate_ngram_model(test_lines, encoded_word_vectors, n=args.ngram)
print(ngram_model_test.head())

print("Writing table to {}.".format(args.outputfile))
if not args.outputfile.lower().endswith("csv"):
    print("Note: The output will be written in .csv format")

train_filename = "%s_train.csv" % args.outputfile.split('.')[0]
test_filename = "%s_test.csv" % args.outputfile.split('.')[0]

ngram_model_train.to_csv(train_filename)
ngram_model_test.to_csv(test_filename)
    
# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.