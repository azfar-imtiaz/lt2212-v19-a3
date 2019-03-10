# LT2212 V19 Assignment 3

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: Muhammad Azfar Imtiaz

## Additional instructions

The following options can be specified in the command line arguments for the `gendata` script:

- Starting line: This specifies which line of the provided text file should the data start getting accumulated from
- Ending line: This specifies till which line of the provided text file the data should be accumulated
- Testing lines: This specifies how many lines from the total number of lines selected should be used for testing purposes. The remaining lines shall be used for training.

The name of the output file provided will be split into two files, a training file and a testing file. For example, if "sampled_data" is the name of the output file provided, this shall generate "sampled_data_train.csv" and "sampled_data_test.csv". The output files generated will be in .csv format, as so to make it easier for the training and testing scripts.

Please ensure aspects such as starting line being lesser than the ending line, and number of testing lines being less than the total number of lines being used.

## Reporting for Part 4

## Reporting for Part Bonus 

For the bonus part of the assignment, I have implemented the option to use POS tags for training the model, instead of the actual words. This is specified by the `-P` command line argument. If this parameter is specified, POS tag vectors shall be generated. If it is not specified, then word vectors shall be generated.