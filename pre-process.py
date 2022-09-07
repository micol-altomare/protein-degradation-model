import pickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def removeRows(data):
    """
    removes rows where the sequence contains any of 'O', 'U', 'B', 'Z', 'X' or if half life is empty
    or if half life is bigger than 100
    """
    l = list()
    for r in data:
        if int(float(r[2])+0.5) < 100 and 'O' not in r[1] and 'U' not in r[1] and 'B' not in r[1] and 'Z' not in r[1] and 'X' not in r[1] and r[2] != '':
            l.append(r)

    return l


def removeLongSequences(data, max_length):
    """
    removes sequences longer than max_length
    """
    l = list()
    for r in data:
        if len(r[1]) <= max_length:
            l.append(r)

    return l


def encodeSequencesInt(data):
    """
    encodes sequences to integers
    """
    aa_mapping = {'A': 1,
                  'C': 2,
                  'E': 3,
                  'D': 4,
                  'G': 5,
                  'F': 6,
                  'I': 7,
                  'H': 8,
                  'K': 9,
                  'M': 10,
                  'L': 11,
                  'N': 12,
                  'Q': 13,
                  'P': 14,
                  'S': 15,
                  'R': 16,
                  'T': 17,
                  'W': 18,
                  'V': 19,
                  'Y': 20,
                  'U': 21, }

    for i in range(len(data)):
        l = list()
        for j in range(len(data[i][1])): 
            l.append(aa_mapping[data[i][1][j]]) 
        data[i][1] = l

    return data


def encodeSequencesHot(data):
    """
    encodes sequences to 8- bit binary vectors
    """
    aa_mapping = {'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'E': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'G': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'F': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'I': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'H': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  'U': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  }

    for i in range(len(data)):
        l = list()
        for j in range(len(data[i][1])):
            l.append(aa_mapping[data[i][1][j]])
        data[i][1] = l
    return data


def paddingInt(data, max_length):
    """
    pads sequences to max_length with 0s
    """
    for i in range(len(data)):
        while len(data[i][1]) < max_length:
            data[i][1].append(0)
    return data


def paddingHot(data, max_length):
    """
    pads sequences to max_length with 0s
    """
    for i in range(len(data)):
        while len(data[i][1]) < max_length:
            data[i][1].append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return data


def removeDuplicates(data):
    """
    removes duplicates from data
    """
    new_data = list()
    for r in data:
        if r not in new_data:
            new_data.append(r)
    return new_data

def findIndex(data, id):
    """
    finds the index of the id in the data
    """
    for i in range(len(data)):
        if data[i][0] == id:
            return i
    return -1


if __name__ == "__main__":
    # open the csv file and read the data
    with open('hela_latest.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()

    max_length = 2000 #***

    # remove the header rows
    data.pop(0)

    # remove duplicates
    data = removeDuplicates(data)

    # remove sequences with 'O', 'U', 'B', 'Z', 'X'
    data = removeRows(data)

    # remove sequences with length bigger then max_length
    data = removeLongSequences(data, max_length)

    # find and print the index with id
    print(findIndex(data, 'Q9Y423'))

    # encode sequences to integers
    # data = encodeSequencesInt(data)
    data = encodeSequencesHot(data)

    # padding sequences with 0s
    # data = paddingInt(data, max_length)
    data = paddingHot(data, max_length)

    # only the second column is used for the sequences
    sequences = [x[1] for x in data]
    # only the third column is used for the labels and data turned to list of float
    labels = [int(float(x[2])+0.5) for x in data] #***
    
    print(len(set(labels)))

    # pickle the data
    # create a filename with the max_length in the name
    processed_seq = 'dataset%s_onehot.pickle' % max_length
    with open(processed_seq, 'wb') as f:
        pickle.dump((sequences, labels), f)
        f.close()
