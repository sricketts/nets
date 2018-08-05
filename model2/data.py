import numpy as np
import collections

# Text file containing words for training
training_file = 'belling_the_cat.txt'

def _read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content



def _build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

"""Returns the dictionary and reverse dictionary
"""
def dataset():
    training_data = _read_data(training_file)
    return _build_dataset(training_data)

