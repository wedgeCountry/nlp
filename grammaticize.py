
# coding: utf-8

# In[76]:


# load data
import json
word_data = json.load(open("words.txt", "r"))

# get all grammatic functions with importance
from collections import defaultdict
grammar = defaultdict(int)
for g_list in word_data.values():
    for kind in g_list:
        grammar[kind] += 1


# In[94]:


# count how frequent a word function is
dimension_counts = sorted(grammar.items(), key = lambda x: -x[1])
# name for words with capital first letter
dimensions = ["name"] + [dim[0] for dim in dimension_counts]
# only use the first n grammar functions as feature vector
n = 8
dimensions = dimensions[:n]
print(dimensions)


# In[110]:


# convert words into a feature vectors
import numpy as np
def word_as_feature(word_function, dimensions):
    assert isinstance(word_function, dict), word_function
    word, functions = list(word_function.items())[0]
    feature = np.zeros(shape=(len(dimensions),1))
    for function in functions:
        if function in dimensions:
            feature[dimensions.index(function)] = 1
    # see if word is a name (capital first letter)
    if word and word[0] == word[0].upper() and word != word.upper():
        feature[0] = 1
    return feature
words = word_data.keys()
word_features = {word: word_as_feature({word: function}, dimensions) for word, function in word_data.items()}


# In[111]:


# load training data
with open("Alice_in_Wonderland.txt", "r") as textfile:
    text = textfile.read()
print(text[:100])
import re
text_words = re.split(r"\s+", text)
# convert training data to grammatical features
training_data = [word_features.get(word, word_as_feature({word: []}, dimensions)) for word in text_words]


# In[112]:


# how many words can we categorize?
known_words = [1 if np.sum(feature)>0 else 0 for feature in training_data]
print(np.mean(known_words))
unique_words = [1 if np.sum(feature)==1 else 0 for feature in training_data]
print(np.mean(unique_words))

