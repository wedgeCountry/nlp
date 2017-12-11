
import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gc

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from collections import defaultdict

winemag = "winemag-data-130k-v2.json"
if not os.path.isfile(winemag):
    raise Exception("File not found: please download the winemag data from https://www.kaggle.com/zynicide/wine-reviews")

with open(winemag, "r") as wfile:
    wine = json.load(wfile)

how_many = 10000

df = pd.DataFrame(wine[:how_many])
#print df
print("Compiling the bag of words" )
bag = defaultdict(int) # of words
words = set()
for descr in df["description"]:
    for word in descr.split(" "):
        words.add(word)
print(words)
min_price = {word: max(df["points"]) for word in words}
max_price = {word: 0 for word in words}

#for descr, price in zip(df["description"], df["points"]):
    #print descr, price

#exit(0)

for descr, price in zip(df["description"], df["points"]):
    #row["description"]
    for word in descr:
        bag[word] += 1
        if word not in min_price:
            continue
        elif min_price[word] == 0:
            min_price[word] = price
            max_price[word] = price
        else:
            min_price[word] = min(price, min_price[word])
            max_price[word] = max(price, max_price[word])
        
bag = list(sorted(bag.items(), key=lambda x: -x[1]))
#print min_price

print("Generating features")
X = []
y = []
for index, descr in enumerate(df["description"]):
    if not df["points"][index] or np.isnan(int(df["points"][index])) \
        or np.isnan(float(df["points"][index])):
        continue
    y.append(int(df["points"][index]))
    X.append(np.array([descr.count(word) for (word, count) in bag]))
    #print X[-1]

n_train = int(0.6*max(how_many, len(y)))

X_train = np.array(X[:n_train])
y_train = np.array(y[:n_train])
X_test = np.array(X[n_train:])
y_test = np.array(y[n_train:])
print(np.shape(X_train))
print("Transform")
pca = PCA(n_components = 50)
pca.fit(X_train, y_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print(len(X_train), len(X_train[0]))
print("Training")
#forest = RandomForestClassifier()
forest = ElasticNet()
forest.fit(X_train, y_train)

for a, b in zip(forest.predict(X_test), y_test):
    print(a, b)

#print(sum( [1* (y == forest.predict(x)) for x, y in zip(X_test, y_test)] )/1. / len(y_test))
print(forest.score(X_test, y_test))
print(sum( (forest.predict(X_test) - y_test)**2))

try:
    forest.feature_importances_
except:
    exit(0)

print("Feature selection")

for i, val in enumerate(forest.feature_importances_):
    if val > 0.003:
        print(val, bag[i])
        
print(sum( (forest.predict(X_test) - y_test)**2))

#clf = ExtraTreesClassifier()
#clf = clf.fit(X, y)
#print(forest.feature_importances_)
#([ 0.04...,  0.05...,  0.4...,  0.4...])
model = SelectFromModel(forest, prefit=True)
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)

print("retraining")
new_forest = RandomForestClassifier()
new_forest.fit(X_train_new, y_train)

#print(new_forest.predict(X_test_new))
print(sum( (new_forest.predict(X_test_new) - y_test)**2))
print(len(X_train[1,:]), len(X_train_new[1,:]))
#for a, b in zip(new_forest.predict(X_test_new), y_test):
    #print(a, b)
