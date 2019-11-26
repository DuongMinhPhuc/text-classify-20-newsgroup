# from __future__ import print_function
import csv
#/home/phucxo/Documents/python AI/ML-learn/classifying-20-newsgroups
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/train.csv')
df = shuffle(df)

train, test = train_test_split(df, test_size=0.2)


x_train = train["message"]
y_train = train['topic']
print(x_train)
print(y_train)
x_test = test['message']
y_test = test['topic']
print(x_test)
print(y_test)

#Vextorize data
print("Extracting features from the dataset using a sparse vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(encoding='latin1')
X_train = vectorizer.fit_transform(x_train)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X_train.shape)

X_test = vectorizer.transform(x_test)


def benchmark(clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f" % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X_test)
    print("done in %fs" % (time() - t0))

    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)

    # Show confusion matrix
    plt.matshow(cm)
    plt.title('Confusion matrix of the %s classifier' % name)
    plt.colorbar()
    plt.show()


print("Testbenching a linear classifier...")
parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'max_iter': 50,
    'alpha': 0.00001,
    'fit_intercept': True,
}

# benchmark(SGDClassifier, parameters, 'SGD')

parameters = {'alpha': 0.01}

benchmark(MultinomialNB, parameters, 'MultinomialNB')
