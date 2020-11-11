#! /usr/bin/env python3
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import sys

def svm(training, preds):
    X_train, X_test, y_train, y_test = train_test_split(training, 
                                                        preds, 
                                                        test_size=0.2, 
                                                        stratify=target)
    clf = make_pipeline(StandardScaler(),
                        svm.LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train, y_train)
    results = clf.predict(X_test)
    print(f1_score(y_test, results, pos_label=1))
    print(precision_score(y_test, results, pos_label = 1, average = 'binary'))

if __name__ == "__main__":
    X = [[0,0], [1, 1]]
    y = [0, 1]
    svm(X, y)
