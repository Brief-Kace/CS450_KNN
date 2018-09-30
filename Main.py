from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import KNN_Classes
#import pandas
def main():
    iris = datasets.load_breast_cancer()

    data_train, data_test, targets_train, targets_test = train_test_split( iris.data, iris.target, test_size=.3,
                                                                                  random_state=56)

    knn_classifier = KNN_Classes.KNN()
    knn_classifier.fit( data_train, targets_train)
    knn_targets_predicted = knn_classifier.predict(data_test, 4, "euclidean")

    print("Accuracy KNN: ", accuracy_score(targets_test, knn_targets_predicted))
    knn_classifier.scale()

    knn_targets_predicted = knn_classifier.predict(data_test, 4, "euclidean")
    print("Accuracy Normalized: ", accuracy_score(targets_test, knn_targets_predicted))

    sklearn_classifier = KNeighborsClassifier()
    sklearn_classifier.fit(data_train, targets_train)
    knn_targets_predicted = sklearn_classifier.predict(data_test)
    print("Accuracy OOTB: ", accuracy_score(targets_test, knn_targets_predicted))


main()
