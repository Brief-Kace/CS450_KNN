import numpy as np
import operator
from sklearn.preprocessing import StandardScaler


class KNNClassifier:
    def __init__(self, data_train, targets_train):
        self.data_train = data_train
        self.targets_train = targets_train
        self.scaler = StandardScaler()
        self.scaler_flag = False

    def predict(self, data_test, k, distance_formula):
        targets = np.array([], dtype=int)
        if self.scaler_flag:
            data_test = self.scaler.transform(data_test)
        for i in range(len(data_test)):
            distances = np.array([], dtype=int)
            for j in range(len(self.data_train)):
                distances = np.append(distances, self.compute_distance(distance_formula, data_test[i], self.data_train[j]))
            class_indeces = self.find_mins(k, distances)
            targets = np.append(targets, self.find_common_target(class_indeces))
        return targets

    def find_common_target(self, classes_indeces):
        classes = []
        for index in classes_indeces:
            classes.append(self.targets_train[index])
        classes_dict = {}
        for item in np.unique(classes_indeces):
            classes_dict[item] = 0

        for key in classes_dict.keys():
            for i in range(len(classes_indeces)):
                if classes_dict[key] == self.targets_train[classes_indeces[i]]:
                    classes_dict[key] = classes_dict[key] + 1
        single_class = self.targets_train[max(classes_dict.items(), key=operator.itemgetter(1))[0]]

        return single_class

    def find_mins(self, k, distances):
        minimums = np.array([], dtype=int)

        for i in range(k):
            for index in np.where(distances == distances.min()):
                minimums = np.append(minimums, index)
                distances[index] = distances.max()
                i += 1
        return minimums

    def compute_distance(self, distance_formula, datapoint1, datapoint2):
        if "manhattan" in distance_formula:
            distance = 0
            for i in range(len(datapoint1)):
                distance += np.absolute(datapoint1[i] - datapoint2[i])
        elif "euclidean" in distance_formula:
            distance = self.euclidean( datapoint1, datapoint2)
        else:
            print("Need valid distance formula in KNN.compute_distance()")
            exit(1)
        return distance

    def euclidean(self, datapoint1, datapoint2):
        sum = 0
        for dimension in range(len(datapoint1)):
            sum += np.power(datapoint1[dimension] - datapoint2[dimension], 2)
        return np.sqrt(sum)

    def scale(self):
        self.scaler_flag = True
        self.scaler.fit(self.data_train)
        self.data_train = self.scaler.transform(self.data_train)



class KNN:
    def fit(self, data_train, targets_train):
        self.Classifier = KNNClassifier(data_train, targets_train)
        self.predict = self.Classifier.predict
        self.scale = self.Classifier.scale
