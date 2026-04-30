import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

np.random.seed(42)

def collec_data(n):
    X0 = 2*np.random.randn(n, 2) + np.array([1, 1])
    Y0 = np.zeros(n)

    X1 = 2*np.random.randn(n, 2) + np.array([-1, -1])
    Y1 = np.ones(n)

    X = np.vstack([X0, X1])
    Y = np.hstack([Y0, Y1])

    return np.c_[X, Y]

def normalization(train_set, test_set):
    train_set_norm = train_set.copy()
    test_set_norm = test_set.copy()
    
    for i in range(train_set.shape[1]-1):

        '''ave = np.mean(train_set[:, i])
        devi = np.std(train_set[:, i])

        train_set_norm[:, i] = (train_set[:, i] - ave) / devi
        test_set_norm[:, i] = (test_set[:, i] - ave) / devi'''

        train_set_norm[:, i] = (train_set[:, i] - np.min(train_set[:, i])) / ((np.max(train_set[:, i]) - np.min(train_set[:, i])))
        test_set_norm[:, i] = (test_set[:, i] - np.min(train_set[:, i])) / (np.max(train_set[:, i]) - np.min(train_set[:, i]))

    return train_set_norm, test_set_norm

def train_test_split(data_set, split):
    n = len(data_set)
    train_size = int(split * n)
    
    indices = np.random.permutation(n)

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train = data_set[train_idx]
    test = data_set[test_idx]

    return train, test

def cross_k_folds_validation(data_set, folds):
    n = data_set.shape[0]
    idx = np.random.permutation(n)

    fold_size = n // folds
    fold_lists = []

    for i in range(folds):
        start = i*fold_size
        end = (i+1) * fold_size if i != folds - 1 else n

        choose_idx = idx[start:end]
        fold = data_set[choose_idx]
        fold_lists.append(fold)
    
    return fold_lists
        

def calc_distance(test_row, train):
    x = np.sum(train*train, 1)
    z = np.sum(test_row*test_row)

    return x + z - 2*np.dot(train, test_row)


def get_knn(train, test_row, num_neighbour):
    distance_row = calc_distance(test_row[:2], train[:, :2])

    idx = np.argsort(distance_row)
    return idx[:num_neighbour]


def predict_knn(train, test_row, num_neighbour):
    neighbours = get_knn(train, test_row, num_neighbour)

    output_values = list(train[neighbours][:, -1])
    
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def k_nearest_neighbours(test_data, train_data, num_neighbours):
    neighbours = list()
    for test_row in test_data:
        se = predict_knn(train_data, test_row, num_neighbours)
        neighbours.append(se)
    
    return neighbours


def accuracy_metric(actual, predict):
    count = 0
    for i in range(len(actual)):
        if actual[i] == predict[i]:
            count += 1
    
    return count / float(len(actual)) * 100.0

def visualization_knn(x_train, y_train, x_test, y_test, prediction, k):
    plt.figure(figsize=(8,6))
    plt.scatter(
        x_train[y_train == 0, 0], 
        x_train[y_train == 0, 1], 
        c = "lightcoral", 
        edgecolors= "k", 
        s = 50,
        label = "Train: class 0")
    
    plt.scatter(
        x_train[y_train == 1, 0], 
        x_train[y_train == 1, 1], 
        c = "lightblue", 
        edgecolors= "k", 
        s = 50,
        label = "Train: class 1")
    
    correct_idx = (prediction == y_test)
    #incorrect_idx = (prediction != y_test)

    plt.scatter(
        x_test[correct_idx, 0],
        x_test[correct_idx, 1],
        c = "green",
        marker= "^",
        s = 80,
        edgecolors= "k",
        label = "Correct prediction"
    )

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(f'KNN Classification Results (k={k})')
    plt.legend()
    plt.show()

data_set = collec_data(100)
#folds = 3
#k_folds_set = cross_k_folds_validation(data_set, folds)

'''scores = []
for i in range(folds):

    train_set = np.concatenate(k_folds_set[:i] + k_folds_set[i+1:])
    test_set = np.array(k_folds_set[i])

    
    train_set_norm, test_set_norm = normalization(train_set, test_set)
    predicts = k_nearest_neighbours(test_set_norm, train_set_norm, 55)
    score = accuracy_metric(test_set[:, -1], predicts)

    scores.append(score)'''

train_set, test_set = train_test_split(data_set, 0.6)
train_set_norm, test_set_norm = normalization(train_set, test_set)
predicts = k_nearest_neighbours(test_set_norm, train_set_norm, 45)
scores = accuracy_metric(test_set[:, -1], predicts)

print(f"accuracy = {np.mean(scores)}%")

visualization_knn(train_set[:, :2], train_set[:, -1], test_set[:, :2], test_set[:, -1], predicts, 45)    


    

