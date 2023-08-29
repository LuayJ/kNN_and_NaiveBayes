import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Normalize the data via the min-max method
def normalize(data):
    data = data.copy()
    for column in data:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    return data


# Calculate the Euclidean Distance between 2 rows (x1 & x2)
def euclideanDistance(x1, x2):
    distance = 0.0
    for i in range(len(x1) - 1):
        distance += (x1[i] - x2[i])**2
    return np.sqrt(distance)


# Get k-nearest-neighbours to a given row test from the train dataset
def nearestNeighbours(train, test, k):
    distances = list()
    neighbours = list()

    for i in range(len(train)):
        temp_dist = euclideanDistance(test, train.iloc[i])
        distances.append((train.iloc[i], temp_dist))

    distances.sort(key=lambda x: x[1])  # Sorts list from nearest to furthest

    for i in range(k):
        neighbours.append(distances[i][0])

    return neighbours


# Predicts the output given the nearest neighbours
# Test should be a row, train is an entire dataset, k should be an odd #
def predict(train, test, k):
    neighbours = nearestNeighbours(train, test, k)
    neighbour_results = [result[-1] for result in neighbours]  # Returns the results of the neighbours
    # print(neighbour_results)
    # The model prediction is whatever neighbour output is greater in frequency
    prediction = max(set(neighbour_results), key=neighbour_results.count)
    return prediction


# Use validation set to find the best value of k
def validate(train, val, k, k_num_correct):
    num_correct = 0
    for row in range(len(val)):
        pred = predict(train, val.iloc[row], k)
        if pred == val.iloc[row][-1]:
            num_correct += 1
    k_num_correct.update({k: num_correct})
    return k_num_correct


def main():
    df = pd.read_csv('heart.csv')
    df = df.sample(frac=1)  # Shuffle the dataset
    kNN_train = df.iloc[:182, :]  # k-NN training set (60% of dataset)
    kNN_val = df.iloc[182:242, :]  # k-NN validation set (20% of dataset)
    kNN_test = df.iloc[242:, :]  # k-NN test set (20% of dataset)

    # Normalizing the training, validation, and test sets
    train_normal = normalize(kNN_train)
    val_normal = normalize(kNN_val)
    test_normal = normalize(kNN_test)

    TP = TN = FP = FN = 0

    k_num_correct = {}  # Stores how many correct answers the model returned per k value
    k = 1  # Initial k value, MUST BE ODD

    # Sends different odd k values to the validate function to find the best k value up to a limit
    while k <= 5:
        k_num_correct = validate(train_normal, val_normal, k, k_num_correct)
        k += 2
        # print(k_num_correct)

    # Sets k to the k value that returned the maximum correct values
    k = max(k_num_correct, key=k_num_correct.get)
    # print(k)

    for row in range(len(test_normal)):
        pred = predict(train_normal, test_normal.iloc[row], k)
        # print('Expected: ', test_normal.iloc[row][-1], ', Predicted: ', pred)

        if pred == 1 and test_normal.iloc[row][-1] == 1:
            TP += 1
        elif pred == 1 and test_normal.iloc[row][-1] == 0:
            FP += 1
        elif pred == 0 and test_normal.iloc[row][-1] == 1:
            FN += 1
        else:  # If both == 0
            TN += 1

    # print(len(test_normal))
    print(TP, FP, FN, TN)

    accuracy = (TP + TN) / (TP + FN + TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * ((precision * recall) / (precision + recall))

    conf_mat = pd.DataFrame(([TP, FP], [FN, TN]),
                            index=['Predicted Positive', 'Predicted Negative'],
                            columns=['Actual Positive', 'Actual Negative'])

    print(conf_mat, '\n')
    print('Accuracy:', accuracy, '\n'
          'Precision:', precision, '\n'
          'Recall:', recall, '\n'
          'F-Score:', f_score)


main()
