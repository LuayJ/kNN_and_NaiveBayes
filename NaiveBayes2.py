import pandas as pd
import numpy as np

# from collections import Counter --> useful for counting # times something occurs

df = pd.read_csv('heart.csv')
df = df.sample(frac=1)  # Shuffle the dataset
bayes_train = df.iloc[:242, :]  # Bayes training set (80% of dataset)
bayes_test = df.iloc[242:, :]  # Bayes test set (20% of dataset)


# Normalize the data via the min-max method
def normalize(data):
    data = data.copy()
    for column in data:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    return data


# Calculates the Gaussian (Normal) Probability
def gaussianProb(x, mean, std):
    exponent = (-(1/2) * ((x - mean)**2) / (std**2))
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(exponent)


# Separates data into 0 and 1 class
def separate(data):
    separated_data = dict()
    for i in range(len(data)):
        row = data[i]
        target = row[-1]
        if target not in separated_data:
            separated_data[target] = list()
        separated_data[target].append(row)
    return separated_data


# Summarizes overall data
def data_summary(data):
    summary = [(np.mean(column), np.std(column), len(column)) for column in zip(*data)]
    del(summary[-1])
    return summary


# Summarizes each class
def class_summary(data):
    separated = separate(data)
    summary = dict()

    for target, row in separated.items():
        summary[target] = data_summary(row)

    return summary


def class_prob(summary, row):
    row_total = sum([summary[label][0][2] for label in summary])
    prob = dict()

    for target, class_sum in summary.items():
        prob[target] = summary[target][0][2] / float(row_total)
        for i in range(len(class_sum)):
            mean, std_dev, count = class_sum[i]
            # print(row[0])
            prob[target] *= gaussianProb(row[i], mean, std_dev)

    return prob


normal_train = np.array(normalize(bayes_train))
normal_test = np.array(normalize(bayes_test))

TP = TN = FP = FN = 0

summary = class_summary(normal_train)

for row in normal_test:
    prob = class_prob(summary, row)
    pred = max(prob, key=prob.get)
    print('Expected:', row[-1], 'Predicted:', pred)

    if pred == 1 and row[-1] == 1:
        TP += 1
    elif pred == 1 and row[-1] == 0:
        FP += 1
    elif pred == 0 and row[-1] == 1:
        FN += 1
    else:  # If both == 0
        TN += 1

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
