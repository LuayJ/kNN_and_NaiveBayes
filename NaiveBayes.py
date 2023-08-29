import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def train(data):
    # Create new dataframes for the different labels
    label0 = pd.DataFrame()
    label1 = pd.DataFrame()

    dict0 = {}
    dict1 = {}
    sex_dict0 = {}
    sex_dict1 = {}
    cp_dict0 = {}
    cp_dict1 = {}
    fbs_dict0 = {}
    fbs_dict1 = {}
    restect_dict0 = {}
    restect_dict1 = {}
    exang_dict0 = {}
    exang_dict1 = {}
    slope_dict0 = {}
    slope_dict1 = {}
    ca_dict0 = {}
    ca_dict1 = {}
    thal_dict0 = {}
    thal_dict1 = {}

    # Assign every row to its appropriate label dataframe
    label0 = label0.append(data[data['target'] == 0])
    label1 = label1.append(data[data['target'] == 1])

    # Probabilities that the outcome is 0 or 1
    p0 = str(len(label0)) + '/' + str(len(bayes_train))
    p1 = str(len(label1)) + '/' + str(len(bayes_train))

    # Calculate the mean and std dev of each column in both classes
    for column in label0:
        mean0 = round(label0[column].mean(), 3)
        mean1 = round(label1[column].mean(), 3)

        std_dev0 = round(label0[column].std(), 3)
        std_dev1 = round(label1[column].std(), 3)

        # Drop any columns with a mean difference < 0.05
        if abs(mean0 - mean1) < 0.05:
            print(column, 'removed')
            # print(abs(mean0 - mean1))
            label0 = label0.drop(columns=column)
            label1 = label1.drop(columns=column)
            continue

        # These are the columns with discrete values
        if column == 'sex' or column == 'cp' or column == 'fbs' \
                or column == 'restecg' or column == 'exang' \
                or column == 'slope' or column == 'ca' or column == 'thal':

            # Update the dictionary for result = 0 with the probabilities of each subclass
            keys = label0[column].value_counts().keys().to_list()
            counts = label0[column].value_counts().to_list()
            for i in range(label0[column].nunique()):
                a = '0' + str(i)
                y = keys[i]
                x = counts[i]
                dict0.update({column + a: str(x) + '/' + str(len(label0))})

                if column == 'sex':
                    sex_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'cp':
                    cp_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'fbs':
                    fbs_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'restecg':
                    restect_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'exang':
                    exang_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'slope':
                    slope_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'ca':
                    ca_dict0.update({y: str(x) + '/' + str(len(label0))})
                elif column == 'thal':
                    thal_dict0.update({y: str(x) + '/' + str(len(label0))})

            # Update the dictionary for result = 1 with the probabilities of each subclass
            keys = label1[column].value_counts().keys().to_list()
            counts = label1[column].value_counts().to_list()
            for i in range(label1[column].nunique()):
                a = '1' + str(i)
                y = keys[i]
                x = counts[i]
                dict1.update({column + a: str(x) + '/' + str(len(label1))})

                if column == 'sex':
                    sex_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'cp':
                    cp_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'fbs':
                    fbs_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'restecg':
                    restect_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'exang':
                    exang_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'slope':
                    slope_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'ca':
                    ca_dict1.update({y: str(x) + '/' + str(len(label1))})
                elif column == 'thal':
                    thal_dict1.update({y: str(x) + '/' + str(len(label1))})

        # elif column =='age' or column == 'trestbps' or column == 'chol' \
        #         or column == 'thalach' or column == ' oldpeak':
        #
        #     prob0 = gaussianProb(label0[column], mean0, std_dev0)
        #     print(prob0)
        #
        #     plt.plot(label0.assign(prob=label0[column].map(label0[column].value_counts(normalize=True))))
        #     label0[column].plot.kde()
        #     label1[column].plot.kde()
        #     plt.show()

    # print(dict0)
    # print(dict1)
    # print(sex_dict0)
    # print(sex_dict1)
    # print(cp_dict0)
    # print(cp_dict1)
    # print(fbs_dict0)
    # print(fbs_dict1)
    # print(restect_dict0)
    # print(restect_dict1)
    # print(exang_dict0)
    # print(exang_dict1)

        # for i in range(len(label0)):
        #     print(label0['age'].iloc[i])
        #     prob0 = gaussianProb(label0['age'].iloc[i], mean0, std_dev0)
        #     print(prob0)


def test(data):
    for column in range(len(data)):
        print(data.iloc[column])


normal_train = normalize(bayes_train)
normal_test = normalize(bayes_test)

train(normal_train)
test(normal_test)
