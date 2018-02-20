import numpy as np
import pandas as pd
import math
from sklearn.metrics import log_loss 
import pdb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def normalize_features(df):
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma


def compute_cost(features, labels, weights, alpha):
    m = len(labels)
    y_pred = np.dot(features, weights)
    y_pred = sigmoid(y_pred)
    negative_y_pred = sigmoid(-y_pred)
    #y_pred = classify(y_pred)

    return log_loss(labels, y_pred)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def gradient_descent(features, labels, weights, learning_rate, num_iterations, alpha):
    m = len(labels)
    cost_history = []

    for i in range(num_iterations):
	#learning_rate = learning_rate*1.2
        predicted_values = np.dot(features, weights)
	negative_predicted_labels = sigmoid(-predicted_values)
	predicted_values = sigmoid(predicted_values)
        weights = weights + learning_rate/m *(np.dot(np.transpose(features), (predicted_values - labels)))
        cost_history.append(compute_cost(features, labels, weights, alpha))
        #print cost_history
    return weights

def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def decision_boundary(prob):
    return 1 if prob >= .5 else 0

def classify(preds):
  decision_boundary_mark = np.vectorize(decision_boundary)
  flat =  decision_boundary_mark(preds).flatten()
  print flat
  return flat



def main():
    df = pd.read_csv('../Dataset/Heart_Disease_noisy _0.3_0.3.csv')
    train, test = train_test_split(df, test_size=0.2)
    df2 = pd.read_csv('../Dataset/Heart_Disease_real.csv')
    labels = df[df.columns[-1]]
    real_labels = df2[df2.columns[-1]]
    features = df[df.columns[:-1]]
    """
    labels = train[train.columns[-1]]
    real_labels = train[train.columns[-1]]
    features = train[train.columns[:-1]]
    """
    features, mu, sigma = normalize_features(features)
    learning_rate = 0.001 # please feel free to change this value
    num_iterations = 1000 # please feel free to change this value
    alpha =0.5
    features_array = np.array(features)
    labels_array = np.array(labels)
    labels_array = np.expand_dims(labels_array, axis = 1)
    weights = np.zeros((len(features.columns), 1))
    compute_cost(features_array, labels_array, weights, alpha)
    weights = gradient_descent(features_array, labels_array, weights, learning_rate, num_iterations, alpha)
    predicted_values = np.dot(features_array, weights)
    predicted_values = sigmoid(predicted_values)
    predicted_labels = classify(predicted_values)
    print(np.array(real_labels))
    print(accuracy(predicted_labels, np.array(real_labels)))

if __name__=='__main__':
    main()




