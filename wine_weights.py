import sys
import numpy as np
import pandas as pd

def classify(row, weights):
    best_val = 0.0
    most_prob_class = list(weights.keys())[0]
    # key is class and value is corresponding list/vector of weights
    for key, value in weights.items():
        activation = 0
        for i in range(len(row)-1):
            activation += row[i]*value[i]
        if activation > best_val:
            best_val = activation
            most_prob_class = key
        else:
            continue
    return best_val, most_prob_class

def comparison_classify(row, weightlist):
    activation = 0
    for i in range(len(row)-1):
        activation += row[i]*weightlist[i]
    return activation if activation >= 0.0 else 0.0

def abs(f):
    if f < 0.0: return 0.0 - f
    else: return f

def learn_weights(csv_path):
    """Learn attribute weights for a multiclass perceptron.

    Args:
        csv_path: the path to the input file.  The data file is assumed to have a header row, and
                  the class value is found in the last column.
    Returns: a dictionary containing the weights for each attribute, for each class, that correctly
            classify the training data.  The keys of the dictionary should be the class values, and
            the values should be lists attribute weights in the order they appear in the data file.
            For example, if there are four attributes and three classes (1-3), the output might take
            the form:
                { 1 => [0.1, 0.8, 0.1000, 0.01],
                  2 => [0.9, 0.01, 0.01000, 0.4],
                  3 => [0.01, 0.2, 0.3, 0.81000] }
    """
    data = pd.read_csv(csv_path)
    weights = {}  # one set of weights for each class
    uniq_classes = pd.Series(data['Class'])
    uniq_classes = np.unique(uniq_classes)
    num_attrs = pd.Series(data.columns)
    for clas in uniq_classes:
        wts = []
        for i in range(len(num_attrs)-1):
            wts.append(0.0)
        weights[clas] = wts
    epochs = 1000
    # epoch = 0
    # while accuracy(csv_path, weights) < 0.8:
    #     epoch += 1
    for epoch in range(epochs):
        # print(weights)
        for row in data.itertuples():
            row2 = list(row)[1:len(row)]
            expected, cl = classify(row2, weights)
            if cl == row2[-1]: continue
            else:
                # print(cl, row2[-1])
                if weights[cl] == [0.0 for weight in weights[cl]] or weights[row2[-1]] == [0.0 for weight in weights[row2[-1]]]:
                    for i in range(len(row2)-1):
                        weights[cl][i] = weights[cl][i] + 0.01
                        weights[row2[-1]][i] = weights[row2[-1]][i] + 0.1
                else:
                    error = abs(comparison_classify(row2, weights[cl]) - comparison_classify(row2, weights[row2[-1]]))
                    for i in range(len(row2)-1):
                        weights[cl][i] = weights[cl][i] - row2[i]*0.001*(1.0/error)*error
                        weights[row2[-1]][i] = weights[row2[-1]][i] + row2[i]*0.001*(1.0/error)*error
    return weights

def accuracy(f, weights):
    data = pd.read_csv(f)
    num_correct = 0
    total = 0
    for row in data.itertuples():
        total += 1
        row2 = list(row)[1:len(row)]
        activation, cl = classify(row2, weights)
        if cl == row2[-1]: num_correct += 1
        else:continue
    return float(num_correct/total)

#############################################

if __name__ == '__main__':
    path_to_csv = sys.argv[1]
    class__weights = learn_weights(path_to_csv)
    for c, wts in sorted(class__weights.items()):
        print("class {}: {}".format(c, ",".join([str(w) for w in wts])))
    print(accuracy(path_to_csv, class__weights))
    # print(accuracy(path_to_csv, {'0': [0.0, 0.0, 0.0], '1': [-0.1, 0.20653640140000007, -0.23418117710000003]}))
