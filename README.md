# MulticlassPerceptron
A multiple-class predicting perceptron for the wine dataset: https://archive.ics.uci.edu/ml/datasets/wine
This is a very basic perceptron that has weights starting at 0 for all attributes. It simply reduces the weights of the wrong class's activation value and increases the weights of the right class's activation value during training. It does reduce the amount of weight adjustment as the number of turns increases. A more advanced version that adjusts weights by margins proportional to difference in the loss function (gradient descent) will be released soon in the future.
If this dataset has a good amount of variety, then I guess the network is half decent for the whole population of data. But the dataset's variance and bias are another verification, for another day.

If anyone can tell me how I can improve the accuracy of the network on the training data from the website above and also the accuracy on the population data, that would be immensely helpful.
Feel free to use it if you think it works.
DISCLAIMER: But if it doesn't work, don't blame because you're technically using a prototype that still requires development.
