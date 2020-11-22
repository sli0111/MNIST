## An Introduction to Digit Classification with MNIST 
November 22, 2020

### Table of Contents

* Introduction
* What is the MNIST dataset?
* Classification models: KNN, Naive Bayes, and Logistic
* Code
* Challenges
* Summary

### Introduction

In this tutorial, we will introduce the MNIST dataset and show how to use K-Nearest Neighbors (KNN), Naive Bayes, and Logisitic Regression to classify images as digits.

The inference problem is:

* given handwritten images of vector X
* classify the images as digits between 0-9 as y

<img src="https://render.githubusercontent.com/render/math?math=X = y(x))">

### What is MNIST? 

MNIST is a large dataset of handwritten images often used for image processing models.  The dataset contains over 60,000 images of 28x28 pixels.  

![dfd](https://en.wikipedia.org/wiki/MNIST_database#/media/File:MnistExamples.png)


### A Quick Review of KNN, Naive Bayes, and Logistic Regression




### Digit Classification in Code

We'll start off by loading the neccesary libraries.  

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Set the randomizer seed so results are the same each time.
np.random.seed(0)
```

Next let's load the dataset.  

```python
# Load the digit data from https://www.openml.org/d/554 or from default local location '~/scikit_learn_data/...'
X, Y = fetch_openml(name='mnist_784', return_X_y=True, cache=False)

# Rescale grayscale values to [0,1].
X = X / 255.0

# Shuffle the input: create a random permutation of the integers between 0 and the number of data points and apply this
# permutation to X and Y.
# NOTE: Each time you run this cell, you'll re-shuffle the data, resulting in a different ordering.
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

print('data shape: ', X.shape)
print('label shape:', Y.shape)

# Set some variables to hold test, dev, and training data.
test_data, test_labels = X[61000:], Y[61000:]
dev_data, dev_labels = X[60000:61000], Y[60000:61000]
train_data, train_labels = X[:60000], Y[:60000]
mini_train_data, mini_train_labels = X[:1000], Y[:1000]
```

To get a sense of the dataset, the handwritten digit images will be visualized with the imshow() function.  

```python
examples = 10
digits = 10
fig, axs = plt.subplots(nrows=digits, ncols=examples, figsize=(10, 10))
plt.rc('image', cmap='gray')

# Plots the first 10 digits from the mini_train_data set
for i in range(examples):
    for j in range(digits) :
        num_match = mini_train_data[mini_train_labels == str(j)]  # Select data for each digit
        num = num_match[np.random.randint(len(num_match))]        # Randomly select examples from the data
        num_image = np.reshape(num, (28,28))                      # Reshape 784 colums to 28x28
        
        # Plots the images of each example for each digit
        axs[j][i].imshow(num_image)
        axs[j][i].axis("off")
```

### References
https://sli0111.github.io/k-nearest-neighbors/


