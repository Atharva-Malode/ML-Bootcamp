# ML Bootcamp Week-3 Day-1🚀
#### __📍Note:__ This repository contains materials for ML Bootcamp Week-3 Day-1, focusing on K-means clustering and K-nearest neighbor (KNN) algorithms.
----

## ➤ Table of Contents 📚

- [Introduction](#➤-introduction✨)
- [K-means Clustering](#➤-k-means-clustering)
- [K-nearest Neighbor (KNN)](#➤k-nearest-neighbor-knn)
- [Installation](#➤-installation)
- [Implementation[K-means Clustering]](#➤-implementation-k-means-clustering)
- [Implementation[K-nearest neighbor(KNN)]](#➤-implementation-k-nearest-neighbor-knn)
- [Steps for Cloning the Repository](#➤-steps-for-cloning-this-repository)
- [Steps for Submission](https://github.com/Atharva-Malode/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/Submission.md)
----

# ➤ Introduction✨

In this session, we will explore two important algorithms in machine learning: K-means clustering and K-nearest neighbor (KNN). These algorithms are widely used for various applications and provide valuable insights into data patterns and classification.

----

## ➤ K-means Clustering

K-means clustering is an unsupervised learning algorithm that aims to divide a dataset into K distinct clusters. It works by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of the data points in each cluster. This algorithm helps in identifying natural groupings within the data.



----

## ➤ K-nearest neighbor (KNN)

- K-nearest neighbor (KNN) is a simple yet powerful supervised learning algorithm used for classification and regression tasks. It predicts the class of a new data point based on the majority class of its K nearest neighbors in the training data. KNN is a versatile algorithm that can handle both numerical and categorical data.
- The K-Nearest Neighbors (KNN) algorithm some formulas:

1. Euclidean Distance:
The Euclidean distance formula is used to measure the distance between two points in a multi-dimensional space. It is a crucial component of the KNN algorithm for finding the nearest neighbors. The formula is as follows:

The Euclidean distance between two points (x1, y1) and (x2, y2) can be calculated using the formula:

```
√((x₂ - x₁)² + (y₂ - y₁)² + ... + (z₂ - z₁)²)
```

### __Note:__ You need to install all the necessary libraries for the implementation

## ➤Installation

To run the code in this repository, you need to have the following libraries installed:

➤ scikit-learn
➤ numpy
➤ pandas

To install the required libraries, open your command prompt or terminal/cell and run the following commands:

```shell
pip install scikit-learn
pip install numpy
pip install pandas
```


## ➤ Implementation [K-means Clustering]

To implement K-means clustering, you can use popular machine learning libraries such as scikit-learn or write your own code using Python. Here is an example code snippet using scikit-learn:

```python
# Import the required libraries
from sklearn.cluster import KMeans

# Create an instance of KMeans with the desired number of clusters (K)
kmeans = KMeans(n_clusters=3)

# Fit the model to the data and predict the clusters
kmeans.fit(data)
labels = kmeans.labels_
```

## ➤ Implementation [K-nearest neighbor (KNN)]

Similar to K-means clustering, you can implement KNN using libraries like scikit-learn. Here is an example code snippet for KNN classification:
```python
# Import the required libraries
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of KNeighborsClassifier with the desired value of K
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the classes for new data points
predictions = knn.predict(X_test)
```

----


 <div align="center">

Made with ❤️ by [Aayush Paigwar](https://github.com/AayushPaigwar)

</div>

