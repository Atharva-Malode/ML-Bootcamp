# ML Bootcamp Week-3 Day-2🚀
#### __📍Note:__ This repository contains materials for ML Bootcamp Week-3 Day-2, focusing on K-means clustering and K-nearest neighbor (KNN) algorithms.
----

## ➤ Table of Contents 📚

- [Introduction](#➤-introduction✨)
- [K-means Clustering](#➤-k-means-clustering💫)
- [K-nearest Neighbor (KNN)](#➤-k-nearest-neighbor-knn)
- [Principal Component Analysis (PCA)](#➤-principal-component-analysis-pca)
- [Installation](#➤-installation)
- [Implementation[K-means Clustering]](#➤-implementation-k-means-clustering)
- [Implementation[K-nearest neighbor(KNN)]](#➤-implementation-k-nearest-neighbor-knn)

- [Steps for Submission]()
----

# ➤ Introduction✨

In this session, we will explore two important algorithms in machine learning: K-means clustering and K-nearest neighbor (KNN). These algorithms are widely used for various applications and provide valuable insights into data patterns and classification.

----

## ➤ K-means Clustering💫

- K-means clustering is an unsupervised learning algorithm that aims to divide a dataset into K distinct clusters. It works by iteratively assigning data points to the nearest cluster centroid and updating the centroids based on the mean of the data points in each cluster. This algorithm helps in identifying natural groupings within the data.

- In other words, K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be two clusters, and for K=3, there will be three clusters, and so on.=


<div id="header" align="center">
<img src="https://github.com/AayushPaigwar/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/images/k-means-clustering.png" alt="Logo" align= "center" width="400" height="250" />
</div>

-  The K-means Clustering algorithm has some formulas

1. The __centroid__ of a cluster is the arithmetic mean of all the data points within that cluster. It is calculated using the formula:

```
Centroid = (x̄₁, x̄₂, ..., x̄ₙ)
```

2. The __Euclidean distance formula__ is used to measure the distance between a data point and the centroid of a cluster. It helps determine the nearest centroid and assigns the data point to that cluster. The formula is as follows:

```
d(x, y) = √((x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)²)
```


## ➤ How does the K-Means Algorithm Work?

- The working of the K-Means algorithm is explained in the below steps:

```
Step-1: Select the number K to decide the number of clusters.

Step-2: Select random K points or centroids. (It can be other from the input dataset).

Step-3: Assign each data point to their closest centroid, which will form the predefined K clusters.

Step-4: Calculate the variance and place a new centroid of each cluster.

Step-5: Repeat the third steps, which means reassign each datapoint to the new closest centroid of each cluster.

Step-6: If any reassignment occurs, then go to step-4 else go to FINISH.

Step-7: The model is ready.
```

----

## ➤ K-nearest neighbor (KNN)

- K-nearest neighbor (KNN) is a simple yet powerful supervised learning algorithm used for classification and regression tasks. It predicts the class of a new data point based on the majority class of its K nearest neighbors in the training data. KNN is a versatile algorithm that can handle both numerical and categorical data.

<div id="header" align="center">
<img src="https://github.com/AayushPaigwar/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/images/knn.png" alt="Logo" align= "center" width="400" height="250" />
</div>

- The K-Nearest Neighbors (KNN) algorithm has some formulas:

1. Euclidean Distance:
The Euclidean distance formula is used to measure the distance between two points in a multi-dimensional space. It is a crucial component of the KNN algorithm for finding the nearest neighbors. The formula is as follows:

The Euclidean distance between two points (x1, y1) and (x2, y2) can be calculated using the formula:

```
√((x₂ - x₁)² + (y₂ - y₁)² + ... + (z₂ - z₁)²)
```

- Example: Suppose, we have an image of a creature that looks similar to cat and dog, but we want to know either it is a cat or dog. So for this identification, we can use the KNN algorithm, as it works on a similarity measure. Our KNN model will find the similar features of the new data set to the cats and dogs images and based on the most similar features it will put it in either cat or dog category.
 

 <div id="header" align="center">
<img src="https://github.com/AayushPaigwar/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/images/cat-dog-example-knn.png" alt="Logo" align= "center" width="400" height="250" />
</div>


## ➤ How does K-Nearest Neighbors [KNN] work?

- The KNN working can be explained on the basis of the below algorithm:
```
Step-1: Select the number K of the neighbors.

Step-2: Calculate the Euclidean distance of K number of neighbors.

Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.

Step-4: Among these k neighbors, count the number of the data points in each category.

Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.
```
----
## ➤ Principal Component Analysis (PCA)

- Principal Component Analysis is an unsupervised learning algorithm that is used for the dimensionality reduction in machine learning. 
- It is a statistical process that converts the observations of correlated features into a set of linearly uncorrelated features with the help of orthogonal transformation.
- These new transformed features are called the Principal Components. It is one of the popular tools that is used for exploratory data analysis and predictive modeling.
- It is a technique to draw strong patterns from the given dataset by reducing the variances.
 
----
### __Note:__ You need to install all the necessary libraries for the implementation

---
## ➤ Installation

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

---

# 📌 Submission Week-3 [Day-1] 

Thank you for your hard work and dedication to this project/work! To ensure a smooth submission process, please [Click Here](https://github.com/Atharva-Malode/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/Submission.md) to follow steps:


---- 

<div id="header" align="center">
<img src="https://github.com/AayushPaigwar/ML-Bootcamp/blob/master/How-to-Submit-a-Collab-File/images/mission-complete-spongebob.gif" alt="Logo" align= "center" width="150" height="150" />
</div>



----

<div align="center">

Made with ❤️ by [Aayush Paigwar](https://github.com/AayushPaigwar)

</div>
