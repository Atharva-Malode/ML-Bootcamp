# **Linear Regression**
#
## **Multivariate Regression**
Multivariate Regression is used to predict a dependent variable based on multiple independent variables. It handles the  scenarios where there are multiple predictors. In multivariate regression, the relationship between the dependent variable and the independent variables is modeled. The model aims to find the best-fitting line or hyperplane that minimizes the difference between the predicted values and the actual values. By considering multiple predictors, multivariate regression allows for more complex and accurate predictions, capturing the relationships among multiple variables simultaneously.

The equation is,\
$y=w_0+w_1x_1+......+w_nx_n+b$

* $y$ represents the dependent variable.
* $w_0$, $w_1$,...., $w_n$ represents weight.
* $x_1$, $x_1$,...., $x_n$ represent independent variables.
* $b$ represents bias.
#
## **Logistic Regression**
Logistic regression is a statistical modeling technique used for binary classification tasks, where the goal is to predict the probability of an outcome falling into one of two classes. Unlike linear regression, logistic regression predicts the probability of an outcome rather than a continuous value.

The equation is,\
$p = \frac{1}{1 + e^{-(w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n)}}$
* $w_0$, $w_1$,...., $w_n$ represents weight.
* $x_1$, $x_1$,...., $x_n$ represent independent variables.
* $e$ denotes the base of the natural logarithm
#
## **Parameters**
Parameters are variables/parameters that change after the algorithm starts learning. They are the internal variables or settings that a model learns from the training data in order to make predictions or perform a specific task. 
> Weight\
> Bias
# 
### **Weight**
Weights are the parameters that determine the strength of the connections between input features and the output. They represent the importance or contribution of each feature in the prediction process.
# 
### **Bias**
Bias represents an additional constant value that is added to the model's predictions, independent of the input data. Adjusting the bias parameter helps align the model's predictions with the desired outcomes.
#
## **Need of Bias and Weights** 

The equation of Linear Regression is,

$y=w_0+w_1x_1+......+w_nx_n+b$

A question arises,
### **Why do we need bias *b* when we already have weights?** 
$y=wx+b$

Let's remove *b* from this equation

$y=wx$

Suppose for an example/datapoint $x=0$,

$y=w(0)$\
$y=0$

This means that when $x=0$ is given as input we get $y=0$ as output.

As we know $y$ is our predicted value but what if our expected value is not zero? That's the reason we add bias so as each and every example should never give 0 as output.

### **What is the significance of weights then?**
Weights tells the strength and direction of global optima to the gradient descent.
#
## **Hyperparameters**
Hyperparameters are variables/parameters that we have to specify before starting the training of model. They are set before training a machine learning model and are not learned directly from the training data. These parameters control the behavior of the learning algorithm and determine how the model is trained.
> Learning Rate\
> Number of iterations/Epochs\
> Cost Function
# 
### **Learning Rate**
In gradient descent, learning rate *α* specifies the size of step function taken by gradient descent in every iteration. It influences how quickly or slowly the model converges to the optimal values.
 > If the learning rate is set to be too **high**, then rather than converging, our model will pass further the point and start diverging, losing the resources.\
> If the learning rate is set to be too **low**, then our model will take a lot of time to reach the convergence or get stuck at local maxima, resulting the performance of the model. 
# 
### **Number of iterations/Epochs**
The number of iterations or epochs in machine learning refers to the number of times the model goes through the entire training dataset during the training process. Each iteration involves making predictions, calculating the error, and updating the model's parameters. setting the appropriate number of iterations requires finding a balance between sufficient learning and avoiding overfitting.
> If number of Epochs are set to be very **high** then it can lead to overfitting.\
> If number of Epochs are set to be very **low** then it can lead to underfitting.
# 
### **Cost Function**
Cost Function is a mathematical measure that quantifies the error or mismatch between the predicted values and the true values in machine learning.


Cost Function = ${\frac{1}{2m} \sum_{i=0}^{m-1} (\hat{y}_i-y_i)^2}$ 


$\hat{y}_i$ = Predicted Value\
$y_i$ = Actual Value
