# **Linear Regression**
#
## **Parameters**
Parameters are variables/parameters that change after the algorithm starts learning.\
**Note:** **Weight** and **Bias** are the parameters of Linear Regression.
# 
The equation of Linear Regression is,

$y=w_0+w_1x_1+......+w_nx_n+b$
# 
A question arises,
### **Why do we need bias *b* when we already have weights?** 
#
$y=wx+b$

Let's remove *b* from this equation

$y=wx$
#
Suppose for an example/datapoint $x=0$,

$y=w(0)$\
$y=0$

This means that when $x=0$ is given as input we get $y=0$ as output.
#
As we know $y$ is our predicted value but what if our expected value is not zero? That's the reason we add bias so as each and every example should never give 0 as output.
#
### **What is the significance of weights then?**
Weights tells the strength and direction of global optima to the gradient descent.
#
## **Hyperparameters**
Hyperparameters are variables/parameters that we have to specify before starting the training of model.
> Learning Rate\
> Number of iterations/Epochs\
> Cost Function
# 
### **Learning Rate**
In gradient descent, learning rate *α* specifies the size of step function taken by gradient descent in every iteration.
 > If the learning rate is set to be too **high**, then rather than converging, our model will pass further the point and start diverging, losing the resources.\
> If the learning rate is set to be too **low**, then our model will take a lot of time to reach the convergence or get stuck at local maxima, resulting the performance of the model. 
# 
### **Number of iterations/Epochs**
*description*
# 
### **Cost Function**
*description*
#




