# Train NN PART I

**xiaohui zhao**

**20210831**

## activation function

sigmod 

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

tanh

$$tanh(x)$$

Relu

$$max(0,x)$$

LeakyRelu
$$max(0.1x,x)$$
Maxout
$$
\max \left(w_{1}^{T} x+b_{1}, w_{2}^{T} x+b_{2}\right)
$$
Elu
$$
\begin{cases}
x & x \geq 0 \\ 
\alpha\left(e^{x}-1\right) & x<0
\end{cases}
$$

SELU
$$
f(x)= \begin{cases}\lambda x & \text { if } x>0 \\ \lambda \alpha\left(e^{x}-1\right) & \text { otherwise }\end{cases}
$$

**problem for softmax:** 
* if the x is very big or small , the upstream gradient will be close to zero , which make the W less change.
* local gradient is always positive , and we assume that all the x is positive , that all $w_i$'s gradient is positive to make the zip zag path.

**problem for Relu:**
* dead Relu: will not update , like the $w\cdot x$ is always negtive so the local gradient is zero , zero multiply any is zero ,so the w will not update.

**Leaky Relu will not dead**



## 

random search

grid search