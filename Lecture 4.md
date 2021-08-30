# NN and Backpropagation

**xiaohui zhao**

**20210828**


## computational graphs

## Backpropagation
chain rule
$$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial q}\frac{\partial q}{\partial y}$$

local gradinet * upstream gradient

max gate only the bigger number will influence the downstream

### A vectorized example:
$$f(x,W) =||W ⋅ x||^2 = \sum_{i=1}^n(W ⋅ x)_i^2$$

$$q = W \cdot x$$
$$\nabla_W f = 2q \cdot x^T$$

## NN

Assuming that
$$f = Softmax(W_2 Relu(W_1 x))$$

Nerual A have $W_1$ ,can identify the 100 feature ,like the left or right face of an animal ,the color of a car.
Nerual B have $W_2$,can get the information from A , which can identify it is an animal or a car.