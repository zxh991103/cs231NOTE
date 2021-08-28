# Loss Function and Optimize

**xiaohui zhao**

**20210827**

## Define Loss Function
given dataset
$${(x_i,y_i)}^N_{i=1}$$

Loss 
$$L =\frac{1}{N} \sum_i L_i(f(x_i,W),y_i)$$

MultiCLass SVM loss

"hinge loss"
$$s=f(x_i,W)$$
$$L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1)$$

### Regularization

$$L =\frac{1}{N} \sum_i L_i(f(x_i,W),y_i)+λ R(W)$$



## optimization

### 