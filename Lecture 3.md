# Loss Function and Optimize

**xiaohui zhao**

**20210827**

## Define Loss Function
given dataset
$${(x_i,y_i)}^N_{i=1}$$

### Loss 
$$L =\frac{1}{N} \sum_i L_i(f(x_i,W),y_i)$$

### MultiCLass SVM loss

"hinge loss"
$$s=f(x_i,W)$$
$$L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1)$$
### softmax

$$L_i = -log (\frac{e^{s_{y_i}}}{\sum_j e^{s_{j}}})$$

## Regularization

$$L =\frac{1}{N} \sum_i L_i(f(x_i,W),y_i)+λ R(W)$$

#### L1
$$R(W)=\sum_k \sum_l |W_{k,l}|$$

#### L2
$$R(W)=\sum_k \sum_l W_{k,l}^2$$

#### Elastic
$$R(W)=\sum_k \sum_l \beta W_{k,l}^2+|W_{k,l}|$$


## optimization

silly : random Search

GD

SGD