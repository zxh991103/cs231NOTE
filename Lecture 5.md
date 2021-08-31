# CNN

**xiaohui zhao**

**20210828**

## convolution layer

32\*32\*3 IMAGE

5\*5\*3 FILTER $w$

biad $b$

we get n 28\*28\*1 map (n is the number of filter kernel)

low -> mid -> high feature

convolution equation:
$$
f[x,y] * g[x,y] = \sum_{n_1=-\infty}^{\infty } \sum_{n_2=-\infty}^{\infty } f[x,y] * g[x-n_1,y-n_2]
$$

### stride 
7 \* 7 input assume 3 \* 3 dilter with stride 2 -> we get 3 \* 3 output

outputsize = $\frac{(N-F)}{stride}+1$
.

### pad

commin pad 0 to the border

outputsize = $\frac{(N-F+2P)}{stride}+1$

pad can be maintain the origin input size , if not it will be quickly shrink

**fliter for the depth,the final iutput is outpustsize1\*outputsize2\*numberOfFILTER**

Example:
input volume 32\*32\*3,10 5\*5 filters (include 3 depth), stride 1 ,pad 2 ,wo have 760 parameters ( 10 \*( 5 \* 5 \*3 +1 **bias**)=760)

## pooling layer
downsample

### maxpooling

typical : not have overlap

1 1 2 4

5 6 7 8       

3 2 1 0

1 2 3 4

↓

6 8

3 4
