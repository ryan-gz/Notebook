## Sinusoidal Position Embedding



苏剑林 https://spaces.ac.cn/

#### 1. 原文介绍

在Transformer 论文原文中，所使用的位置编码为

![image-20221021112029232](/Users/tomo/Library/Application Support/typora-user-images/image-20221021112029232.png)

在这里，$$p_{k,2i}$$，$$p_{k,2i+1}$$ 分别是位置 *pos* 的编码向量的第2*i*, 2*i* + 1个分量，即每个pos的奇数分量用sin偶数分量用cos，*$$d_{model}$$* 是向量的维度，下图以(10,64) 的张量为例，可以看到只有前20多个位置编码的表征是有信息量的：

![image-20230616164401489](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164401489.png)

在不考虑位置信息的情况下，文本成分之间的注意力是完全对称的，也就是说，所谓的注意力机制是无法学习到上下关系的，我们无法从结果上区分句子成分的顺序。

<img src="https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164419412.png" alt="image-20230616164419412" style="zoom:50%;" />

这种全对称性需要添加一个新的编码来打破。比如在每个位置上都加上一个不同的编码向量：

<img src="https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164432669.png" alt="image-20230616164432669" style="zoom:50%;" />

#### 2. 倒推论文的二维位置编码

这里简化问题，先仅考虑两个位置编码，m和n为标量，x为自变量，并把位置编码p看做是误差项，使用泰勒展开式展开至2阶：

<img src="https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164448969.png" alt="image-20230616164448969" style="zoom:50%;" />

有意思的来了，可以看到除了最后一项外，第6项是第一个同时包含 , $$p_{m}$$与$$p_{n}$$ 的交互项，希望它能表达一定的相对位置信息。

（泰勒展开是基于原函数的仿造函数，在二阶情况下能模拟一小段源函数的曲率和形状。）

![image-20230616164601705](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164601705.png)

![image-20230616164551803](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164551803.png)

先假设，上式子中的$$\mathcal H$$(黑塞矩阵)为单位矩阵$$\mathcal H=\mathcal I$$，包含相对位置信息的数据项就简化为$$\langle p_{m}，p_{n}\rangle$$ 这两个位置编码的内积。问题再次简化为，存在某个函数*g*使得，
$$
\langle p_{m},p_{n}\rangle=g(m-n)
$$
在前例中，我们将位置编码简化为64维度并可视化，这里的推导进一步简化为2维度。并假设想向量$$[x,y]$$为复数$$x+yi$$，这里根据复数的内积法则，
$$
\langle p_{m},p_{n}\rangle=\langle x_{m}+y_{m}i,x_{n}+y_{n}i\rangle=x_{m}x^*_{n}+y_{m}i*y^*_{n}i=x_{m}x_{n}+y_{m}y_{n}
$$

$$
p_{m}p^*_{n}=(x_{m}+y_{m}i)*(x_{n}-y_{n}i)=x_{m}x_{n}+y_{m}y_{n}-x_{m}y_{n}i+x_{n}y_{m}i
$$

由上式可以得出，
$$
\langle p_{m},p_{n}\rangle=Re[p_{m}p^*_{n}]
$$
为了满足（1）中提出的要求，可以假设存在$q_{m-n}$使得，
$$
p_{m}p^*_{n}=q_{m-n}
$$
这里，使用极坐标求解，先设 $p_{m}=r_me^{i\theta_m},p^*_{n}=r_me^{-i\theta_n},q_{m-n}=R_{m-n}e^{i\Theta_{m-n}}$，有下式子
$$
r_mr_n=R_{m-n}
$$

$$
\theta_m-\theta_n=\Theta_{m-n}
$$

对于(6)，令n=m，得$r^2_m=R_0$，可得到$r_m$为常数，直接令$r_m=1$；

对于(7)，代入$n=0$可得到，$\theta_m-\theta_0=\Theta_m$即$\theta_m=\Theta_m$，得到$\theta_m-\theta_n=\theta_{m-n}$，为等差数列。

因此，在二维情况下，位置编码的解为，
$$
p_m=e^{im\theta}=\begin{pmatrix}
cos(m\theta)\\   
sin(m\theta)\\
\end{pmatrix}
$$
更高维度的偶数维位置编码，可以表示为多个二维位置编码的组合
$$
p_m
=\begin{pmatrix}
e^{im\theta_0}\\
e^{im\theta_1}\\
...\\
e^{im\theta_{d/2-1}}\\
\end{pmatrix}
=\begin{pmatrix}
cos(m\theta_0)\\   
sin(m\theta_0)\\
cos(m\theta_1)\\   
sin(m\theta_1)\\
...\\
cos(m\theta_{d/2-1})\\   
sin(m\theta_{d/2-1})\\
\end{pmatrix}
$$
现在距离原文，只差最后一步，就是为什么选择$\theta_j=10000^{-2j/d}$，是因为随着$|m-n|$的增大，$\langle p_{m},p_{n}\rangle$会有趋向于0的特征，将该值代入，内积满足线性叠加性
$$
\langle p_{m},p_{n}\rangle=Re[e^{i(m-n)\theta_0}+e^{i(m-n)\theta_1}+...+e^{i(m-n)\theta_{d/2-1}}]\\
=\frac{d}{2}\cdot Re\begin{bmatrix} 
\frac{\sum_{\theta=0}^{d/2-1}e^{i(m-n)\cdot10000^{-2j/d}}}
{\frac{d}{2}}
\end{bmatrix}\\
\approx\frac{d}{2}\cdot Re\begin{bmatrix}
\int^{1}_{0}e^{i(m-n)\cdot10000^{-t}}dt
\end{bmatrix}x
$$
因此，问题就简化为积分$\int^{1}_{0}e^{i(m-n)\cdot10000^{-t}}dt$的拟合问题，直接在pyplot 画出来，

![image-20230616164516460](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164516460.png)

同时，苏神也比较了多种函数的可能性

![image-20230616164531022](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230616164531022.png)

所以，不一定需要严格按照原文的公式也可以达到类似的效果。

