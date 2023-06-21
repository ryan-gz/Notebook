## Rotary Position Embedding 

苏剑林 https://spaces.ac.cn/

#### 1. 证明思路

在RoPE中，我们的出发点就是“通过绝对位置编码的方式实现相对位置编码”，这样做既有理论上的优雅之处，也有实践上的实用之处，假设通过下述运算来给 qkv添加绝对位置信息，
$$
\widetilde{q}_m=f(q,m),\widetilde{k}_n=f(k,n)
$$
根据SPE章节的经验，相对位置信息本质来源于内积，故有假设，
$$
\langle f(q,m),f(k,n)\rangle=g(q,k,m-n)
$$
这里还需要几个前提条件，可以假设当绝对位置为0时候，因为没有位置信息，所以，
$$
f(q,0)=q,f(k,0)=k
$$
由上篇经验有，
$$
\langle p_{m},p_{n}\rangle=Re[p_{m}p^*_{n}]
$$
所以有，
$$
\langle f(q,m),f(k,n)\rangle=g(q,k,m-n)=Re[f(q,m),f^*(k,n)]
$$
为了简化上式，假设存在
$$
f(q,m)f^*(k,n)=g(q,k,m-n)
$$
使用复数来展开函数f，有
$$
f(q,m)=R_f(q,m)e^{i\Theta_f(q,m)}\\
f^*(k,n)=R_f(k,n)e^{-i\Theta_f(k,n)}\\
g(q,k,m-n)=R_g(q,k,m-n)e^{i\Theta_g(q,k,m-n)}\\
$$
那么代入方程式后得到方程式
$$
R_f(q,m)R_f(k,n)=R_g(q,k,m-n)\\
\Theta_f(q,m)-\Theta_f(k,n)=\Theta_g(q,k,m-n)
$$
对于(8)-1，令 $m=n$ 有，
$$
R_f(q,m)R_f(k,m)=R_g(q,k,0)=R_f(q,0)R_f(k,0)=||q||\cdot||k||
$$
因此，我们可以简单假设$R_f(q,m)=||q||$，即不依赖于位置$m$，

这里用到了式(3)的假设，所以位置0处的长度为模长，对于(8)-2，同样令 $m=n$ 有，
$$
\Theta_f(q,m)-\Theta_f(k,m)=\Theta_g(q,k,0)=\Theta_f(q,0)-\Theta_f(k,0)=\Theta(q)-\Theta(k)
$$
这里的$\Theta(q)$为本身的q的角度，这里最后的等式要说明一下，
$$
f(q,0)=q,f(k,0)=k\\
\frac{q}{k}
=\frac{f(q,0)}{f(k,0)}
=\frac{R_f(q,0)e^{i\Theta_f(q,0)}}{R_f(k,0)e^{i\Theta_f(k,0)}}
=\frac{||q||}{||k||}\cdot\frac{e^{i\Theta_f(q,0)}}{e^{i\Theta_f(k,0)}}
\\
$$
上式左右两边交换律，得到，
$$
\frac{q/||q||}{k/||k||}

=\frac{1\cdot e^{i\Theta(q)}}{1\cdot e^{i\Theta(k)}}
=\frac{e^{i\Theta_f(q,0)}}{e^{i\Theta_f(k,0)}}\\
\Theta(q)-\Theta(k)=\Theta_f(q,0)-\Theta_f(k,0)
$$
拿到式(10)左右两侧，得到$\Theta_f(q,m)-\Theta_f(k,m)=\Theta(q)-\Theta(k)$，即可令，
$$
\Theta_f(q,m)-\Theta(q)=\Theta_f(k,m)-\Theta(k)=\varphi(m)
$$
由上式得到，$\Theta_f(q,m)=\Theta(q)+\varphi(m)$，令$m=n-1$，则有，
$$
\varphi(m)-\varphi(m-1)\\
=\Theta_f(q,m)-\Theta(q)-\Theta_f(k,m-1)+\Theta(k)\\
=\Theta_g(q,k,1)-\Theta(q)+\Theta(k)
$$
令 $\theta=\Theta_g(q,k,1)-\Theta(q)+\Theta(k)$，则$\varphi(m)=m\theta$为等差数列，

#### 2. 编码方法

现在我们得到了二维的RoPE，
$$
f(q,m)=R_f(q,m)e^{i\Theta_f(q,m)}\\
=||q||e^{i\Theta(q)+m\theta}
=||q||e^{i\Theta(q)}\cdot e^{i m\theta}=qe^{i m\theta}
$$
根据复数乘法的集合意义，该变换的本质为向量的旋转，因此成为“旋转位置编码”，矩阵形式为，
$$
e^{i m\theta}=cos\ m\theta+i\cdot sin\ m\theta\\
f(q,m)=
\begin{bmatrix}
cos\ m\theta\ \ -sin\ m\theta\\
sin\ m\theta\ \ \ cos\ m\theta
\end{bmatrix}\cdot
\begin{bmatrix}
q_0\\
q_1
\end{bmatrix}\\
$$
比如，$m\theta=60°，cos\ m\theta =0.5 ,sin\ m\theta=\sqrt{3}/2\approx 0.87$，上式的转换如图，
$$
f(q,m)=1.37\cdot q_0-0.37\cdot q_1
$$
<img src="https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230417162458903.png" alt="image-20230417162458903" style="zoom:33%;" />

将二维拓宽到任意偶数纬度的RoPE，均为二维的拼接，

<img src="https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230417162435206.png" alt="image-20230417162435206" style="zoom:33%;" />

由上可知，Q，K的Attention可以简化为,
$$
Attention(Q,K)=(\mathcal R_mq)^\mathsf{T}(\mathcal R_nk)=q^\mathsf{T}\mathcal R_m^\mathsf{T}\mathcal R_nk=q^\mathsf{T}\mathcal R_{m-n}k
$$
最后，我们指出，RoPE是目前唯一一种可以用于线性Attention的相对位置编码。这是因为其他的相对位置编码，都是直接基于Attention矩阵进行操作的，但是线性Attention并没有事先算出Attention矩阵，因此也就不存在操作Attention矩阵的做法，所以其他的方案无法应用到线性Attention 中。而对于RoPE来说，它是用绝对位置编码的方式来实现相对位置编码，不需要操作Attention矩阵，因此有了应用到线性Attention的可能性。





















