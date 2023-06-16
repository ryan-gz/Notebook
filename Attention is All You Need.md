## Attention is All You Need



#### 1. 序列编码

使用NN处理sequence，首要工作是 token 后的coding。过去十多年流行的两个主流做法是：

1.1. RNN

recurrent NN的本质为递归神经网络，无论是LSTM还是GRU都是在原始递归框架上的改造。因此，RNNs也带有递归算法天然缺陷：计算速度慢、无法并行计算。

<font color=red>Su: RNN无法很好地学习全局结构信息，因为他的本质是一个马尔可夫决策过程。</font>


Ryan: 什么是马尔可夫决策过程？

**马可夫决策过程**（英语：Markov decision process，**MDP**）是[离散时间](https://zh.wikipedia.org/w/index.php?title=離散時間&action=edit&redlink=1)[随机](https://zh.wikipedia.org/wiki/隨機)[控制](https://zh.wikipedia.org/wiki/最佳控制)过程

具体原理没懂，为何能和RNN串上？

1.2. CNN

使用卷积进行窗口式遍历，CNN相较于RNN更容易捕捉到全局的结构信息。

1.3. Attention

新的序列编码范式，在原文中，使用的是矩阵点乘完成了对全局信息的获取。

#### 2. Attention Layer

<img src="https://spaces.ac.cn/usr/uploads/2018/01/458889390.png" alt="img" style="zoom: 50%;" />
$$
Attention(Q,K,V)=softmax(\frac{QK^\top}{\sqrt{d_k}})\cdot V
$$
首先声明，输入为形状为 $(n , d_k)$的序列Q(query)，查询键为 $(m , d_k)$的序列K(key)，结果矩阵为 $(m , d_v)$的序列V(value)，因此我们可以认为Attention层就是把 $(n , d_k)$序列转化为 $(n , d_v)$的序列。

如果细节到每个维度的计算，上式的意思就是通过$q_t$ 这个query，通过与各个$k_s$ 内积的并softmax的方式，来得到$q_t$与各个 $k_s$相似度，然后加权求和，得到一个*d* 维的向量。

<font color=red>Su: 其中因子$\sqrt{d}$起到调节作用，使得内积不至于太大(太大的话softmax后就非0即1)</font>

##### 2.1. 拓展——性能瓶颈

我们可以将Attention看成一个二元联合分布(实际上是n个一元分布，不过这个细节并不重要)，如果序列长度都为n，也就是每个元有n个可能的取值，那么这个分布共有$n^2$ 个值。但是，我们将Q, K分别投影到低维后，各自的参数量只有$n × (d/h)$，总的参数量是$2nd/h$。

所以，式(1)就相当于用$2nd/h$的参数量去逼近一个本身有 $n^2$ 个值的量，而我们通常有$2nd/h≪n2$ ，尤其是h比较大时更是如此，因此这种建模有点“强模型所难”，这就是原论文中的“低秩瓶颈(Low-Rank Bottleneck)”的含义。







#### 3. Multi-Head Attention

<img src="https://spaces.ac.cn/usr/uploads/2018/01/2809060486.png" alt="img" style="zoom:50%;" />

本质就是通过权重矩阵，完成同一件事件做N遍。
$$
head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)
$$

$$
MultiHead(Q,K,V) = Concat(head_1,head_2,...,head_h)
$$

#### 4. Self Attention

在Google的原文中，大部分的注意里为Self Attention ，即“自注意力”，所谓Self Attention就是Attention(X,X,X)，即寻找序列内部的联系。

原本中所使用的就是 Self Multi-Head Attention:
$$
Y = MultiHead(X,X,X)
$$

#### 5. Position Embedding

见 Sinusoidal Position Embedding

#### 6. 不足之处

1. Su：无法对位置信息编码进行有校的建模，这里引入的三角函数编码只是缓解方案

   Ryan：后来的旋转位置编码也并非建模，是不是固定的编码已成最佳解决方案

2. Su：并非所有问题都需要长程的、全局的依赖的，也有很多问题只依赖于局部结构，这时候用纯Attention也不大好。事实上，Google似乎也意识到了这个问题， 因此论文中也提到了一个restricted版的Self-Attention(不过论文正文应该没有用到它)，它假设当前词只与前后*r*个词发生联系，因此注意力也只发生在这 2*r* + 1个词之间，这样计算量就是*𝒪*(*nr*)，这样也能捕捉到序列的局部结构了。但是很明显，这就是卷积核中的卷积窗口的概念。

3. 改进的工作也相继涌现:

   1. 有改进预训练任务的，XLNET的PLM；ALBERT的SOP等;
   2. 有改进归一化的，Post-Norm向Pre-Norm改变，T5中去掉了Layer Norm里边的beta;
   3. 有改进模型结构的，Transformer-XL等;
   4. 有改进训练方式的，ALBERT的参数共享等；



