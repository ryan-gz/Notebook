# LLaMA

## Abstract

AIGC、LLM这块几乎是今年最火的话题，因为项目开发的关系也终于有机会正式接触相关的知识内容。其实和之前的NLP中的transformer、finetune范式一脉相承，我非常希望知道为什么在chatGPT3后生成式模型可以变现出相当的智能，他是如何完成逻辑推理和在线学习的？

需要学习的知识很多，先从最近接触并复现的LLaMa（6b&13b version）开始。在Intro中，Meta提到一个观点，it seems more parameters will lead to better performance，所谓的「量变引发质变」是解释AIG的主流观点；但 MetaAI 后面驳斥了这个观点，在 Hoffmann 最近的研究中显示大基数corpora下最好的表现来自更小的模型而不是越大越好。

A simple line: a **7B model** continues to improve(performance) even after 1T tokens.

为了证明语言模型的性能和参数规模并不存在严格的正相关关系，文章给出了几个参照组对象，分别是GPT-3\Chinchilla\PaLM，这些业内SOAT的LLM，另外还有GLM\BLOOM\GLM。

## Architecture

### Pre-training Data

使用了混合的语料结构，具体如下：

| Dataset|Sampling prop.|Epochs| Disk size|
| ----- | ---- | ---- | ---- |
| CommonCrawl | 67.0% | 1.10 | 3.3 TB |
| C4 | 15.0% | 1.06 | 783 GB |
| Github | 4.5% | 0.64 | 328 GB |
| Wikipedia | 4.5% | 2.45 | 83 GB |
| Books | 4.5% | 2.23 | 85 GB |
| ArXiv | 2.5% | 1.06 | 92 GB |
| StackExchange | 2.0% | 1.03 | 78 GB |



### Tokenizer

LLaMa 的 Tokenizer比较特别，使用的是BPE（byte pair encoding）范式。使用的是 Sentence-Piece的实现方案，这块已经集成到大部分框架里。

**这里简单介绍了一下 BPE的实现方案，就是为所有的表征数值赋予一个独立的数字，然后回滚到一个个字节，从而压缩为一个未知的utf-8字符。（说实话没懂，还得看下BPE的原理）**

### Model Imporvements

- Pre-normalization[GPT3]

  为了提高训练的稳定性，我们选择对sub-layer的输入进行层规一化，并剔除了原transformer中对输出的归一化。

  **这里使用的是RMSNorm（Zhang and Sennrich）。**

- SwiGLU activation function[PaLM]

  为了提高模型的表现，选择使用SwiGLU作为激活函数替换了原始的ReLU。

- Rotary Embeddings[GPTNeo]

  这里同样使用了**RoPE**，苏神的旋转位置编码性能优越性和原理在去年有专题写过，需要再复习下。

### Optimizer

使用了 AdamW 作为优化器，并使用了 cosine learning rate schedule 作为学习率的优化策略。 

### Efficient Implementaion

通过优化模型的实现方法来

- causal multi-head attention 因果多头注意力

  节约显存开销，减少训练时长。

  实现方法是不存储attention的权重与key/query的分数

- save activations

  通过存储**激活层的checkpoint结果**来减少反向传播中激活层的重复计算，取代了PyTorch的autograd，使用

## Tasks&Benchmarks

常见LLM的验证任务如下：

- Common Sense Reasoning(常识)
- Closed-book Question Answering(闭卷问答)
- Reading Comprehension
- Mathematical Reasoning
- Code Genration
- Massive Multitask Language Understanding
- Evolution of performance during training(在线学习)

## Instruction Fintuning 

在MMLU（Massive Multitask Language Understanding）任务中，作者发现非常小规模的指令微调可以大大提高模型在MMLU任务中的表现能力。



