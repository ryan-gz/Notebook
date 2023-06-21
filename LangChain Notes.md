# LangChain Notes

## Get started

### Modules

- Model I/O
  Interface with language models

  提供语言模型的接口

- Data connection
  Interface with application-specific data

  提供应用场景数据的接口

- Chains
  Construct sequences of calls

  制作一系列的call，目前不懂什么意思

- Agents
  Let chains choose which tools to use given high-level directives

  让chains选择使用工具来给定高级指令

- Memory
  Persist application state between runs of a chain

  在chain的运行程序之间保持应用程序状态

- Callbacks
  Log and stream intermediate steps of any chain

  流式log

## Model I/O

![image-20230619140637381](https://raw.githubusercontent.com/ryanzhangga1991/img_cache/main/uPic/image-20230619140637381.png)

- Format Part

  制作模型输入的prompt

  [Prompts](https://python.langchain.com/docs/modules/model_io/prompts/): Templatize, dynamically select, and manage model inputs

- Predict Part

  通过接口传递给LM生成结果

  [Language models](https://python.langchain.com/docs/modules/model_io/models/): Make calls to language models through common interfaces

- Parse Part

  生成模型输出

  [Output parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/): Extract information from model outputs

### Prompts

### Prompt templates

> Parametrize model inputs
>
> 参数化模型输入

#### few-shot prompt templates

制作few-shot prompt 模板，使用下面两个函数可以实现简易问答，但是比较笨

```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# 根据QA分句制作模板
example_prompt = PromptTemplate(input_variables=["question", "answer"], 
                                template="Question: {question}\n{answer}")
# 输入QA，并定义suffix
prompt = FewShotPromptTemplate(examples=examples,
                               example_prompt=example_prompt,
                               suffix="Question: {input}",
                               input_variables=["input"])
```

更符合智能语言模型的思路是引入语义分析，使用embedding对example做了预处理

```python
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


example_selector = SemanticSimilarityExampleSelector.from_examples(examples,                                                                   OpenAIEmbeddings(),Chroma,k=1)
prompt = FewShotPromptTemplate(example_selector=example_selector, 
                               example_prompt=example_prompt,
                               suffix="Question: {input}",
                               input_variables=["input"])
```



### Example selectors

> Dynamically select examples to include in prompts
>
> 动态选择要包含在 prompts 中的示例



## Embedding

```python
# model config
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "GanymedeNil/text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
llm_model_dict = {
    "chatglm": {
        "ChatGLM-6B": "THUDM/chatglm-6b",
        "ChatGLM-6B-int4": "THUDM/chatglm-6b-int4",
        "ChatGLM-6B-int8": "THUDM/chatglm-6b-int8",
        "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe"
    },
    "belle": {
        "BELLE-LLaMA-Local": "/pretrainmodel/belle",
    },
    "vicuna": {
        "Vicuna-Local": "/pretrainmodel/vicuna",
    }
}
```

























