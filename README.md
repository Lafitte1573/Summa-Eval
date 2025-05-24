# Summa-Eval: An Open Package for the Evaluation of LLM Summarization

----

Summa-Eval 是一个用于评价 LLM 文本摘要性能的开源工具，它可以采用三种方式来执行大模型摘要并实现结果评价评价，包括：

- Ollama 方式
- OpenAI API 方式
- Pytorch 原生方式

本项目默认使用 `ROUGE` F1 分数作为评价指标，并且已经在 `dataset` 目录中提供了四个常用文本摘要数据集 CNN/DM、XSum、LCSTS，以及 SAMSum 的测试集文件。
如有需要，使用者可以自行在该目录下添加更多测试集，此外，与文本摘要类似的序列到序列（Sequence-to-Sequence, Seq2Seq）任务也可以采用本框架进行测评

----

## 安装依赖
请确保您的环境中已经安装了 ollama、openai，以及 pytorch
```
1. 安装 ollama
curl -fsSL https://ollama.com/install.sh | sh

2. 安装 openai
pip install openai

3. 安装 pytorch
pip install torch
```

## 使用第三方方式实现摘要生成与评价

## 调用Qwen2-7B做摘要生成
首先在命令行输入如下命令运行Qwen2-7B模型
```
ollama run qwen2
```
然后运行代码
```
python ollama-summa.py
```

## 调用Qwen2-72B评价生成摘要的质量
首先在命令行输入如下命令运行Qwen2-72B模型
```
ollama run qwen2:72b
```
然后运行代码
```
python ollama-evaluation.py
```

# vllm方式实现摘要生成与评价
## 调用Qwen2-7B-Instruct做摘要生成
首先允许vllm从ModelScope下载模型：
```
export VLLM_USE_MODELSCOPE=True
```
然后输入如下命令运行Qwen2-7B-Instruct模型
```
nohup python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model Qwen/Qwen2-7B-Instruct &
```
如果是第一次运行该模型，推荐使用下述方式，以方便查看模型下载是否成功
```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model Qwen/Qwen2-7B-Instruct
```
最后，运行代码
```
python vllm-summa.py
```

## 调用Qwen2-72B评价生成摘要的质量
首先运行Qwen2-72B模型
```
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-72B --model Qwen/Qwen2-72B
```
然后执行代码
```
python vllm-evaluation.py
```