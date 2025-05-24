## Summa-Eval: An Open Package for the Evaluation of LLM Summarization

Summa-Eval 是一个用于评价 LLM 文本摘要性能的开源工具，它可以采用三种方式来执行大模型摘要并实现结果评价评价，包括：

- Ollama 方式
- OpenAI API 方式
- Pytorch 原生方式

本项目默认使用 `ROUGE` F1 分数作为评价指标，并且已经在 `dataset` 目录中提供了四个常用文本摘要数据集 CNN/DM、XSum、LCSTS，以及 SAMSum 的测试集文件。
如有需要，使用者可以自行在该目录下添加更多测试集，此外，与文本摘要类似的序列到序列（Sequence-to-Sequence, Seq2Seq）任务也可以采用本框架进行测评

----
### 安装依赖

请确保您的环境中已经安装了 ollama、openai，以及 pytorch
```
1. 安装 ollama
curl -fsSL https://ollama.com/install.sh | sh

2. 安装 openai
pip install openai

3. 安装 pytorch
pip install torch
```

完成以上安装后，拉取本项目到本地，并进入项目根目录：

```bash
git clone https://github.com/Lafitte1573/Summa-Eval.git/
cd Summa-Eval
```

----

### 基于第三方部署的摘要生成与评价

基于第三方部署是指我们使用 `Ollama` 工具在本地部署的模型，或者使用 `OpenAI API` 直接调用服务商在云端部署的 LLM：

- **Ollama 方式**：如果采用 `Ollama` 方式，请执行以下命令：

```bash
python run_with_api.py --client openai --model <your_model_name> --dataset <dataset_name>
```

> 请确保你的 Ollama 环境中已经安装了 `<your_model_name>` 所指定的模型！

- **OpenAI 方式**：如果采用 `OpenAI API` 调用在线模型，请首先确保您在 run_with_api.py 脚本的第 16~20 行中设置了服务商地址和密钥，然后执行以下命令：

```bash
python run_with_api.py --client openai --model <your_model_name> --dataset <dataset_name>
```

> 请确保命令中的 `<your_model_name>` 与服务商命名规范一致！

----

### 基于原生 Pytorch 的摘要生成与评价

基于 Ollama 的本地部署方式通常适合参数量较大的开源模型（如 Qwen 系列 32/72B 模型），而基于 OpenAI API 的在线调用更加适合商用非开源模型和超大参数量的 MoE 模型（如 GPT-4o、Deepseek 等）。
然而 Ollama 无法做到批量推理，API 调用方式通常需要按 tokens 用量付费。
因此，对于参数量在 0.5B~9B 的模型，推荐在单卡 GPU 上直接采用 Pytorch 运行模型，以节省开支并达到加速效果。命令如下：

```bash
python run.py --model <your_model_name> --batch_size <batch_size> --dataset <dataset_name>
```

> 请注意：
> 1. 您需要在 `run.py` 脚本的第 24 行指明模型的存放路径，并且确保 `<your_model_name>` 与本地模型存放路径的命名规范一致；
> 2. 您可以通过设置恰当的 `<batch_size>` 来控制模型推理的并发数量，从而提高模型推理速度；