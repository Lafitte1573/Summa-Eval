## Summa-Eval: An Open Package for the Evaluation of LLM Summarization

Summa-Eval 是一个用于评价 LLM 文本摘要性能的轻量化开源工具，它主要由 `run_with_api.py` 和 `run.py` 两个脚本构成，可以采用三种方式运行大模型并评价其摘要结果，包括：

- Ollama 方式：在 `run_with_api.py` 脚本中实现
- OpenAI API 方式：在 `run_with_api.py` 脚本中实现
- Pytorch 原生方式：在 `run.py` 脚本中实现

本项目默认使用 `ROUGE` F1 分数作为评价指标，并且已经在 `dataset` 目录中提供了四个常用文本摘要数据集 CNN/DM、XSum、LCSTS，以及 SAMSum 的测试集文件。
如有需要，使用者可以自行在该目录下添加更多测试集，此外，与文本摘要类似的序列到序列（Sequence-to-Sequence, Seq2Seq）任务也可以采用本框架进行测评

----
### 安装依赖

请确保您的环境中已经安装了 ollama、openai，以及 pytorch
```
1.安装 ollama
curl -fsSL https://ollama.com/install.sh | sh

2.安装 openai
pip install openai

3.安装 pytorch
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
> 1.您需要在 `run.py` 脚本的第 24 行指明模型的存放路径，并且确保 `<your_model_name>` 与本地模型存放路径的命名规范一致；
> 2.您可以通过设置恰当的 `<batch_size>` 来控制模型推理的并发数量，从而提高模型推理速度；

----

### 评价结果

命令行运行 `print_to_table.py` 脚本可打印评价结果为表格

#### SAMSUM
| Model                      | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-W |
|----------------------------|---------|---------|---------|---------|
 | Qwen2.5-0.5B               | 16.48   | 5.25    | 12.95   | 14.53   | 
 | Meta-Llama-3.1-8B-Instruct | 39.61   | 15.08   | 30.79   | 30.79   | 
 | Qwen2.5-7B-Instruct        | 29.49   | 10.66   | 22.00   | 22.57   | 
 | glm-4-9b-chat-hf           | 34.26   | 13.33   | 26.48   | 26.58   | 
 | deepseek-v3                | 28.15   | 10.32   | 21.26   | 22.22   | 
 | qwen3-235b-a22b            | 29.04   | 9.87    | 21.41   | 21.91   | 
 | deepseek-r1                | 32.26   | 10.26   | 23.86   | 23.89   | 
 | Qwen3-8B                   | 31.36   | 10.57   | 23.43   | 23.46   | 
 | llama3:70b                 | 39.25   | 14.50   | 30.25   | 30.25   | 
 | qwen2.5:72b                | 33.80   | 12.41   | 25.31   | 25.34   | 

#### XSUM
| Model                      | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-W |
|----------------------------|---------|---------|---------|---------|
 | Qwen2.5-0.5B               | 10.34   | 1.47    | 8.34    | 8.35    | 
 | Meta-Llama-3.1-8B-Instruct | 29.54   | 8.93    | 21.47   | 21.52   | 
 | Qwen2.5-7B-Instruct        | 19.61   | 4.83    | 13.44   | 13.62   | 
 | glm-4-9b-chat-hf           | 22.92   | 5.78    | 16.34   | 16.40   | 
 | Qwen3-8B                   | 18.34   | 4.02    | 12.31   | 12.33   | 
 | llama3:70b                 | 16.42   | 3.66    | 14.51   | 14.51   | 
 | qwen2.5:72b                | 22.59   | 5.38    | 18.09   | 18.09   | 
 | deepseek-ai/DeepSeek-R1    | 19.89   | 4.21    | 15.66   | 15.73   | 
 | qwen3-235b-a22b            | 22.90   | 4.93    | 16.08   | 16.17   | 

#### CNN-DM
| Model                      | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-W |
|----------------------------|---------|---------|---------|---------|
 | Qwen2.5-0.5B               | 19.84   | 8.34    | 13.61   | 16.96   | 
 | Meta-Llama-3.1-8B-Instruct | 32.23   | 11.37   | 20.90   | 26.65   | 
 | Qwen2.5-7B-Instruct        | 25.94   | 8.69    | 16.27   | 21.82   | 
 | glm-4-9b-chat-hf           | 33.48   | 12.08   | 21.62   | 28.36   | 
 | Qwen3-8B                   | 24.99   | 8.35    | 15.44   | 19.81   | 
 | deepseek-ai/DeepSeek-V3    | 33.95   | 12.52   | 20.81   | 26.81   | 
 | qwen2.5:72b                | 33.21   | 11.63   | 20.03   | 26.25   | 
 | llama3:70b                 | 40.68   | 15.13   | 25.49   | 32.85   | 

#### LCSTS
| Model                      | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-W |
|----------------------------|---------|---------|---------|---------|
 | Qwen2.5-7B-Instruct        | 12.05   | 1.59    | 11.97   | 11.97   | 
 | glm-4-9b-chat-hf           | 7.85    | 0.81    | 7.79    | 7.79    | 
 | Meta-Llama-3.1-8B-Instruct | 11.58   | 1.42    | 11.52   | 11.52   | 
 | Qwen3-8B                   | 12.70   | 1.67    | 12.61   | 12.61   | 
 | llama3:70b                 | 11.87   | 1.27    | 11.75   | 11.75   | 
 | qwen2.5:72b                | 13.21   | 1.54    | 13.08   | 13.08   | 
 | qwen3-235b-a22b            | 12.81   | 1.74    | 12.68   | 12.68   | 

#### 结果分析

根据以上评价结果，Llama3-70B 模型的摘要性能综合最优。
此外，如果仅以 `ROUGE` 分数作为评价指标，高性能大模型，如 Deepseek 系列、Qwen3 系列等其性能甚至不如  Qwen2.5-7B-Instruct 模型。

我们对此结果的分析如下：

1. 当前最先进的 Deepseek 系列、Qwen3 系列模型为通用模型，并且适应了对话风格的输入输出格式，容易输出口语化的描述，但句式不固定；
2. 具有推理能力的大模型（如 DS-R1、Qwen3 系列），其输出格式通常难以用指令严格控制，加剧了问题“1”；
3. 提示词的编写严重影响大模型的生成效果，导致模型性能不稳（本项目并未对提示词进行专门优化，见 `utils.py` 脚本 `get_prompt` 函数）
4. 大模型的训练方式使其倾向于输出长文本内容，因此在类似 XSum 的高抽象性数据集上比较吃亏

----

### 未来工作

本项目用较为简单的方式实现了大模型文本摘要的评价过程，未来考虑做三方面的优化：

1. 添加更多评价指标（如 `BERTScore`，`BARTScore`，`GPT` 评价等），而不仅仅是 `ROUGE` 分数；
2. 添加更多评价数据集，并且不局限于文本摘要任务；
3. 在 `run.py` 中添加多卡分布式推理，以实现以原生 Pytorch 方式运行大参数量 LLM（如 Qwen2.5-72B、Llama3-70B 等）。

----

### 引用

欢迎您勘误本项目的代码以及进行二次开发。如果您再工作中参考了项目提供的评价结果，请按照以下格式进行引用：

```bibtex
@misc{summa_eval,
  author = {Jiaxin Duan},
  title = {Summa-Eval: An Open Package for the Evaluation of LLM Summarization},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Lafitte1573/Summa-Eval}},
}
```
