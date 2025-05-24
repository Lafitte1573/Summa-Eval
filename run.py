import json
from openai import OpenAI
import os
from os import path
import numpy as np

from rouge_score import rouge_scorer
from tqdm import tqdm
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

from utils import get_prompt, get_keys, load_data, save_data

# 初始化评估器（支持多种Rouge类型）
scorer = rouge_scorer.RougeScorer(
    rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],  # 可选的评估类型
    use_stemmer=True  # 是否启用词干提取（对英文更有效）
)

model_dir = '<your_model_directory>'  # 替换为自己的模型存放路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_with_batch_inference(file_path, src_key='summary', tag_key='dialogue', model_id='deepseek-r1', reason=False,
                              batch_size=4, max_new_tokens=256):
    lines = load_data(file_path)

    model = AutoModelForCausalLM.from_pretrained(
        path.join(model_dir, model_id),
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        path.join(model_dir, model_id),
        trust_remote_code=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token

    results = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [],
    }
    
    # 根据数据集的不同来获取提示
    prompt = get_prompt(file_path)

    model.eval()
    for i in tqdm(range(0, len(lines), batch_size)):
        span = range(i, min(i + batch_size, len(lines)))
        # references = [lines[j][src_key] for j in span]
        # 过滤掉太长的文本（不做截断，直接让过长的文本不参与计算评价结果）
        valid_span = [j for j in span if type(lines[j][tag_key]) is str]  # and len(lines[j][tag_key].split(' ')) <= 2000
        references = [lines[j][src_key] for j in valid_span]
        input_texts = [
            tokenizer.apply_chat_template([{
                'role': 'user',
                'content': lines[j][tag_key] + '\n\n' + prompt
            }],
                add_generation_prompt=True,
                tokenize=False
            ) for j in valid_span
        ]
        # print(input_texts)
        tokenized_inputs = tokenizer(
            input_texts,
            padding='longest',  # 动态填充到批次最长文本
            truncation='longest_first',  # 当输入文本超过模型最大长度时，优先截断长句，而不是短句
            max_length=1024,  # 设置模型最大接受长度
            return_tensors='pt'
        )
        # print(encoded_inputs['input_ids'].shape)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 1,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
        }

        # 生成步骤
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):  # 混合精度优化
            outputs = model.generate(
                input_ids=tokenized_inputs['input_ids'].to(device),
                attention_mask=tokenized_inputs['attention_mask'].to(device),
                **generation_args
            )

        # 立即转移结果到 CPU 并释放 GPU 显存
        outputs = outputs.cpu()

        # 获取输出的时候需要跳过输入部分
        predictions = tokenizer.batch_decode(
            outputs[:, tokenized_inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )

        predictions = [pred.split(tokenizer.eos_token)[0] for pred in predictions]  # 截断无效部分
        if reason:
            predictions = [pred.split('</think>')[-1].strip() for pred in predictions]

        if i == 0:
            print(outputs.shape)
            print(repr(predictions[0]))

        # 计算分数
        batch_scores = [scorer.score(refer, pred) for refer, pred in zip(references, predictions)]

        # 记录结果
        for scores in batch_scores:
            for key in scores:
                results[key].append(scores[key].fmeasure)

        del tokenized_inputs
        torch.cuda.empty_cache()

    # 保存结果
    dataset_name = file_path.split('/')[0]
    final_results = {'model': model_id}
    for key in results.keys():
        final_results[key] = np.mean(results[key])
    save_data(final_results, dataset_name)


if __name__ == '__main__':
    local_models = [
        'Qwen2.5-0.5B',
        'Qwen3-8B', 
        'Meta-Llama-3.1-8B-Instruct', 
        'Qwen2.5-7B-Instruct', 
        'glm-4-9b-chat-hf',
    ]
    
    reasoning_models = [
        'Qwen3-8B',
    ]

    eval_datasets = [
        'SAMSum/test.json',
        'xsum/test.csv',
        'cnn-dm/test-00000-of-00001.parquet',
        'lcsts/dev.jsonl'
    ]
    
    # 超参数
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=local_models, required=True)
    args.add_argument('--batch_size', type=int, default=32)
    args.add_argument('--max_new_tokens', type=int, default=256)
    args = args.parse_args()
    print(args)

    for eval_file in eval_datasets:
        # 获取源序列、目标序列所对应的键名
        src_key, tag_key = get_keys(eval_file)
        save_file = f'results/{eval_file.split("/")[0]}.jsonl'
        try:
            saved_models = [s['model'] for s in [json.loads(line) for line in open(save_file, 'r')]]
            if args.model in saved_models:
                print(f'In {save_file}, {args.model} exists, skip')
                continue
        except FileNotFoundError:
            print(f'{save_file} not found, start to eval')

        # 开始评测
        eval_with_batch_inference(
            eval_file,
            src_key=src_key,
            tag_key=tag_key,
            model_id=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            reason=True if args.model in reasoning_models else False,
        )
