import argparse

import ollama
import json
from openai import OpenAI
import os
from os import path
import numpy as np

from run import scorer
from tqdm import tqdm
import random as rm

from utils import get_prompt, get_keys, load_data, save_data

# OpenAI Client
client = OpenAI(
    base_url='<base_url_here>',
    api_key='<your_api_key_here>',
)


def eval_with_openai(client, file_path, ref_key='summary', dial_key='dialogue', model_id='deepseek-r1', reason=False, limits=2000):
    lines = load_data(file_path)
    if limits > 0:
        rm.shuffle(lines)
        lines = lines[:limits]

    results = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [],
    }

    prompt = get_prompt(file_path)

    for i in tqdm(range(len(lines))):  # len(lines)
        reference = lines[i][ref_key]
        try:  # 调用 API 的话有可能次数超过限制
            response = client.chat.completions.create(
                model=model_id,
                messages=[{
                    'role': 'user',
                    'content': lines[i][dial_key] + '\n\n' + prompt
                }],
                # stream=True
            )
            # prediction = ''
            # for chunk in response:
            #     prediction += chunk.choices[0].delta.content
            prediction = response.choices[0].message.content
            if reason:
                prediction = prediction.split('</think>')[-1].strip()
            if i == 0:
                print(prediction)

            # 计算分数
            scores = scorer.score(reference, prediction.strip())

            # 记录结果
            for key in scores:
                results[key].append(scores[key].fmeasure)
        except Exception as e:
            print(e)
            continue

    # 保存结果
    dataset_name = file_path.split('/')[0]
    final_results = {'model': model_id}
    for key in results.keys():
        # print(key, results[key])
        final_results[key] = np.mean(results[key])
    save_data(final_results, dataset_name)


def eval_with_ollama(file_path, ref_key='summary', dial_key='dialogue', model_id='qwen2-7b-instruct', reason=False, limits=2000):
    lines = load_data(file_path)

    lines = [line for line in lines if type(line[dial_key]) is str]  # 过滤无效数据
    rm.shuffle(lines)

    results = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [],
    }

    prompt = get_prompt(file_path)

    for i in tqdm(range(min(limits, len(lines)))):
        reference = lines[i][ref_key]
        response = ollama.chat(
            model=model_id,
            messages=[{
                'role': 'user',
                'content': lines[i][dial_key] + '\n\n' + prompt
            }]
        )
        prediction = response['message']['content']
        if reason:
            prediction = prediction.split('</think>')[-1].strip()

        if i == 0:
            print(prediction)

        scores = scorer.score(reference, prediction)

        # 记录结果
        for key in scores:
            results[key].append(scores[key].fmeasure)
        # break

    # 保存结果
    dataset_name = file_path.split('/')[0]
    final_results = {'model': model_id}
    for key in results.keys():
        # print(key, results[key])
        final_results[key] = np.mean(results[key])
    save_data(final_results, dataset_name)


if __name__ == '__main__':
    remote_models = [
        'deepseek-v3', 'qwen3-235b-a22b', 'deepseek-r1'
    ]

    eval_datasets = {
        'samsum': 'SAMSum/test.json',
        'xsum': 'xsum/test.csv',
        'cnn-dm': 'cnn-dm/test-00000-of-00001.parquet',
        'lcsts': 'lcsts/dev.jsonl',
    }

    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='qwen2')
    args.add_argument('--client', type=str, default='ollama', choices=['ollama', 'openai'])
    args.add_argument('--reason', action='store_true', help='Use reason model.')
    args.add_argument('--dataset', type=str, default='samsum', choices=eval_datasets.keys(), help='Dataset to evaluate.')
    args = args.parse_args()
    print(args)

    eval_file = eval_datasets[args.dataset]
    ref_key, dial_key = get_keys(eval_file)

    # 运行评价
    if args.client == 'ollama':
        eval_with_ollama(
            eval_file,
            model_id=args.model,
            ref_key=ref_key,
            dial_key=dial_key,
            reason=True if args.reason else False
        )
    else:
        eval_with_openai(
            client,
            eval_file,
            model_id=args.model,
            ref_key=ref_key,
            dial_key=dial_key,
            reason=True if args.reason else False
        )


