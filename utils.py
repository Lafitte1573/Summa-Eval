import json
import os
from os import path
import pandas as pd


def get_prompt(file_path):
    if file_path.startswith('SAMSum'):
        prompt = 'The summary of this dialogue is:'
    elif file_path.startswith('cnn'):
        prompt = 'The summary of this text is:'
    elif file_path.startswith('xsum'):
        prompt = 'Summarize the main idea of this text in a few words:'
    elif file_path.startswith('lcsts'):
        prompt = '请用一句话总结文本内容:'
    else:
        raise ValueError("Unsupported dataset. Please provide a dataset from SAMSum, CNN, or XSum.")
    return prompt


def load_data(file_path, data_dir='dataset'):
    if file_path.endswith('.json'):
        return json.load(open(path.join(data_dir, file_path), 'r'))
    elif file_path.endswith('.csv'):
        return pd.read_csv(path.join(data_dir, file_path)).to_dict(orient='records')
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(path.join(data_dir, file_path)).to_dict(orient='records')
    elif file_path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path.join(data_dir, file_path), 'r')]
    else:
        raise ValueError("Unsupported file format. Please provide a .json, .csv, or .parquet file.")


def get_keys(eval_file):
    if eval_file.startswith('cnn-dm'):
        src_key, tar_key = 'highlights', 'article'
    elif eval_file.startswith('lcsts'):
        src_key, tar_key = 'summary', 'content'
    else:
        src_key, tar_key = 'summary', 'dialogue'
    return src_key, tar_key


def save_data(json_line, save_file, save_dir='results'):
    if not path.exists(save_dir):
        os.mkdir(save_dir)
    with open(f'results/{save_file}.jsonl', 'a') as f:
        json.dump(json_line, f, ensure_ascii=False)
        f.write('\n')
