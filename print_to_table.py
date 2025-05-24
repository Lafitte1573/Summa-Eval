import json
from os import path

if __name__ == '__main__':
    eval_datasets = [
        'SAMSum.jsonl',
        'xsum.jsonl',
        'cnn-dm.jsonl',
        'lcsts.jsonl'
    ]
    for file in eval_datasets:
        print("#### " + file.split('.')[0].upper())
        print('| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-W |')
        print(r'| ---  | --- | --- | --- | --- |')
        lines = [json.loads(line) for line in open(path.join('results', file)).readlines()]
        for line in lines:
            # print(line)
            print(' | ' + ' | '.join([f'{100*float(v):.2f}' if k != 'model' else v for k, v in line.items()]) + ' | ')
        print()
