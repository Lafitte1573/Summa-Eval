from run import load_data, data_dir
from os import path

data = load_data(path.join(data_dir, 'xsum/test.csv'))
print(len(data), data[0])
dialogues = [d['dialogue'] for d in data]
print(len(dialogues), dialogues[0])
lens = [len(dial.split(' ')) for dial in dialogues if type(dial) is str]
print(max(lens), min(lens), sum(lens) / len(lens))

filter_dialogs = [dial for dial in dialogues if type(dial) is str and len(dial.split(' ')) <= 2000]
print(len(filter_dialogs))
exit()

import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

plt.hist(lens, bins=50)
plt.xlabel('Dialogue Length (words)')
plt.title('Dialogue Length Distribution')
plt.savefig('xsum_input_length_dist.png')
