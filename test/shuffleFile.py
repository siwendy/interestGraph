'''
Created on 2019年1月17日

@author: zhangyanqing1
'''
import numpy as np

infile = open("corpus_v2.20180327.raw.train", 'r')
expect_num = 1000
lines = []
for i in range(38):
    lines.append([])

for line in infile:
    label = (int)(line.strip().split('\t')[0])
    lines[label - 1].append(line)
infile.close()

print('pre-resample')
for i in range(len(lines)):
    length = len(lines[i])
    print(length)
    if length >= expect_num:
        lines[i] = lines[i][0:1000]
    else:
        while(len(lines[i]) < expect_num):
            lines[i].append(lines[i][np.random.randint(0, length - 1)])

print('after-resample')
for l in lines:
    print(len(l))

new_lines = []
for label in lines:
    for l in label:
        new_lines.append(l)
np.random.shuffle(new_lines)
ofile = open("train.txt", 'w')
for line in new_lines:
    ofile.write(line)
ofile.close()
