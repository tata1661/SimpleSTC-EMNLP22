from datasets import load_dataset
import time
import pickle as pkl
start = time.time()
datasets = load_dataset('wikitext', 'wikitext-103-raw-v1')
print(len(datasets['train']))

with open('train_raw.txt', 'w') as f:
    count = 0
    for i in datasets['train']['text']:
        if i == '':
            count += 1
        f.write(i)
print(count)
f.close()

title = []
with open('train_raw.txt', 'r') as fin:
    count = 0 
    for item in fin.readlines():
        if item[:2] != ' =':
            pass
        elif item[:5] == ' = = ':
            pass
        else:
            count += 1
            title.append(item)
fin.close()
print(len(title))
with open('title_raw.txt', 'w') as fout:
    for item in title:
        fout.write(item)
fout.close()

text = []
with open('train_raw.txt', 'r') as fin:
    flag = 0
    for item in fin.readlines():
        if flag == 1:
            if item[:2] != ' =':
                text.append(item)
            else:
                flag = 0 
        if flag == 0:
            if item[:2] != ' =':
                pass
            elif item[:5] == ' = = ':
                pass
            else:
                flag = 1
                text.append(item)
fin.close()
with open('processed.txt', 'w') as fout:
    for txt in text:
        if txt == None:
            fout.write('\n')
        else:
            fout.write(txt)
fout.close()


text = {}
with open('train_raw.txt', 'r') as fin:
    flag = 0
    for item in fin.readlines():
        if flag == 1:
            if item[:2] != ' =':
                left = item.find('(')
                right = item.find(')')
                item = item[:left] + item[right + 1:]
                text[title].append(item)
            else:
                flag = 0 
        if flag == 0:
            if item[:2] != ' =':
                pass
            elif item[:5] == ' = = ':
                pass
            elif item[-4:] == ' = \n':
                title = item.strip('\n').strip(' = ')
                tmp = title.split(' ')
                if title.istitle():
                    text[title] = []
                    flag = 1
                elif len(tmp) >= 2 and len(tmp) <= 10 :
                    if tmp[0].isdigit() and tmp[1].istitle():
                        text[title] = []
                        flag = 1
                elif len(tmp) >= 4:
                    if tmp[0].isdigit() and tmp[1]== '-' and tmp[2].isdigit() and tmp[3].istitle() or tmp[0].isdigit() and tmp[1] == '(' and tmp[-1] == ')' and 'and' not in tmp and '>' not in tmp:
                        text[title] = []
                        flag = 1
print(len(text))

pop_list = []
for key, val in text.items():
    for item in val:
        if len(item) < 20:
            pop_list.append(key)
            break

for item in pop_list:
    text.pop(item)
print(len(text))
doc_list = []
for key, val in text.items():
    for item in val:
        if len(item) < 100:
            continue
        doc_list.append(item)
print(len(doc_list))
with open('doc_list.txt', 'w') as fout:
    fout.writelines(doc_list)
fout.close()

