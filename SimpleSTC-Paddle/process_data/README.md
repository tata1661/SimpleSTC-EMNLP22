## Prepare the raw data

Please download the Glove6B Pretrained Embedding and split the vocab and the embeddings list.

"WikiText-103" can be obtained from Hugging Face “datasets” libaries.

Please check the path in the scripts before run them.
Global Pool Process
```
python raw_doc_pool.py
python pool_process.py
```

Please prepare the corpus in a 'txt' file, and in the format that one line with one text.
Then please prepare the split file that one line represents one text, like "0\tvalid\tpos", wchich is "index split class".
Please check the path in the scripts before run them.
Data Process
```
python dataset_process.py
```

