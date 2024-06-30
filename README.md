# Multi-Document Question Answering with Lightweight Embeddings-Based Document Reranker

## Package Setup

For BM25 usage, you have to install Pyserini package. There is an [installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md). The installation process we used (pytorch installation is included here):

```bash
conda install -c conda-forge openjdk=21 maven -y
conda install -c conda-forge lightgbm nmslib -y
conda install -c pytorch faiss-cpu -y
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pyserini
```

Then, you can install other packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

You need the StrategyQA dataset along with its Wikipedia corpus:

```bash
wget https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
wget https://storage.googleapis.com/ai2i/strategyqa/data/corpus-enwiki-20200511-cirrussearch-parasv2.jsonl.gz

unzip strategyqa_dataset.zip
gzip -dv corpus-enwiki-20200511-cirrussearch-parasv2.jsonl.gz
```

Then, preprocess the Wikipedia corpus. For this, run [preprocess.py](https://github.com/nurmybtw/lightweight-mdqa/blob/main/preprocess.py).

Finally, run the indexing script needed for BM25 retrieval:

```bash
sh create_indexed_corpus.sh
```

## Reproducing Results

Each component of the system is contained within its own python script. Since the system is pipelined, you have to run these scripts sequentially. Following table shows the order of execution:

| #   | File                                                                                                | Output                     |
| --- | --------------------------------------------------------------------------------------------------- | -------------------------- |
| 1   | [decompose.py](https://github.com/nurmybtw/lightweight-mdqa/blob/main/decompose.py)                 | `decomposed_test_set.json` |
| 2   | [initial_retrieval.py](https://github.com/nurmybtw/lightweight-mdqa/blob/main/initial_retrieval.py) | `torerank_test_set.json`   |
| 3   | [rerank.py](https://github.com/nurmybtw/lightweight-mdqa/blob/main/rerank.py)                       | `reranked_test_set.json`   |
| 4   | [aggregate.py](https://github.com/nurmybtw/lightweight-mdqa/blob/main/aggregate.py)                 | `preds_test_set.json`      |

After this, `preds_test_set.json` should be submitted to the [StrategyQA leaderboard](https://leaderboard.allenai.org/strategyqa/submissions/public) in order to evaluate it on the official test set (it is the closed test set).
