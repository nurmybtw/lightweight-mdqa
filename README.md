# Multi-Document Question Answering with Lightweight Embeddings-Based Document Reranker

## Package Setup

For BM25 usage, you have to install Pyserini package. There is an [installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md). The installation process we used:

```bash
conda install -c conda-forge openjdk=21 maven -y
conda install -c conda-forge lightgbm nmslib -y
conda install -c pytorch faiss-cpu pytorch -y
pip install pyserini
```

Then, you can install other packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

You need the StrategyQA dataset along with its Wikipedia corpus:

```bash
mkdir dataset
cd dataset

wget https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
wget https://storage.googleapis.com/ai2i/strategyqa/data/corpus-enwiki-20200511-cirrussearch-parasv2.jsonl.gz

unzip strategyqa_dataset.zip
gzip -dv corpus-enwiki-20200511-cirrussearch-parasv2.jsonl.gz
```

Then, preprocess the Wikipedia corpus. For this, run [preprocess.py]().

Finally, run the indexing script needed for BM25 retrieval:

```bash
sh create_indexed_corpus.sh
```

## Reproducing Results

Each component of the system is contained within its own python script. Since the system is pipelined, you have to run these scripts sequentially.
