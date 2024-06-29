#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./dataset/full_corpus_for_indexing.jsonl \
  --index ./dataset/indexed_corpus \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1