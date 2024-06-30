#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./for_indexing \
  --index ./indexed_corpus \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1