from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm

searcher = LuceneSearcher('./train_index')

test_set = json.load(
    open("./output/decomposed_test_set.json", 'r', encoding='utf8'))
paragraphs = json.load(
    open("./dataset/full_corpus_paragraphs.json", 'r', encoding='utf8'))

to_rerank = []
k = 500
for item in tqdm(test_set):
    decompositions = []
    for d in item['decompositions']:
        hits = searcher.search(d, k=k)
        temp = []
        for hit in hits:
            doc = paragraphs[hit.docid]
            doc['id'] = hit.docid
            temp.append(doc)
        decompositions.append({
            "question": d,
            "documents": temp
        })
    to_rerank.append({
        "qid": item['qid'],
        "question": item['question'],
        "decompositions": decompositions
    })

json.dump(to_rerank, open("./output/torerank_test_set.json", 'w'))