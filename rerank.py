import json
from models.mxbai import MXBAIReranker
import tqdm

reranker = MXBAIReranker()

data = json.load(open("./torerank_test_set.json",'r', encoding='utf8'))
batch_size = 16

# Assign similarity scores
for item in tqdm(data):
    for d in item['decompositions']:
        scores = []
        for i in range(len(d["documents"])//batch_size + 1):
            batch = d['documents'][i*batch_size: (i+1)*batch_size]
            scores += reranker.get_rerank_score(d['question'], [p["content"] for p in batch])
        for p, score in zip(d["documents"], scores):
            p['score'] = score

# Rerank according to similarity scores
for item in tqdm(data):
    ranked = []
    for d in item['decompositions']:
        ranked = ranked + d['documents']
    ranked.sort(key=lambda x: x['score'], reverse=True)
    for p in ranked:
        del p['score']
    item['ranked'] = ranked

json.dump(data, open("./reranked_test_set.json","w"))