import json
import os

os.makedirs('for_indexing')
paragraphs = {}
with open('./enwiki-20200511-cirrussearch-parasv2.jsonl', 'r') as full_corpus:
    with open('./for_indexing/full_corpus_for_indexing.jsonl', 'w') as output:
        for i, line in enumerate(full_corpus):
            paragraph = json.loads(line)
            title = paragraph['title']
            content = paragraph['para']
            para_id = paragraph['para_id']
            id_ = f'{title}-{para_id}'
            
            json_string = json.dumps({
                "id": id_,
                "contents": f'{title} {content}'
            })
            output.write(json_string + '\n')
            paragraphs[id_] = {
                'title': title,
                'content': content
            }

json.dump(paragraphs, open("./full_corpus_paragraphs.json", 'w'))