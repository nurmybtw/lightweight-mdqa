import json
from models.openai import generate
from tqdm import tqdm

def generate_answer(question, documents):
    system_prompt = '''
        For each example, use the documents to create an "Answer" and an "Explanation" to the "Question". Just answer yes or no. Give you answer in the following JSON format:
        {
            "explanation": <explanation>,
            "answer": <yes or no>
        }

        Example:
        Input:
        {
        "question": "Did Pink Floyd have a song about the French Riviera?",
        "documents": [
            {
                "id": 1,
                "title": "San Tropez (song)",
                "content": "'San Tropez' is the fourth track from the album Meddle by the band Pink Floyd. This song was one of several to be considered for the band's 'best of' album, Echoes: The Best of Pink Floyd."
            },
            {
                "id": 2,
                "title": "French Riviera",
                "content": "The French Riviera (known in French as the Côte d'Azur [kot daˈzyʁ]; Occitan: Còsta d'Azur [ˈkɔstɔ daˈzyɾ]; literal translation 'Azure Coast') is the Mediterranean coastline of the southeast corner of France. There is no official boundary, but it is usually considered to extend from Cassis, Toulon or Saint-Tropez on the west to Menton at the France–Italy border in the east, where the Italian Riviera joins. The coast is entirely within the Provence-Alpes-Côte d'Azur (Région Sud) region of France. The Principality of Monaco is a semi-enclave within the region, surrounded on three sides by France and fronting the Mediterranean."
            },
            {
                "id": 3,
                "title": "Moon Jae-in",
                "content": "Moon also promised transparency in his presidency, moving the presidential residence from the palatial and isolated Blue House to an existing government complex in downtown Seoul."
            },
            {
                "id": 4,
                "title": "Saint-Tropez",
                "content": "Saint-Tropez (US: /ˌsæn troʊˈpeɪ/ SAN-troh-PAY, French: [sɛ̃ tʁɔpe]; Occitan: Sant-Tropetz , pronounced [san(t) tʀuˈpes]) is a town on the French Riviera, 68 kilometres (42 miles) west of Nice and 100 kilometres (62 miles) east of Marseille in the Var department of the Provence-Alpes-Côte d'Azur region of Occitania, Southern France."
            }
        ]
        }

        Output:
        {
            "explanation": "According to [Document 1], 'San Tropez' is a song by Pink Floyd about the French Riviera. This is further supported by [Document 4], which states that Saint-Tropez is a town on the French Riviera.",
            "answer": "yes"
        }
    '''

    res = generate(json.dumps({
        "question": question,
        "documents": [
            {
                "id": i + 1,
                "title": doc["title"],
                "content": doc["content"]
            } for i, doc in enumerate(documents[:10])
        ]
    }), system_prompt, model='gpt-4o')
    res = json.loads(res)

    return {
        "answer": res['answer'],
        "explanation": res['explanation'],
        "documents": documents[:10]
    }

print('Loading test set...')
test_set = json.load(open("./reranked_test_set.json",'r', encoding='utf8'))
print('Loaded test set')

preds = {}
for item in tqdm(test_set):
    doc = generate_answer(item['question'], item['ranked'])
    preds[item['qid']] = {
        "answer": True if doc['answer'] == 'yes' else False,
        "decomposition": [d['question'] for d in item['decompositions']],
        "paragraphs": [p['id'] for p in doc['documents']]
    }

json.dump(preds, open("./preds_test_set.json", "w"))