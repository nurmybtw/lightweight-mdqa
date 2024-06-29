import json
import os
from models.openai import generate
import numpy as np
from tqdm import tqdm

def decompose(question):
    system_prompt = '''
    Decompose a question in self-contained sub-questions that will help the system find the related evidence documents for answering the original question. Give the original question when no decomposition is needed. Answer in the following JSON format:
    {
        decompositions: ['Decomposition 1', 'Decomposition 2', ...]
    }

    Example 1: 
    Input: {question: "Is Hamlet more common on IMDB than Comedy of Errors?"}
    Output: {decompositions: ["How many listings of Hamlet are there on IMDB?", "How many listing of Comedy of Errors is there on IMDB?"]}

    Example 2: 
    Input: { question: "Are birds important to badminton?" } 
    Output: { decompositions: ["Are birds important to badminton?"] }

    Example 3: 
    Input: { question: "Is it legal for a licensed child driving Mercedes-Benz to be employed in US?" } 
    Output: { decompositions: [ "What is the minimum driving age in the US?", "What is the minimum age for someone to be employed in the US?" ] }

    Example 4: 
    Input: { question: "Are all cucumbers the same texture?" } 
    Output: { decompositions: ["Are all cucumbers the same texture?"] }

    Example 5: 
    Input: { question: "Hydrogen's atomic number squared exceeds number of Spice Girls?" } 
    Output: { decompositions: [ "What is the atomic number of hydrogen?", "How many Spice Girls are there?" ] }
'''
    res = generate(question, system_prompt, model='gpt-4o')
    return json.loads(res)['decompositions']
    

# print(decompose('Are more people today related to Genghis Khan than Julius Caesar?'))

test_set = json.load(open("/kaggle/input/strategyqa-split/strategyqa_val_split.json","r", encoding="utf8"))

for item in tqdm(test_set):
    if "decompositions" not in item:
        item['decompositions'] = decompose(item['question'])

json.dump(test_set, open("decomposed_test_set.json","w"))

