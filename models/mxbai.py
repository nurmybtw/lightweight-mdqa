from typing import Dict
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers.util import cos_sim

class MXBAIReranker:
    def __init__(self):
        model_id = 'mixedbread-ai/mxbai-embed-large-v1'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def pooling(self, outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def get_rerank_score(self, question, documents):
        task = 'Represent this sentence for searching relevant passages: '
        queries = [
            self.get_detailed_instruct(task, question),
        ]
        
        input_texts = queries + documents
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', max_length=512, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        outputs = self.model(**inputs).last_hidden_state
        embeddings = self.pooling(outputs, inputs, 'cls')

        similarities = cos_sim(embeddings[0], embeddings[1:])
        return similarities.numpy()[0,:]