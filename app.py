from transformers import BertJapaneseTokenizer, BertModel
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#Load AutoModel from huggingface model repository
tokenizer = BertJapaneseTokenizer.from_pretrained("sonoisa/sentence-bert-base-ja-mean-tokens")
model = BertModel.from_pretrained("sonoisa/sentence-bert-base-ja-mean-tokens")

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
    sentences = request.json

    #Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    return jsonify(sentence_embeddings.tolist())

import os

if __name__ == "__main__":
    app.run(port=int(os.getenv('PORT',3000)))
