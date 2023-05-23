import uvicorn
import pickle
import sklearn
import pandas as pd
from fastapi import FastAPI, Response
import pandas as pd
import dill
import inspect
import numpy as np
import torch
from PIL import Image
import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
from tensorflow import keras

tokenizer_obj = Tokenizer()

class CPU_Unpickler(dill.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

with open('model.pickle','rb') as f:
    model = CPU_Unpickler(f).load()

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Amazon Review Sentiment Analysis'}

@app.get('/predict/')
def predict(review:str, response: Response):
    samples = tokenizer_obj.texts_to_sequences([review])
    pad = pad_sequences(samples, maxlen=381)
    response.headers['Access-Control-Allow-Origin']= "*"
    return {'output': model.predict(x=pad).tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
