from fastapi import FastAPI

import uvicorn
import pydantic
import nltk
import pydantic
from fastapi import FastAPI
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from enums.disaster_degrees import disasterDegree

nltk.download('punkt')
import re
import string
import tensorflow as tf
from keras import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Embedding
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import LSTM
from keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from Model.my_model import Tweet

from services.predictionservice import PredictionService

import tensorflow as tf
from keras.models import load_model
from Model.my_model import Tweet
import logging
from Model.my_model import Tweet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.get_logger().setLevel(logging.ERROR)







loaded_model = load_model('LSTM_RNN_Model.h5')

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
    max_age=3600,
)


@app.get("/")
def get():
    return {"hello":"world"}

@app.post("/predict")
async def get_body(tweet: Tweet):
    predictionService = PredictionService(tweet.tweet_url)
    final_data = predictionService.prediction()
    pred_val = loaded_model.predict(final_data['string_embedded_doc'])
    print(f"prediction value : {pred_val}") 
    if pred_val > 0.6 :
        tweet.isDisaster = disasterDegree.sure
    elif pred_val<0.6 and pred_val>0.4 :
        tweet.isDisaster = disasterDegree.maybe
    else :
        tweet.isDisaster = disasterDegree.no
    return {"final text":final_data['final_text'],"isDisaster":tweet.isDisaster}




if __name__ == "__main__":
    loaded_model.summary()
    uvicorn.run("main:app", host="127.0.0.1", port=5000, log_level="info")
