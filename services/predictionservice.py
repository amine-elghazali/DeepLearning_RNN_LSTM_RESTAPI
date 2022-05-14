VOC_SIZE = 20000
sent_length = 100

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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


def remove_emojis(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_stopwords(text):
    stop = stopwords.words('english')
    return " ".join([word for word in text.split() if word not in (stop)])


def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)


def clean_text(text):

    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = remove_html(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text


class PredictionService:
    s = ""

    def __init__(self, text_pred):
        self.s = text_pred

    def nlp_preproc(self):
        self.s = clean_text(self.s)
        self.s = word_tokenize(self.s)
        print(self.s)
        lemma = WordNetLemmatizer()
        for index, word in enumerate(self.s):
            self.s[index] = lemma.lemmatize(word)
        self.s = ' '.join(self.s)
        print(self.s)

    def prediction(self):
        self.nlp_preproc()
        print("prediction meth :")
        string_one_hot = one_hot(self.s, VOC_SIZE)
        print(string_one_hot)
        string_embedded_doc = pad_sequences([string_one_hot], padding='pre', maxlen=sent_length)
        print(string_embedded_doc)
        print(self.s)

        return {"final_text":self.s,"string_embedded_doc":string_embedded_doc}
