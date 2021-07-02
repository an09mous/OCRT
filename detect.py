import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import pandas as pd
from loader import Batch

dataset=mnist.load_data()
(X_train,y_train),(X_test,y_test)=dataset

X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

from keras.utils import to_categorical
#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

model.predict(X_test[:4])
y_test[:4]
from prep import Preprocessor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
X=[]
y=[]
n=0
with open('trainingdata.txt', 'rt') as training_data:
    n=int(training_data.readline())
    for line in training_data:
        y.append(int(line[0]))
        X.append(line[2:])
    
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
clf=MultinomialNB()
classifier=Pipeline([('vect', CountVectorizer(stop_words='english',max_features=1000)),
                     ('tfidf',TfidfTransformer()),
                     ('clf',AdaBoostClassifier(base_estimator=clf))])
classifier.fit(X,y)
x=[]
for i in range(int(input())):
    x.append(input())
    
    
from typing import Tuple, List
from nn_lite import *



from model import Model, DecoderType



def solve(img):
    decoder_type = DecoderType.BestPath
    model = Model(list(open(FilePaths.fn_char_list).read()), decoder_type, must_restore=True)
    return infer(model, img)


def get_img_height() -> int:
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def infer(model, fn_img):
    img = fn_img
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)

    return recognized

