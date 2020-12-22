from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from PIL import Image

from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import joblib

def classify():

    # load dataset
    data = load('trainset_faces_embeddings.npz')
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Dataset: train=%d' % (trainX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)

    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    
    # save train model
    joblib.dump(model,"face_id_model.sav")
    
    # predict
    yhat_train = model.predict(trainX)

    # score
    score_train = accuracy_score(trainy, yhat_train)

    # summarize
    print('Accuracy: train=%.3f' %(score_train*100))

def test_classify():
    
    # load dataset
    data = load('testset_faces_embeddings.npz')
    testX, testy = data['arr_0'], data['arr_1']
    print('Dataset: train=%d' % (testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(testy)
    testy = out_encoder.transform(testy)

    # load save model
    
    model = joblib.load("face_id_model.sav")
    
    # predict
    yhat_test = model.predict(testX)

    # score
    score_test = accuracy_score(testy, yhat_test)

    # summarize
    print('Accuracy: test=%.3f' %(score_test*100))

    
    