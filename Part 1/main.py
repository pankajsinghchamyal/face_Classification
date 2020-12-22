import os
import mtcnn
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

import tensorflow as tf
import crop_image
import classifier
# import embedding

base_dir = r"/home/pankaj/Desktop/veerbm/trainset"

def main(path):

    X, y = list(), list()
    # X_val , y_val = list(), list()
    for maindir in os.listdir(base_dir):
        main_dir_path = os.path.join(base_dir, maindir)
        for subdir in os.listdir(main_dir_path):
            sub_dir_path = os.path.join(main_dir_path, subdir)
            id_person = str(subdir)
            for person in os.listdir(sub_dir_path):
                person_path = os.path.join(sub_dir_path , person)
                print(person_path)
                
                results,pixels = crop_image.extract_face(person_path)

                if len(results)==1:

                    face_array = crop_image.cut_face(results,pixels)
                    y.append(id_person)
                    X.append(face_array)
    
    savez_compressed('trainset_faces-dataset.npz', X, y)
    # load the face dataset
    data = load('trainset_faces-dataset.npz',allow_pickle=True)
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Loaded: ', trainX.shape, trainy.shape)
    
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face_pixels in trainX:
        embedding = get_embedding(face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    
    savez_compressed('trainset_faces_embeddings.npz', newTrainX, trainy)
    classifier.cassify()
    
main(base_dir)