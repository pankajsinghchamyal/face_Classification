import os
import mtcnn
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
from PIL import Image


detector = mtcnn.MTCNN()

# extract a single face from a given photograph
def extract_face(file_path):

    image = Image.open(file_path)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    return results,pixels

def cut_face(results, pixels, required_size=(160, 160)):

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array