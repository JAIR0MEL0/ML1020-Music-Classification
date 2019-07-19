from flask import Flask,render_template,url_for,request
import numpy as np
import librosa as lbr
import tensorflow.keras.backend as K
import warnings

import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

def load_track():
        filename  = 'audio.mp3'
        enforce_shape=None

        WINDOW_SIZE = 2048
        WINDOW_STRIDE = WINDOW_SIZE // 2
        N_MELS = 128
        MEL_KWARGS = {
                'n_fft': WINDOW_SIZE,
                'hop_length': WINDOW_STRIDE,
                'n_mels': N_MELS
                }
        warnings.filterwarnings('ignore')
        new_input, sample_rate = lbr.load(filename, mono=True, duration=40.0)
        features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T
        if enforce_shape is not None:
                if features.shape[0] < enforce_shape[0]:
                        delta_shape = (enforce_shape[0] - features.shape[0],
                                       enforce_shape[1])
                        features = np.append(features, np.zeros(delta_shape), axis=0)
                elif features.shape[0] > enforce_shape[0]:
                        features = features[: enforce_shape[0], :]
        features[features == 0] = 1e-6
        return (np.log(features), float(new_input.shape[0]) / sample_rate)

def predict_data():
        path = 'audio.mp3'
        tmp_features, _ = load_track()
        default_shape = tmp_features.shape
        x = np.zeros((1,) + default_shape, dtype=np.float32)            
        x[0], _ = load_track()
        return (x)

@app.route('/',methods=['POST'])
def predict():
        GENRES = ['Electronic', 'Experimental', 'Folk', 'HipHop', 'Instrumental', 'International', 'Pop','Rock']
        from tensorflow.keras.models import Model, load_model
        genre_model = Model()
        genre_model = load_model('model_40s.h5')
        if request.method == 'POST':
                pred_file='audio.mp3'
                x = predict_data()
                predictions = genre_model.predict(x)
                my_prediction = GENRES[np.argmax(predictions[0])]
                print (my_prediction)
        return (my_prediction)


if __name__ == '__main__':
        app.run(debug=True)
