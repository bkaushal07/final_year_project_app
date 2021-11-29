import keras
import numpy as np
import librosa

class livePredictions:

    def __init__(self, path):

        self.path = path

    def load_model(self):
        '''
        Loading h5 model of saved CNN(trained for 2000 epochs).
        '''
        self.loaded_model = keras.models.load_model(self.path)
        self.loaded_model._make_predict_function()
        return self.loaded_model.summary()

    def makepredictions(self,file_name):
        '''
        Processing the recorded/uploaded files and extracts the key MFCC features, needed to properly represent the audio.
        '''
        data, sampling_rate = librosa.load(file_name)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print(predictions)
        print( "Prediction is", " ", self.convertclasstoemotion(predictions))

        return self.convertclasstoemotion(predictions)

    def convertclasstoemotion(self, pred):
        '''
        Converts Numeric classes to the name of the corressponding emotion.
        '''
        self.pred  = pred

        if pred == 0:
            pred = "NEUTRAL"
            return pred
        elif pred == 1:
            pred = "CALM"
            return pred
        elif pred == 2:
            pred = "HAPPY"
            return pred
        elif pred == 3:
            pred = "SAD"
            return pred
        elif pred == 4:
            pred = "ANGRY"
            return pred
        elif pred == 5:
            pred = "FEARFUL"
            return pred
        elif pred == 6:
            pred = "DISGUST"
            return pred
        elif pred == 7:
            pred = "SURPRISED"
            return pred

if __name__ == "__main__":
    '''
    Pre-saved model, which is stored as a h5 file in the directory is loaded as soon as the app starts. 
    This prevents time consumption, every time the model has to predict the emotion, as the model needs to be loaded only once.
    '''
    pred = livePredictions(path='Emotion_Voice_Detection_Model.h5')

    pred.load_model()
    pred.makepredictions()