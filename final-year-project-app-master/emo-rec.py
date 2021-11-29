from flask import Flask, render_template, request
from LivePredictions import livePredictions
app = Flask(__name__)

pred_obj = None


def init_model():
   global pred_obj
   #model loaded from the app directory once at the beginning of the app
   pred_obj = livePredictions(path='Emotion_Voice_Detection_Model.h5')
   pred_obj.load_model()
   print("model loaded")

#default page rendered as the server is hit
@app.route('/')
def index():
   return render_template("website3.0.html")

#when the data is live recorded from the user, this route is hit. 
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   global pred_obj
   if request.method == 'POST':
      f = request.files['file']
      f.save("sentence.wav")
     
      pred_result = pred_obj.makepredictions("sentence.wav")
      return pred_result 

#when the audio file is uploaded, this route is hit.
@app.route('/fileupload', methods = ['GET', 'POST'])
def wav_upload():
   global pred_obj
   if request.method == 'POST':
      f = request.files['file']
      name = f.filename
      f.save((f.filename))
      pred_result = pred_obj.makepredictions((f.filename))
      return pred_result 

#starting point of flask app
if __name__ == '__main__':
   init_model()
   app.run(debug = False)