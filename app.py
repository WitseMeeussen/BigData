from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import re


import random as rnd
from fastai import *
from fastai.vision.all import *

# fixes a dependancy on PosixPath by load_learner
import pathlib


app = Flask(__name__)


# learn is the model that is loaded from the pkl file. this file contains a model that predicts 14 different archtectal styles


def format_example(imgpath,cat,prediction):
   return {"image":"/data/"+imgpath,
                           "category":cat,
                           "prediction":prediction[0],
                           "correct_class":"example_"+str(cat==prediction[0])}

def format_sample(imgpath,prediction):
   return {"image":"/UploadFolder/"+imgpath,
            "prediction":prediction[0],
            "sample_class":"sample_visable"}

def get_examples():
   examples = []
   for cat in app.config['CATEGORIES']:
      n = rnd.choice(range(app.config['N_MAX']))
      imgpath = cat+"/"+cat+str(n)+".png"
      prediction = app.config['learn'].predict(app.config['DATAPATH']+imgpath)

      examples.append(format_example(imgpath,cat,prediction))
   
   return examples

@app.route('/')
def home():
   examples = get_examples()
   return render_template('home.html',examples=examples,sample={"sample_class":"sample_hidden"})

@app.route("/testImage_post", methods=['post'])
def testImage_post():

   # clears the upload folder, to prevent to much space being used
   for img in os.listdir(app.config['UPLOAD_FOLDER']):
      os.remove(os.path.join(app.config['UPLOAD_FOLDER'],img))

   # requests file from the posted files, this is suposed to be an image
   image = request.files['file']
   imgpath = secure_filename(image.filename)

   #regex function to see if uploaded file was an image, else return to home function
   if re.sub('.*\\.','',imgpath) not in app.config['ALLOWED_IMGTYPES']:
      return redirect(url_for('home'))

   image.save(os.path.join(app.config['UPLOAD_FOLDER'], imgpath))

   #test the posted image with the model
   prediction = learn.predict(os.path.join(app.config['UPLOAD_FOLDER'], imgpath))

   sample = format_sample(imgpath,prediction)
   examples = get_examples()
   return render_template('home.html',examples=examples,sample=sample)

if __name__ == '__main__':
   temp = pathlib.PosixPath
   pathlib.PosixPath = pathlib.WindowsPath
   app.config['UPLOAD_FOLDER'] = './static/UploadFolder'
   app.config['DATAPATH']= "./static/data/"
   # These are the categories used in the model and the data
   app.config['CATEGORIES'] = os.listdir("./static/data/")
   app.config['ALLOWED_IMGTYPES'] = ['png','jpeg','gif','jfif','jpg']
   # uses top 20 images from the scrapers
   app.config['N_MAX'] = 20

   app.config['learn'] = load_learner('exportBig.pkl')
   app.run()