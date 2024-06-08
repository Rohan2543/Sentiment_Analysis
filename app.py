from distutils.log import debug
from fileinput import filename
import os
import io
from os import environ
from flask import *
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask_cors import CORS
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
import Preforcessing
# Example using NLTK for data cleaning
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Example using TfidfVectorizer
model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

# Load the model from the file
loaded_model = joblib.load('sentiment_model.joblib')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Remove punctuation and convert to lowercase
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    
    # Remove stop words and apply stemming
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    
    return text

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="feedback"
)
mycursor = mydb.cursor()


app = Flask(__name__)
CORS(app)
app.secret_key = "abc"

@app.route('/')  
def main():
    #return render_template("signin.html")
    Respon=make_response("hii")
    return Respon

@app.route('/signin')  
def signin():  
    Respon=make_response("hii")
    return Respon

@app.route('/success', methods = ['POST','GET'])  
def success():
    #http://192.168.137.27:5555/success
    Respondata=""
    Strval="Select * from feedbacktb where Feedbacktype!='positive' and Feedbacktype!='negative'"
    mycursor.execute(Strval)
    myresult = mycursor.fetchall()
    for x in myresult:
        user_input=str(x[2])

        user_input = preprocess_text(user_input)
        user_review = [user_input]

        # Predict sentiment for user input
        user_pred = loaded_model.predict(user_review)

        # Print the predicted sentiment
        print(f"Review: {str(x[2])}")
        print(f"Predicted sentiment: {user_pred[0]}")

        sql="update feedbacktb set Feedbacktype='"+user_pred[0]+"' where FID="+str(x[0])
        mycursor.execute(sql)
        mydb.commit()
        Respondata="Successfully"       

    Respon=make_response(Respondata)
    return Respon

@app.route('/Mainpage', methods=['GET'])  
def Mainpage():
    Respon=make_response("")
    #return Respon
    #return render_template("Mainpage.html")
    return Respon
            
@app.route('/shutdown')
def shutdown():
    sys.exit()
    os.exit(0)
    return
   
if __name__ == '__main__':
   HOST = environ.get('SERVER_HOST', '0.0.0.0')
   #HOST = environ.get('SERVER_HOST', 'localhost')
   try:
      PORT = int(environ.get('SERVER_PORT', '5555'))
   except ValueError:
      PORT = 5555
   app.run(HOST, PORT)
   #app.run(debug=True)
