
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 13:42:30 2019

@author: alinemati
https://www.dropbox.com/s/p0ry8m4jg0vfdav/default.zip

"""


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask import Flask, request, render_template
from werkzeug import secure_filename
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')




#this part is for RFNL ############################

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    
    df= pd.read_csv("RNFL.csv")
    data_selection = df.iloc[:,3:1029].fillna(0)
    #print(data_selection.head())
    X_train=data_selection.iloc[:,0:1024]
    
    y_train_Gender=data_selection['Gender']
    y_train_disease=data_selection['disease']
    #print(X.head())
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(X_train.shape  , X_test.shape, y_test.shape )
	#Naive Bayes Classifier
    clf_Gender =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_Gender)
    clf_disease =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_disease)

    #clf.score(X_test,y_test)
    if request.method == 'POST':
              try:
                  data = pd.read_csv(request.files.get('file'), header=None)
                  data_selection = data.iloc[:,0:1024].fillna(0)
                  #print(df)
                  #f.save(secure_filename(f.filename))
                  #print(type(df))
                  my_prediction_gender = clf_Gender.predict(data_selection) # maximum of data
                  #my_prediction = clf.predict_proba(data)
                  my_prediction_disease = clf_disease.predict(data_selection)
          #print(my_prediction)
                  return render_template('result.html',prediction_gender = my_prediction_gender[0] , prediction_disease = my_prediction_disease[0])
                  
              except ValueError:
                   return   "Oops!  That was no csv file.  Try again..."
      #f =request.files['file']

      #return "we are here. "

    else:
      return render_template('home.html')
#End of RFNL ############################



      
  
#this part is for GCL+ ############################


def uploader_GCL():
    
    df= pd.read_csv("GCL+.csv")
    data_selection = df.iloc[:,3:1029].fillna(0)
    
    #print(data_selection.head())
    X_train=data_selection.iloc[:,0:1024]
    
    y_train_Gender=data_selection['Gender']
    y_train_disease=data_selection['disease']
    #print(X.head())
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(X_train.shape  , X_test.shape, y_test.shape )
	#Naive Bayes Classifier
    clf_Gender =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_Gender)
    clf_disease =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_disease)

    #clf.score(X_test,y_test)
    if request.method == 'POST':
      #f =request.files['file']
              try:
                  data = pd.read_csv(request.files.get('file'), header=None)
                  data_selection = data.iloc[:,0:1024].fillna(0)
                  #print(df)
                  #f.save(secure_filename(f.filename))
                  #print(type(df))
                  my_prediction_gender = clf_Gender.predict(data_selection) # maximum of data
                  #my_prediction = clf.predict_proba(data)
                  my_prediction_disease = clf_disease.predict(data_selection)
          #print(my_prediction)
                  return render_template('result.html',prediction_gender = my_prediction_gender[0] , prediction_disease = my_prediction_disease[0])
                  
              except ValueError:
                   return   "Oops!  That was no csv file.  Try again..."
   
    else:
      return render_template('home.html')
#End of GCL+ ############################






  
#this part is for GCL++ ############################


def uploader_GCL__():
    
    df= pd.read_csv("GCL++.csv")
    data_selection = df.iloc[:,3:1029].fillna(0)
    #print(data_selection.head())
    X_train=data_selection.iloc[:,0:1024]
    
    
    
    y_train_Gender=data_selection['Gender']
    y_train_disease=data_selection['disease']
    #print(X.head())
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print(X_train.shape  , X_test.shape, y_test.shape )
	#Naive Bayes Classifier
    clf_Gender =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_Gender)
    clf_disease =  DecisionTreeClassifier(random_state=0).fit(X_train,y_train_disease)

    #clf.score(X_test,y_test)
    if request.method == 'POST':
              try:
                  data = pd.read_csv(request.files.get('file'), header=None)
                  data_selection = data.iloc[:,0:1024].fillna(0)
                  #print(df)
                  #f.save(secure_filename(f.filename))
                  #print(type(df))
                  my_prediction_gender = clf_Gender.predict(data_selection) # maximum of data
                  #my_prediction = clf.predict_proba(data)
                  my_prediction_disease = clf_disease.predict(data_selection)
          #print(my_prediction)
                  return render_template('result.html',prediction_gender = my_prediction_gender[0] , prediction_disease = my_prediction_disease[0])
                  
              except ValueError:
                   return   "Oops!  That was no valid csv file.  Try again..."
   
    else:
      return render_template('home.html')
#End of GCL++ ############################





if __name__ == '__main__':
	#app.run(debug=True)
    app.run(host='0.0.0.0', debug=True, port=5000)
