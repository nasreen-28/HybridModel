from flask import Flask,render_template, url_for, request , redirect
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 
#from application.score import score
#from sklearn import linear_model


from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import pandas as pd
import json

app=Flask(__name__,template_folder='admission',static_folder='admission')

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
  return render_template('predict.html')


@app.route('/calc',methods=['GET','POST'])
def calc():
  if request.method=="POST":
    location=request.form['location']
    cutoff=request.form['cutoff']
    department=request.form['department']
    autonomous=request.form['autonomous']
    datapreprocessing(location,cutoff,department,autonomous)    
  return render_template('answer.html',college=out_college,chance=out_coa)


def datapreprocessing(location,cutoff,department,autonomous):
  with open('Admission.csv') as file:  
      df1=pd.read_csv(file)
      df2=df1.dropna()
      global label_name
      label_dist = preprocessing.LabelEncoder()
      label_dept = preprocessing.LabelEncoder()
      label_name = preprocessing.LabelEncoder()
      global dataf
      dataf=df2
      newdist=df2[['DISTRICT']].copy()
      newdept=df2[['Department']].copy()
      dataf.rename(columns={'DISTRICT': 'ENC_DISTRICT'}, inplace=True)
      dataf.rename(columns={'Department': 'ENC_Department'}, inplace=True)
      dataf['ENC_DISTRICT']= label_dist.fit_transform(df2['ENC_DISTRICT']) 
      dataf['ENC_Department']= label_dept.fit_transform(df2['ENC_Department'])
      dataf['College_Name']= label_name.fit_transform(df2['College_Name']) 
      newdist[['ENC_DISTRICT']]=dataf[['ENC_DISTRICT']]
      newdist.drop_duplicates()
      list_dist=newdist.set_index('DISTRICT').T.to_dict('records')
      dict_dist=list_dist[0]
      newdept[['ENC_Department']]=dataf[['ENC_Department']]
      newdept.drop_duplicates()
      list_dept=newdept.set_index('Department').T.to_dict('records')
      dict_dept=list_dept[0]
      in_cutoff=int(cutoff)
      in_autonomous=int(autonomous)
      in_location = dict_dist[location]
      in_department = dict_dept[department]
      input=[[in_location,in_autonomous,in_department,in_cutoff]]
      in_college=predictCollegeName(input)
      input=[[in_college,in_location,in_autonomous,in_department,in_cutoff]]
      global out_college
      global out_coa
      in_coa=predictChanceOfAdmit(input)
      out_college =label_name.inverse_transform(in_college)
      print(out_college)
      out_college=out_college[0]
      out_coa = in_coa[0]
      out_coa=out_coa*100
      out_coa=round(out_coa,2)


def predictCollegeName(input):
  dv=dataf.iloc[:,[2,3,4,5]].values
  iv=dataf.iloc[:,1].values
  #x, y = make_classification(n_samples=1000, n_features=4,
                        # n_informative=2, n_redundant=0,
                        # random_state=0, shuffle=False)
  #print(x)
  x_train, x_test, y_train, y_test = train_test_split(dv, iv, test_size=0.2,random_state=42)
  #classifier = MLPClassifier(hidden_layer_sizes=(30,20,10), max_iter=300,activation = 'relu',solver='adam',random_state=1)
  classifier = MLPClassifier(hidden_layer_sizes=(50,40),
                             activation='logistic',
                             solver='adam',    
                            random_state=5,
                            verbose=True,
                            learning_rate_init=0.01)
  classifier.fit(dv, iv)
  y_pred = classifier.predict(input)
  in_college=y_pred[0]
  print(classifier.score(x_test,y_test))
  #predictChanceOfAdmit(input,in_college)
  #ans =label_name.inverse_transform(y_pred)
  #print(ans)
  return y_pred

  #print(input)

def predictChanceOfAdmit(input):
  #print(input)
  x=dataf.iloc[:,[1,2,3,4,5]].values
  y=dataf.iloc[:,6].values
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  #regr = linear_model.LinearRegression().fit(x_train, y_train)
  regr = MLPRegressor(hidden_layer_sizes=(50,40),random_state=1, max_iter=500).fit(x_train, y_train)
  #clf=LogisticRegression(random_state=0).fit(x_train, y_train)
  y_pred=regr.predict(input)
  print(y_pred)
  print(regr.score(x_test,y_test))
  return y_pred
  #print(y)









if __name__ == "__main__":
    app.run()


