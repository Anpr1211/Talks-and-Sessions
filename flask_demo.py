# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:32:52 2021

@author: ankit
"""

# importing libraries
from flask import Flask, request, jsonify
from pickle import load
import numpy as np

# defining the app
app = Flask(__name__)

def scale_x_reg(X):
    scaler_x = load(open(r'C:\Users\ankit\Downloads\scaler_x_reg.pkl', 'rb'))
    
    scaled_x = scaler_x.transform(np.array(X).reshape(-1, 1))
    return scaled_x

def scale_y_reg(pred):
    scaler_y = load(open(r'C:\Users\ankit\Downloads\scaler_y_reg.pkl', 'rb'))
    
    scaled_y = scaler_y.inverse_transform(pred)
    return scaled_y

@app.route('/inferreg', methods=['POST'])
def inferreg():
    global model_reg
    
    input = request.json['predictors']
    scaled_x = scale_x_reg(input)
    
    prediction = model_reg.predict(scaled_x.reshape(1, 10))
    scaled_pred = scale_y_reg(prediction)
    
    response = {'predicted': scaled_pred[0][0]}
    
    return jsonify(response)

def scale_x_logreg(X):
    scaler_x = load(open(r'C:\Users\ankit\Downloads\scaler_x_logreg.pkl', 'rb'))
    
    scaled_x = scaler_x.transform(np.array(X).reshape(-1, 1))
    return scaled_x

@app.route('/inferlogreg', methods=['POST'])
def inferlogreg():
    global model_logreg
    
    input = request.json['predictors']
    scaled_x = scale_x_reg(input)
    
    prediction = model_logreg.predict(scaled_x.reshape(1, 28))
    
    response = {'predicted': int(prediction[0])}
    
    return jsonify(response)

# run the app
if __name__ == '__main__':
    global model_reg
    model_reg = load(open(r'C:\Users\ankit\Downloads\model_reg.pkl', 'rb'))
    
    global model_logreg
    model_logreg = load(open(r'C:\Users\ankit\Downloads\model_logreg.pkl', 'rb'))
    
    app.run(host='127.0.0.1', port='8050', debug=True)