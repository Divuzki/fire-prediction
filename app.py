from flask import Flask, render_template, request, session, jsonify, send_file, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import sys
import os
from libs.Database import Database as database
from libs.LSTM_FNN_Model import LSTM_FNN_Model
import base64
from io import BytesIO 


path = sys.path
parent = os.path.dirname(__file__)
loc = parent + '\libs'
try:
    path.index(loc)
except(ValueError):
    sys.path.append(loc)

app = Flask('__name__')
app.secret_key  = b'k843h/jd6uJU73R6778r6ibYGU'
model = LSTM_FNN_Model()
model_created = False

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    session['trained'] = False
    return render_template('login.html')

@app.route('/train_page', methods=['POST', 'GET'])
def train_page(): 
    if request.method == 'POST':
        home_directory = os.path.dirname(os.path.abspath(__file__)) 
        model.prepare_model(home_directory) 
        session['trained'] = True
        return render_template('train_page.html', training_complete=True, trained=True, message="Model training completed successfully.")
    
    
    trained = session.get('trained') if session.get('trained') else False
    return render_template('train_page.html', trained=trained, message='Model trained on this session')

@app.route('/evaluate_page', methods=['POST', 'GET'])
def evaluate_page():
    if request.method == 'POST':
        
        home_directory = os.path.dirname(os.path.abspath(__file__)) 
        model.prepare_model(home_directory)        
        loss, accuracy = model.evaluate_model(home_directory)
        
        calc = model.compute_feature_importance()
        print(calc)
        
        return render_template('evaluate_page.html', evaluation_complete=True, loss=loss, accuracy=accuracy)
    
    return render_template('evaluate_page.html')
    

@app.route('/predict_page', methods=['POST', 'GET'])
def predict_page():
    try:
        if request.method == 'POST':
            user_input = {key: float(value) for key, value in request.form.items()} 
            user_input_df = pd.DataFrame([user_input]) 
            
            home_directory = os.path.dirname(os.path.abspath(__file__))   
            prediction = model.predict_user_input(home_directory, user_input_df)
            prediction_label = "Fire Risk Detected" if prediction[0][0] > 0.5 else "No Fire Risk"
            
            print('prediction_label:', prediction_label)
            return render_template('predict_page.html', prediction=prediction_label)        
    except Exception as e:
        return render_template('predict_page.html', error=str(e))
    return render_template('predict_page.html')

    
@app.route('/profile')
def profile():
    if not session.get('details'):
        return render_template('login.html')

    user_details = session.get('details')
    username=user_details[2]
    name=user_details[5]
    email=user_details[6]
    department=user_details[8]
    phone=user_details[7]
    return render_template("profile.html", name=name, user=user_details, username=username, email=email, phone=phone, department=department)
  
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
 
'''
DATABASE
'''

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/forgot_login')
def forgot_login():
    return render_template("forgot_login.html")

@app.route('/update', methods=['POST', 'GET'])
def update():
    if request.method == 'POST':
        name = request.form.get('name')
        department = request.form.get('department')
        phone = request.form.get('phone')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        _id = request.form.get('id')

        from libs.Database import Database as db
        details = db().update(_id, name, department, phone, email, username, password)

        if len(details) > 0:
            session['details'] = details
            return render_template("profile.html", message="Profile updated successfully", name=details[5], user=details)
        elif len(details) == 0:
            return render_template("profile.html", message='Please check your password and try again.', user = session.get('details'))
        else:
            return render_template("profile.html", message='Unknown error', user= session.get('details'))
    else:
        return render_template("profile.html")

@app.route('/doregister', methods=['POST', 'GET'])
def doregister():
    if request.method == 'POST':
        name = request.form.get('name')
        department = request.form.get('department')
        phone = request.form.get('phone')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        from libs.Database import Database as db
        details = db().register(name, department, phone, email, username, password)

        if details == 'done' :
            return render_template("login.html", message="Registration was successfull")
        elif details == 'user_exits':
            return render_template("register.html", message='User with the same email exists.')
        else:
            return render_template("register.html")
    else:
        return render_template("register.html")


@app.route('/dologin', methods=['GET', 'POST'])
def dologin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        from libs.Database import Database as db
        details = db().login(username, password)

        if details and 'details' in details and len(details['details']) > 0:
            session['details'] = details['details']
            user_details = session.get('details')
            name = user_details[5] if len(user_details) > 5 else "User"

            data_db = db().getAll()
            ln = len(data_db)

            return render_template('train_page.html', history=data_db, size=ln, name=name)
        else:
            return render_template("login.html", message='Invalid user details')

    # Handle GET requests: Show the login page
    return render_template("login.html")

@app.route('/do_reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form.get('username')
        new_password = request.form.get('new_password')
        re_new_password = request.form.get('re_new_password')

        if not username:
            return render_template("forgot_login.html", message='Invalid user')

        if not new_password:
            return render_template("forgot_login.html", message='Please enter new password')

        if not re_new_password:
            return render_template("forgot_login.html", message='Enter same new password again')

        if new_password != re_new_password:
            return render_template("forgot_login.html", message='Both passwords do not match')


        from libs.Database import Database as db
        details = db().reset_password(username, new_password)


        if details == 'done' :
            return render_template("login.html", message="Password reset successfull")
        elif details == 'failed' :
            return render_template("forgot_login.html", message="Unknown user")
        else:
            return render_template("forgot_login.html", message="Unknwon error")


@app.route('/logout')
def logout():
    session['details'] = None
    return render_template("login.html")



if __name__ == '__main__':
    app.run(debug=True)
