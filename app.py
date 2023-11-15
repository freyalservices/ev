from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Get user input from the form
        battery = float(request.form['battery'])
        motor_temperature = float(request.form['temp'])
        motor_rpm = float(request.form['rpm'])
        range_val = float(request.form['range'])
        tyre_pressure = float(request.form['pressure'])
        brake_fluid = float(request.form['fluid'])
        coolant_level = float(request.form['coolant'])
        lubricant_level = float(request.form['lubricant'])

        df = pd.read_csv('dataset/random_data.csv')
        numeric_columns = df.select_dtypes(include='number')
        scaler = StandardScaler()
        numeric_columns_standardized = scaler.fit_transform(numeric_columns)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(numeric_columns_standardized)
        df['anomaly'] = model.predict(numeric_columns_standardized)
        joblib.dump(model, 'model.joblib')
        loaded_model = joblib.load('model.joblib')

        # Create a dictionary with user input
        user_input = {
            'Battery_Voltage_(Lead-acid)_(V)': battery,
            'Motor_Temperature': motor_temperature,
            'Motor_RPM': motor_rpm,
            'Range_(km)': range_val,
            'Tyre_Pressure_(psi)': tyre_pressure,
            'Brake_Fluid_(%)': brake_fluid,
            'Coolant_Level_(%)': coolant_level,
            'Lubricant_Level_(%)': lubricant_level
        }

        user_input_df = pd.DataFrame([user_input])
        user_input_standardized = scaler.transform(user_input_df)

        anomaly_label = loaded_model.predict(user_input_standardized)[0]
        anomaly_score = loaded_model.decision_function(user_input_standardized)[0]
        
        expected_date = ''
        if anomaly_label == 1:
            # Add three months (approximately 90 days) to the current date
            current_date = datetime.now()
            expected_date = (current_date + timedelta(days=90)).strftime('%Y-%m-%d')

        elif anomaly_label == -1:
            # Add 15 days to the current date
            current_date = datetime.now()
            expected_date = (current_date + timedelta(days=15)).strftime('%Y-%m-%d')

        # providing maintenace assistance
        df = pd.read_csv('dataset/random_data.csv')
        X = df.drop('Maintenance_Needed_(Area)', axis=1)
        y = df['Maintenance_Needed_(Area)']
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        input_data = pd.DataFrame({
            'Battery_Voltage_(Lead-acid)_(V)': [battery],
            'Motor_Temperature': [motor_temperature],
            'Motor_RPM': [motor_rpm],
            'Range_(km)': [range_val],
            'Tyre_Pressure_(psi)': [tyre_pressure],
            'Brake_Fluid_(%)': [brake_fluid],
            'Coolant_Level_(%)': [coolant_level],
            'Lubricant_Level_(%)': [lubricant_level]
        })

        # Make predictions using the trained model
        prediction = clf.predict(input_data)

        # Convert the numerical label back to the original category using LabelEncoder
        predicted_area = le.inverse_transform(prediction)
        assessment = predicted_area[0]

        # Render the template with the user input and anomaly result
        return render_template('index.html', anomaly_label=anomaly_label, anomaly_score=anomaly_score, user_input=user_input,expected_date=expected_date, assessment=assessment)

    return render_template('index.html')

def getKeyValue(key):
    key = key.replace("_","")
    return key
app.jinja_env.globals.update(getKeyValue=getKeyValue) 

if __name__ == '__main__':
    app.run(debug=True)