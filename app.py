from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

app = Flask(__name__)

# Load the Keras model and scaler
model = tf.keras.models.load_model('model.h5')
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Ensure the path to your dataset is correct
ipl_data = pd.read_csv('ipl_data.csv')
unique_batsmen = ipl_data['batsman'].unique()
unique_bowlers = ipl_data['bowler'].unique()


@app.route('/')
def index():
    return render_template('index.html', unique_batsmen=unique_batsmen, unique_bowlers=unique_bowlers)

# Define the prediction route


@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    venue = int(request.form['venue'])
    bat_team = int(request.form['bat_team'])
    bowl_team = int(request.form['bowl_team'])
    batsman = int(request.form['batsman'])
    bowler = int(request.form['bowler'])

    # Combine the data into a NumPy array for the model input
    input_data = np.array([[venue, bat_team, bowl_team, batsman, bowler]])

    # Apply the same scaling that was used during model training
    input_data_scaled = scaler.transform(input_data)

    # Predict the total using the loaded model
    prediction = model.predict(input_data_scaled)

    # Return the prediction and render the result in the HTML
    predicted_score = float(prediction[0][0])
    return render_template('index.html', prediction=predicted_score,
                           unique_batsmen=unique_batsmen, unique_bowlers=unique_bowlers)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
