from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('training/CKD.pkl', 'rb'))

# Simple encoders as per training
binary_map = {'no': 0, 'yes': 1}
rbc_map = {'normal': 0, 'abnormal': 1}
pus_cell_map = {'normal': 0, 'abnormal': 1}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        red_blood_cells = rbc_map[request.form['red_blood_cells']]
        pus_cell = pus_cell_map[request.form['pus_cell']]
        blood_glucose_random = float(request.form['blood_glucose_random'])
        blood_urea = float(request.form['blood_urea'])
        pedal_edema = binary_map[request.form['pedal_edema']]
        anemia = binary_map[request.form['anemia']]
        diabetesmellitus = binary_map[request.form['diabetesmellitus']]
        coronary_artery_disease = binary_map[request.form['coronary_artery_disease']]

        input_data = [red_blood_cells, pus_cell, blood_glucose_random, blood_urea,
                      pedal_edema, anemia, diabetesmellitus, coronary_artery_disease]
        features = np.array(input_data).reshape(1, -1)
        prediction = model.predict(features)[0]
        if prediction == 1:
            return render_template('ckd_positive.html')
        else:
            return render_template('ckd_negative.html')
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
