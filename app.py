import joblib
import os
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load Model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'breast_cancer_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    result_class = ""
    
    if request.method == 'POST':
        try:
            # Inputs matching the model's feature order
            features = [
                float(request.form['clump_thickness']),
                float(request.form['cell_size']),
                float(request.form['cell_shape']),
                float(request.form['marginal_adhesion']),
                float(request.form['mitoses'])
            ]
            
            final_features = np.array([features])
            prediction = model.predict(final_features)
            
            # 0 = Benign, 1 = Malignant
            if prediction[0] == 1:
                prediction_text = "MALIGNANT (High Risk)"
                result_class = "danger"
            else:
                prediction_text = "BENIGN (Safe)"
                result_class = "safe"
                
        except Exception as e:
            prediction_text = f"Error: {str(e)}"
            result_class = "error"

    return render_template('index.html', prediction_text=prediction_text, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)