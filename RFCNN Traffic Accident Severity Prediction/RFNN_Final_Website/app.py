from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__, template_folder='./templates')

# Load the trained model
model = joblib.load('rfcnn.model')
X_test=pd.read_csv('X_test.csv')
y_test=pd.read_csv('y_test.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics.html')
def analytics():
    return render_template('analytics.html')

@app.route('/map.html')
def map():
    return render_template('map.html')

@app.route('/models.html')
def models():
    return render_template('models.html')

@app.route('/random_forest.html')
def random_forest():
    return render_template('random_forest.html')

@app.route('/xgboost.html')
def xg_boost():
    return render_template('xgboost.html')

@app.route('/cnn.html')
def cnn():
    return render_template('cnn.html')

@app.route('/voting.html')
def voting():
    return render_template('voting.html')

@app.route('/logistic.html')
def logistic():
    return render_template('logistic.html')

@app.route('/rfcnn.html')
def rfcnn():
    return render_template('rfcnn.html')

@app.route('/knn.html')
def knn():
    return render_template('knn.html')

@app.route('/adaboost.html')
def adaboost():
    return render_template('adaboost.html')

@app.route('/decision.html')
def decision():
    return render_template('decision.html')

@app.route('/bbc.html')
def bbc():
    return render_template('bbc.html')

@app.route('/naives.html')
def naives():
    return render_template('naives.html')


@app.route('/features.html')
def features():
    return render_template('features.html')

@app.route('/traffic_map.html')
def traffic():
    return render_template('traffic_map.html')

@app.route('/predict.html')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is CSV
    if file.filename.split('.')[-1].lower() != 'csv':
        return jsonify({'error': 'File must be a CSV'})

    # Read the CSV file
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': 'Error reading CSV file', 'details': str(e)})

    # Make predictions
    try:
        predictions = model.predict(df.values)
        #accuracy = model.score(X_test, y_test) # Assuming you have X_test and y_test datasets
    except Exception as e:
        return jsonify({'error': 'Error making predictions', 'details': str(e)})

    return jsonify({'predictions': predictions.tolist()})#, 'accuracy': accuracy

if __name__ == '__main__':
    app.run(debug=True)
