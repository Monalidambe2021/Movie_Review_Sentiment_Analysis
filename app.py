# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the support vector classifier model and TfidfVectorizer object from disk
filename = 'movie_review_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('count-Vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data)
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)