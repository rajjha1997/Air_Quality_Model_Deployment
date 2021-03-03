import pandas as pd
from flask import Flask, render_template, url_for, request
import pickle

# load model
model = pickle.load(open('random_forest_regression_model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('real_2018.csv')

    # predictions
    # The order of inputs must match the order of columns
    # in the dataframe that you used to train your model
    # otherwise you will get an error when you try to make a prediction.
    # If the inputs you are receiving are not in the correct order
    # you can easily reorder them after you create the dataframe
    res = model.predict(df.iloc[:,:-1].values)

    res = res.tolist()

    return render_template('result.html',prediction = res)

if __name__ == '__main__':
	app.run(debug=True)
