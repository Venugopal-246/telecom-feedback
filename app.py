from flask import Flask, jsonify, render_template, request
import pandas as pd
from utils.sentiment import get_sentiment

app = Flask(__name__)
df = pd.read_csv('data/customers_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

# General summary endpoint
def filter_location(df, loc):
    return df[df['Location'] == loc] if loc else df

@app.route('/gender_summary')
def gender_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    return jsonify(data['gender'].value_counts().to_dict())

@app.route('/senior_summary')
def senior_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    return jsonify(data['SeniorCitizen'].value_counts().to_dict())

@app.route('/internet_summary')
def internet_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    return jsonify(data['InternetService'].value_counts().to_dict())

@app.route('/contract_summary')
def contract_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    return jsonify(data['Contract'].value_counts().to_dict())

@app.route('/tickets_summary')
def tickets_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    return jsonify(data[['numAdminTickets','numTechTickets']].to_dict(orient='records'))

@app.route('/feedback_summary')
def feedback_summary():
    loc = request.args.get('Location', '')
    data = filter_location(df, loc)
    sentiments = data['CustomerFeedback'].apply(get_sentiment)
    return jsonify(sentiments.tolist())

if __name__ == '__main__':
    app.run(debug=True)
