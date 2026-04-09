from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation import recommend
from sentiment import predict_sentiment
from flask import send_file
from predict import predict_movie

app = Flask(__name__)
CORS(app)

# Home route
@app.route('/')
def home():
    return "CineVerse AI Backend Running!"

# EDA Route
@app.route('/chart')
def chart():
    import eda
    return jsonify({"msg": "Chart created"})

@app.route('/top_movies')
def top_movies():
    import eda
    return send_file('static/top_movies.png', mimetype='image/png')

@app.route('/rating_dist')
def rating_dist():
    import eda
    return send_file('static/rating_dist.png', mimetype='image/png')

@app.route('/language')
def language():
    import eda
    return send_file('static/language.png', mimetype='image/png')



# Recommendation API
@app.route('/recommend', methods=['POST'])
def get_recommend():
    data = request.json
    movie = data.get('movie')

    result = recommend(movie)
    return jsonify(result)

# Sentiment API
@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    data = request.json
    review = data.get('text')   

    result = predict_sentiment(review)
    return jsonify({"sentiment": result})   

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    rating = data.get('rating')
    genre = data.get('genre')
    year = data.get('year')

    result = predict_movie(rating, genre, year)

    return jsonify({"prediction": result})

# Run server
if __name__ == '__main__':
    app.run(debug=True)
 
    