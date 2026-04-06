import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/movies.csv")

# Avoid errors
df.columns = df.columns.str.lower()

# Fill missing values
df["overview"] = df["overview"].fillna("")
df["genre"] = df["genre"].fillna("")

# Combine features
df["combined"] = df["overview"] + " " + df["genre"]

# Convert text → numbers
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()

    # Make title case-insensitive
    df["title_lower"] = df["title"].str.lower()

    if movie_name not in df["title_lower"].values:
        return ["Movie not found"]

    idx = df[df["title_lower"] == movie_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in scores[1:10]:  # top 9 movies
        recommendations.append(df.iloc[i[0]]["title"])

    return recommendations