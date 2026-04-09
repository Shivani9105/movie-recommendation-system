import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

df = pd.read_csv("data/movies.csv")


df.columns = df.columns.str.lower()

df["overview"] = df["overview"].fillna("")
df["genre"] = df["genre"].fillna("")


df["combined"] = df["overview"] + " " + df["genre"]*2


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Fuzzy Matching Function

def find_closest_movie(name):
    name = name.lower()

    df["title_lower"] = df["title"].str.lower()

    match = difflib.get_close_matches(name, df["title_lower"], n=1)

    if match:
        # Return original title
        return df[df["title_lower"] == match[0]]["title"].values[0]

    return None



# Recommendation Function

def recommend(movie_name):
    # Step 1: Find closest movie
    movie_name = find_closest_movie(movie_name)

    if not movie_name:
        return ["Movie not found"]

    # Step 2: Case handling
    df["title_lower"] = df["title"].str.lower()
    movie_name_lower = movie_name.lower()

    if movie_name_lower not in df["title_lower"].values:
        return ["Movie not found"]

   
    idx = df[df["title_lower"] == movie_name_lower].index[0]

    
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    
    recommendations = []
    for i in scores[1:10]:
        recommendations.append(df.iloc[i[0]]["title"])

    return recommendations