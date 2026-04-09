import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/movies.csv")
df.columns = df.columns.str.lower()


# LABEL (TARGET)

def label_movie(rating):
    if rating >= 8:
        return "Hit"
    elif rating >= 6:
        return "Average"
    else:
        return "Flop"

df["status"] = df["rating"].apply(label_movie)


# GENRE TO NUMBER

df["genre_code"] = df["genre"].astype("category").cat.codes


# FEATURES & TARGET

X = df[["rating", "genre_code", "year"]]
y = df["status"]


# TRAIN MODEL

model = DecisionTreeClassifier()
model.fit(X, y)


# PREDICTION FUNCTION

def predict_movie(rating, genre, year):
    # Convert genre to code
    genre_map = dict(zip(df["genre"], df["genre_code"]))

    genre_code = genre_map.get(genre, 0)

    result = model.predict([[rating, genre_code, year]])

    return result[0]