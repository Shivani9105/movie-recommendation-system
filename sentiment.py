import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv('data/reviews.csv')

# Input and output
X = df['review']
y = df['sentiment']

# Convert text to numbers
cv = TfidfVectorizer()
X = cv.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Function to predict sentiment
def predict_sentiment(text):
    text = cv.transform([text])
    return model.predict(text)[0]


# Test
