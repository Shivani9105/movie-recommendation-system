import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/movies.csv")


# Top 10 Movies by Rating

top_movies = df.sort_values(by="rating", ascending=False).head(10)

plt.figure()
plt.barh(top_movies["title"], top_movies["rating"])
plt.xlabel("Rating")
plt.title("Top 10 Movies by Rating")
plt.tight_layout()
plt.savefig("static/top_movies.png")
plt.close()



# Rating Distribution

plt.figure()
plt.hist(df["rating"], bins=10)
plt.xlabel("Rating")
plt.title("Rating Distribution")
plt.tight_layout()
plt.savefig("static/rating_dist.png")
plt.close()



# Language Distribution

plt.figure()
df["language"].value_counts().plot(kind='bar')
plt.xlabel("Language")
plt.title("Language Distribution")
plt.tight_layout()
plt.savefig("static/language.png")
plt.close()