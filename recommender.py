import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie dataset
df = pd.read_csv("movies.csv")

# Convert genres to lowercase and clean them
df['genres'] = df['genres'].str.replace('|', ' ', regex=False).str.lower()

# Ask the user for movies they have watched
user_movies = input("Enter movies you've watched (comma separated): ").strip().lower().split(',')

# Clean user input
user_movies = [movie.strip() for movie in user_movies]

# Filter watched movies from dataset
watched = df[df['title'].str.lower().isin(user_movies)]

if watched.empty:
    print("No matching movies found in database. Please check your input.")
    exit()

# Combine genres of watched movies to build user profile
user_profile = ' '.join(watched['genres'].tolist())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['genres'])

# Vector for the user's profile
user_vector = vectorizer.transform([user_profile])

# Compute cosine similarity
similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

# Add similarity scores to the dataframe
df['similarity'] = similarity_scores

# Exclude already watched movies
recommendations = df[~df['title'].str.lower().isin(user_movies)]

# Sort by similarity score
recommendations = recommendations.sort_values(by='similarity', ascending=False)

# Show top 5 recommendations
print("\nTop Movie Recommendations for You:\n")
print(recommendations[['title', 'genres']].head(5).to_string(index=False))
