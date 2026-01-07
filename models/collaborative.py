import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringModel:
    def __init__(self, matrix_path, user_means_path, movies_path):
        self.user_item_centered = pd.read_csv(matrix_path, index_col='user_id')
        self.user_means = pd.read_csv(user_means_path, index_col='user_id')
        self.movies = pd.read_csv(movies_path)
        self.user_sim_df = None

    def fit(self):
        user_sim_matrix = cosine_similarity(self.user_item_centered.fillna(0))
        self.user_sim_df = pd.DataFrame(
            user_sim_matrix, 
            index=self.user_item_centered.index, 
            columns=self.user_item_centered.index
        )
        print("CF Model: Đã xây dựng xong ma trận tương đồng User-User.")

    def predict_rating(self, user_id, movie_id):
        m_id = str(movie_id)
        if user_id not in self.user_item_centered.index or m_id not in self.user_item_centered.columns:
            return 0
        
        sim_scores = self.user_sim_df[user_id]
        movie_ratings = self.user_item_centered[m_id]
        
        rated_idx = movie_ratings.notna()
        if not rated_idx.any():
            return 0
            
        num = np.dot(sim_scores[rated_idx], movie_ratings[rated_idx])
        den = np.sum(np.abs(sim_scores[rated_idx]))
        
        return num / den if den != 0 else 0

    def recommend(self, user_id, top_n=10):

        user_ratings = self.user_item_centered.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index.tolist()
        

        predictions = []
        for m_id in unrated_movies:
            pred_score = self.predict_rating(user_id, m_id)
            predictions.append((m_id, pred_score))

        predictions.sort(key=lambda x: x[1], reverse=True)

        top_predictions = predictions[:top_n]
        top_movie_ids = [int(p[0]) for p in top_predictions]

        result = self.movies[self.movies['movie_id'].isin(top_movie_ids)]
        return result[['movie_id', 'title', 'genres_text']]


cf_model = CollaborativeFilteringModel(
    'user_item_matrix_centered.csv', 
    'user_mean_ratings.csv', 
    'movies_cleaned.csv'
)
cf_model.fit()

print(f"\nTop 10 phim gợi ý cho User {user_test} dựa trên Collaborative Filtering:")
print(cf_model.recommend(user_id=user_test, top_n=10))