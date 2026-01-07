import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display

class CollaborativeFilteringModel:
    def __init__(self, user_item_matrix, movies_df):
        self.user_item_centered = user_item_matrix.copy()
        self.user_item_centered.columns = self.user_item_centered.columns.astype(int)
        self.movies = movies_df.copy()
        self.movies['movie_id'] = self.movies['movie_id'].astype(int)
        self.user_sim_df = None

    def fit(self):
        user_sim_matrix = cosine_similarity(self.user_item_centered)
        self.user_sim_df = pd.DataFrame(
            user_sim_matrix,
            index=self.user_item_centered.index,
            columns=self.user_item_centered.index
        )
        print("CF Model: Đã xây dựng xong ma trận tương đồng.")

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.user_sim_df.index or movie_id not in self.user_item_centered.columns:
            return 0

        sim_scores = self.user_sim_df[user_id]
        movie_ratings = self.user_item_centered[movie_id]

        rated_idx = (movie_ratings != 0)
        
        if not rated_idx.any():
            return 0

        num = np.dot(sim_scores[rated_idx], movie_ratings[rated_idx])
        den = np.sum(np.abs(sim_scores[rated_idx]))

        return num / den if den != 0 else 0

    def recommend(self, user_id, top_n=10):

        user_ratings = self.user_item_centered.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()

        if not unrated_movies:
            return "User này đã xem hết phim!"

        print(f"Đang tính toán dự báo cho {len(unrated_movies)} phim...")
        predictions = []
        for m_id in unrated_movies:
            score = self.predict_rating(user_id, m_id)
            predictions.append((m_id, score))
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_movie_ids = [p[0] for p in predictions[:top_n]]


        result = self.movies[self.movies['movie_id'].isin(top_movie_ids)]
        

        result['movie_id'] = result['movie_id'].astype('category')
        result['movie_id'] = result['movie_id'].cat.set_categories(top_movie_ids, ordered=True)
        return result.sort_values('movie_id')[['movie_id', 'title', 'genres_text']]

cf_model = CollaborativeFilteringModel(user_item_centered, movies)
cf_model.fit()

kq = cf_model.recommend(user_id=1, top_n=10)
display(kq)