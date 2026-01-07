import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringModel:
    def __init__(self, matrix_path, user_means_path):
        self.user_item_centered = pd.read_csv(matrix_path, index_col='user_id')
        self.user_means = pd.read_csv(user_means_path, index_col='user_id')
        self.user_sim_df = None

    def fit(self):
        user_sim_matrix = cosine_similarity(self.user_item_centered.fillna(0))
        self.user_sim_df = pd.DataFrame(
            user_sim_matrix, 
            index=self.user_item_centered.index, 
            columns=self.user_item_centered.index
        )
        print("CF Model: Đã xây dựng xong ma trận User-User Similarity.")

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

    def get_full_prediction(self, user_id, movie_id):
        pred_centered = self.predict_rating(user_id, movie_id)
        user_mean = self.user_means.loc[user_id, 'mean_rating']
        return pred_centered + user_mean

cf_model = CollaborativeFilteringModel('user_item_matrix_centered.csv', 'user_mean_ratings.csv')
cf_model.fit()
u_id, m_id = 1, 10
pred = cf_model.get_full_prediction(u_id, m_id)
print(f"\nDự đoán số sao User {u_id} sẽ chấm cho Movie {m_id}: {pred:.2f}")