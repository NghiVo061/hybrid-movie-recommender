import pandas as pd
import joblib
import os
import numpy as np
from typing import List, Dict


class CollaborativeModel:
    """
    USER-BASED COLLABORATIVE FILTERING
    Mean-Centered + Cosine Similarity
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')

        self.data_dir = data_dir
        self.common_dir = os.path.join(os.path.dirname(data_dir), 'common')

        self.movies = None
        self.user_item_matrix = None          # centered matrix
        self.user_sim_matrix = None
        self.user_mean_ratings = None
        self.is_ready = False

        self._load_data()

    def _load_data(self):
        print(">> [CF] Loading data...")
        try:
            self.movies = pd.read_csv(
                os.path.join(self.common_dir, 'movies_cleaned.csv')
            )

            self.user_item_matrix = pd.read_csv(
                os.path.join(self.data_dir, 'user_item_matrix_centered.csv'),
                index_col=0
            )

            self.user_sim_matrix = joblib.load(
                os.path.join(self.data_dir, 'user_similarity_matrix.pkl')
            )

            self.user_mean_ratings = pd.read_csv(
                os.path.join(self.data_dir, 'user_mean_ratings.csv'),
                index_col=0
            )

            # Ép kiểu an toàn
            self.user_item_matrix.index = self.user_item_matrix.index.astype(int)
            self.user_item_matrix.columns = self.user_item_matrix.columns.astype(int)
            self.user_sim_matrix.index = self.user_sim_matrix.index.astype(int)
            self.user_sim_matrix.columns = self.user_sim_matrix.columns.astype(int)

            self.is_ready = True
            print(f">> [CF] Ready! Users: {len(self.user_item_matrix)}")

        except Exception as e:
            print("❌ CF Load Error:", e)
            self.is_ready = False

    # =====================================================
    # GET SIMILAR USERS (NEW – DÙNG CHO TAB 3)
    # =====================================================
    def get_similar_users(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """
        Trả về danh sách user có gu giống user_id nhất
        Dựa trên cosine similarity (user-user)
        """
        if not self.is_ready:
            return []

        if user_id not in self.user_sim_matrix.index:
            return []

        sim_vector = self.user_sim_matrix.loc[user_id].drop(user_id)

        top_similar = (
            sim_vector
            .sort_values(ascending=False)
            .head(top_k)
        )

        results = []
        for uid, score in top_similar.items():
            results.append({
                "userId": int(uid),
                "similarity": round(float(score), 3)
            })

        return results

    # =====================================================
    # RECOMMEND
    # =====================================================
    def recommend(self, user_id: int, top_k: int = 10, k_neighbors: int = 50) -> pd.DataFrame:
        if not self.is_ready or user_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        sim_vector = self.user_sim_matrix.loc[user_id].drop(user_id)
        neighbors = sim_vector.nlargest(k_neighbors)
        neighbors = neighbors[neighbors > 0]

        if neighbors.empty:
            return pd.DataFrame()

        neighbor_ratings = self.user_item_matrix.loc[neighbors.index]

        # Weighted sum of deviations
        weighted_sum = np.dot(neighbors.values, neighbor_ratings.values)

        rated_mask = (neighbor_ratings != 0).astype(float)
        sum_weights = np.dot(np.abs(neighbors.values), rated_mask.values)
        sum_weights = np.where(sum_weights == 0, 1e-9, sum_weights)

        pred_deviation = weighted_sum / sum_weights
        user_mean = self.user_mean_ratings.loc[user_id, 'mean_rating']
        pred_scores = np.clip(pred_deviation + user_mean, 0.5, 5.0)

        # Remove watched movies
        user_seen = self.user_item_matrix.loc[user_id] != 0
        valid_mask = (~user_seen.values) & (sum_weights > 0.001)

        scores = pd.Series(pred_scores, index=self.user_item_matrix.columns)
        scores = scores[valid_mask].sort_values(ascending=False).head(top_k)

        if scores.empty:
            return pd.DataFrame()

        result = pd.DataFrame({
            'movieId': scores.index,
            'score': scores.values
        })

        result = result.merge(self.movies, on='movieId', how='left')
        return result.reset_index(drop=True)

    # =====================================================
    # PREDICT
    # =====================================================
    def predict(self, user_id: int, movie_id: int, k_neighbors: int = 50) -> float:
        if not self.is_ready:
            return 0.0

        if user_id not in self.user_item_matrix.index:
            return float(self.user_mean_ratings['mean_rating'].mean())

        if movie_id not in self.user_item_matrix.columns:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        sim_vector = self.user_sim_matrix.loc[user_id].drop(user_id)
        neighbors = sim_vector.nlargest(k_neighbors)
        neighbors = neighbors[neighbors > 0]

        neighbor_ratings = self.user_item_matrix.loc[neighbors.index, movie_id]

        mask = neighbor_ratings != 0
        neighbors = neighbors[mask]
        neighbor_ratings = neighbor_ratings[mask]

        if neighbors.empty:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        weighted_sum = np.dot(neighbors.values, neighbor_ratings.values)
        sum_weights = np.abs(neighbors.values).sum()

        deviation = weighted_sum / sum_weights
        user_mean = self.user_mean_ratings.loc[user_id, 'mean_rating']

        return float(np.clip(user_mean + deviation, 0.5, 5.0))

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        return self.recommend(user_id, top_k)
