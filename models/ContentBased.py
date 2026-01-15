import pandas as pd
import joblib
import os
import numpy as np


class ContentBasedModel:
    def __init__(self, data_dir: str = None, min_similarity: float = 0.05):
        """
        data_dir: trỏ tới data/processed/evaluation
        """
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')

        self.data_dir = data_dir

        # common/ là anh em của evaluation/
        self.common_dir = os.path.join(os.path.dirname(data_dir), 'common')

        self.min_similarity = min_similarity

        self.movies = None
        self.cosine_sim = None
        self.tfidf_matrix = None
        self.id_to_index = None
        self.ratings = None

        self.is_ready = False

        self._load_data()

    # =========================================================
    # LOAD DATA
    # =========================================================
    def _load_data(self):
        print(f">> [Content-Based] Loading data...")
        print(f"   • common_dir: {self.common_dir}")
        print(f"   • data_dir  : {self.data_dir}")

        try:
            # -------- COMMON FILES --------
            movies_path = os.path.join(self.common_dir, 'movies_cleaned.csv')
            cosine_path = os.path.join(self.common_dir, 'cosine_similarity_matrix.pkl')
            tfidf_path = os.path.join(self.common_dir, 'tfidf_matrix.pkl')

            if not os.path.exists(movies_path):
                raise FileNotFoundError("movies_cleaned.csv not found")

            if not os.path.exists(cosine_path):
                raise FileNotFoundError("cosine_similarity_matrix.pkl not found")

            if not os.path.exists(tfidf_path):
                raise FileNotFoundError("tfidf_matrix.pkl not found")

            # -------- LOAD COMMON --------
            self.movies = pd.read_csv(movies_path)
            self.cosine_sim = joblib.load(cosine_path).astype(np.float32)
            self.tfidf_matrix = joblib.load(tfidf_path)

            # -------- RATINGS (FULL DATA FOR UI) --------
            ratings_path = os.path.join(
                os.path.dirname(self.data_dir),  # processed/
                'production',
                'ratings_cleaned.csv'
            )

            if not os.path.exists(ratings_path):
                raise FileNotFoundError("ratings_cleaned.csv not found in production/")

            self.ratings = pd.read_csv(ratings_path)

            # -------- ID → INDEX --------
            self.id_to_index = pd.Series(
                self.movies.index,
                index=self.movies['movieId']
            ).to_dict()

            # -------- CLEAN TEXT COLS --------
            for col in ['genres', 'tag', 'overview']:
                if col in self.movies.columns:
                    self.movies[col] = self.movies[col].fillna('')

            self.is_ready = True
            print(f">> [Content-Based] READY ✔")
            print(f"   • Movies : {len(self.movies)}")
            print(f"   • Ratings: {len(self.ratings)}")

        except Exception as e:
            print("❌ [Content-Based] FAILED")
            print("→ ERROR:", repr(e))
            self.is_ready = False

    # =========================================================
    # PREDICT
    # =========================================================
    def predict(self, user_id: int, movie_id: int) -> float:
        if not self.is_ready:
            return 0.0

        if movie_id not in self.id_to_index:
            return 0.0

        user_data = self.ratings[self.ratings['userId'] == user_id]
        if user_data.empty:
            return 0.0

        # lấy 50 phim gần nhất
        profile = user_data.sort_values(
            by='timestamp', ascending=False
        ).head(50)

        valid = profile['movieId'].isin(self.id_to_index.keys())
        profile = profile[valid]

        if profile.empty:
            return float(user_data['rating'].mean())

        indices = [self.id_to_index[m] for m in profile['movieId']]
        ratings = profile['rating'].values

        movie_idx = self.id_to_index[movie_id]
        sims = self.cosine_sim[movie_idx][indices]
        sims = np.where(sims < self.min_similarity, 0, sims)

        if sims.sum() == 0:
            return float(np.mean(ratings))

        pred = np.dot(sims, ratings) / np.sum(sims)
        return float(np.clip(pred, 0.5, 5.0))

    # =========================================================
    # RECOMMEND
    # =========================================================
    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        if not self.is_ready:
            return pd.DataFrame()

        user_data = self.ratings[self.ratings['userId'] == user_id]
        if user_data.empty:
            return pd.DataFrame()

        profile = user_data.sort_values(
            by='timestamp', ascending=False
        ).head(50)

        valid = profile['movieId'].isin(self.id_to_index.keys())
        profile = profile[valid]

        if profile.empty:
            return pd.DataFrame()

        watched_idx = [self.id_to_index[m] for m in profile['movieId']]
        ratings = profile['rating'].values

        sim_subset = self.cosine_sim[watched_idx]
        sim_subset = np.where(sim_subset < self.min_similarity, 0, sim_subset)

        weighted_sum = np.dot(ratings, sim_subset)
        sim_sum = np.sum(sim_subset, axis=0)
        sim_sum = np.where(sim_sum == 0, 1, sim_sum)

        preds = weighted_sum / sim_sum
        preds = np.clip(preds, 0.5, 5.0)

        # loại phim đã xem
        full_watched = user_data['movieId'].values
        watched_all_idx = [
            self.id_to_index[m]
            for m in full_watched
            if m in self.id_to_index
        ]
        preds[watched_all_idx] = -1

        top_idx = np.argsort(preds)[-top_k:][::-1]
        valid_idx = [i for i in top_idx if preds[i] >= 3.0]

        if not valid_idx:
            valid_idx = top_idx[:top_k]

        result = pd.DataFrame({
            'movieId': self.movies.iloc[valid_idx]['movieId'].values,
            'score': preds[valid_idx]
        })

        result = result.merge(self.movies, on='movieId', how='left')

        return result.reset_index(drop=True)

    # =========================================================
    # EVALUATION COMPAT
    # =========================================================
    def recommend_for_user(self, user_id: int, top_k: int = 10):
        return self.recommend(user_id, top_k)


# =========================================================
# TEST STANDALONE
# =========================================================
if __name__ == "__main__":
    cb = ContentBasedModel()
    print("Ready:", cb.is_ready)

    if cb.is_ready:
        uid = 414
        recs = cb.recommend(uid, 5)
        print(recs[['title', 'score']].head())
