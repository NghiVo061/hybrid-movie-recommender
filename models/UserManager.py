import pandas as pd
import os
from collections import Counter
from typing import List, Dict


class UserManager:
    """
    UserManager
    -----------------
    Quản lý:
    - Search User
    - User Profile
    - User Persona
    Sử dụng DUY NHẤT dữ liệu PRODUCTION (ratings_cleaned.csv)
    """

    def __init__(self, data_dir: str = None):
        try:
            # =====================================================
            # PATH SETUP
            # =====================================================
            if data_dir is None:
                current_file = os.path.abspath(__file__)
                project_root = os.path.dirname(os.path.dirname(current_file))
                data_dir = os.path.join(
                    project_root, 'data', 'processed', 'production'
                )

            self.data_dir = data_dir
            self.common_dir = os.path.join(
                os.path.dirname(self.data_dir), 'common'
            )

            print(f">> [UserManager] Loading data from {self.data_dir}")

            # =====================================================
            # LOAD MOVIES METADATA
            # =====================================================
            movies_path = os.path.join(
                self.common_dir, 'movies_cleaned.csv'
            )
            if not os.path.exists(movies_path):
                raise FileNotFoundError("movies_cleaned.csv NOT FOUND")

            self.movies = pd.read_csv(movies_path)

            self.movie_title_map = dict(
                zip(self.movies['movieId'], self.movies['title'])
            )
            self.movie_genre_map = dict(
                zip(self.movies['movieId'], self.movies['genres'])
            )

            # =====================================================
            # LOAD RATINGS (PRODUCTION ONLY)
            # =====================================================
            ratings_path = os.path.join(
                self.data_dir, 'ratings_cleaned.csv'
            )
            if not os.path.exists(ratings_path):
                raise FileNotFoundError("ratings_cleaned.csv NOT FOUND")

            self.ratings = pd.read_csv(ratings_path)

            self.ratings['userId'] = self.ratings['userId'].astype(int)
            self.ratings['movieId'] = self.ratings['movieId'].astype(int)
            self.ratings['rating'] = self.ratings['rating'].astype(float)

            # =====================================================
            # USER STATISTICS
            # =====================================================
            self.user_counts = (
                self.ratings
                .groupby('userId')
                .size()
                .to_dict()
            )

            self.user_means = (
                self.ratings
                .groupby('userId')['rating']
                .mean()
                .to_dict()
            )

            self.all_user_ids = sorted(self.user_counts.keys())

            self.is_ready = True
            print(f">> [UserManager] Ready! {len(self.user_counts)} users loaded.")

        except Exception as e:
            print(f">> [UserManager ERROR] {e}")
            self.is_ready = False

    # =====================================================
    # SEARCH USER (FIXED – UI FRIENDLY)
    # =====================================================
    def search_user(self, keyword: str = "", limit: int = 10) -> List[Dict]:
        if not self.is_ready:
            return []

        results = []

        # ===============================
        # CASE 1: NO KEYWORD
        # → SHOW USER 1,2,3,... (NOT TOP ACTIVE)
        # ===============================
        if not keyword or str(keyword).strip() == "":
            for uid in self.all_user_ids[:limit]:
                results.append({
                    "userId": uid,
                    "count": self.user_counts.get(uid, 0),
                    "mean": round(self.user_means.get(uid, 0.0), 2),
                    "type": "default"
                })
            return results

        # ===============================
        # CASE 2: SEARCH BY USER ID
        # ===============================
        keyword = str(keyword).strip()

        exact_match = []
        partial_match = []

        for uid in self.all_user_ids:
            uid_str = str(uid)

            if uid_str == keyword:
                exact_match.append({
                    "userId": uid,
                    "count": self.user_counts.get(uid, 0),
                    "mean": round(self.user_means.get(uid, 0.0), 2),
                    "type": "exact"
                })
            elif keyword in uid_str:
                partial_match.append({
                    "userId": uid,
                    "count": self.user_counts.get(uid, 0),
                    "mean": round(self.user_means.get(uid, 0.0), 2),
                    "type": "partial"
                })

        # ƯU TIÊN MATCH CHÍNH XÁC
        results = exact_match + partial_match

        return results[:limit]

    # =====================================================
    # USER PROFILE / PERSONA
    # =====================================================
    def get_user_profile(self, user_id: int) -> Dict:
        if not self.is_ready:
            return {"error": "UserManager not ready"}

        if user_id not in self.user_counts:
            return {"id": user_id, "error": "User not found"}

        history = self.ratings[self.ratings['userId'] == user_id]
        count = self.user_counts[user_id]
        avg_rating = self.user_means[user_id]

        # ---------- TOP MOVIES ----------
        top_movies = []
        top_rows = history.sort_values(
            by='rating',
            ascending=False
        ).head(5)

        for _, row in top_rows.iterrows():
            mid = row['movieId']
            top_movies.append({
                "title": self.movie_title_map.get(mid, f"Movie {mid}"),
                "rating": round(float(row['rating']), 1)
            })

        # ---------- TOP GENRES ----------
        threshold = max(4.0, avg_rating)
        high_rated = history[history['rating'] >= threshold]['movieId']

        if len(high_rated) < 3:
            high_rated = history['movieId']

        genre_counter = Counter()
        for mid in high_rated:
            genres = self.movie_genre_map.get(mid, "")
            genre_counter.update(
                g for g in genres.split('|')
                if g and g != '(no genres listed)'
            )

        # ---------- INTERACTION LEVEL ----------
        if count >= 50:
            level = "High"
        elif count >= 20:
            level = "Medium"
        else:
            level = "Low"

        return {
            "id": int(user_id),
            "total_watched": int(count),
            "avg_rating": round(float(avg_rating), 2),
            "top_genres": [g for g, _ in genre_counter.most_common(3)],
            "top_movies": top_movies,
            "interaction_level": level
        }
