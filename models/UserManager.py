import pandas as pd
import os
from collections import Counter
from typing import List, Dict

class UserManager:
    """
    UserManager
    -----------------
    Module quản lý thông tin User (Search, Profile, Persona).
    Tách biệt hoàn toàn khỏi thuật toán gợi ý (CB / CF / Hybrid).

    Mục tiêu:
    - Tránh trùng lặp logic giữa các model
    - Dùng chung cho UI, Evaluation, Hybrid
    - Truy vấn nhanh (cache RAM, không dùng ma trận nặng)
    """

    def __init__(self, data_dir: str = None):
        # Logic tìm đường dẫn (Fallback)
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')

        self.data_dir = data_dir
        self.common_dir = os.path.join(os.path.dirname(data_dir), 'common')

        print(f">> [UserManager] Loading user metadata from {self.data_dir}...")
        try:
            # 1. LOAD MOVIE METADATA
            self.movies = pd.read_csv(
                os.path.join(self.common_dir, 'movies_cleaned.csv')
            )

            self.movie_title_map = dict(zip(self.movies.movieId, self.movies.title))
            self.movie_genre_map = dict(zip(self.movies.movieId, self.movies.genres))

            # 2. LOAD USER RATINGS (NHẸ)
            ratings_path = os.path.join(self.data_dir, 'train_data.csv')
            if not os.path.exists(ratings_path):
                raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

            self.ratings = pd.read_csv(ratings_path)

            # 3. CACHE USER STATISTICS
            # Số lượng tương tác của mỗi user
            self.user_counts = self.ratings['userId'].value_counts().to_dict()

            # Điểm trung bình mỗi user
            self.user_means = (
                self.ratings
                .groupby('userId')['rating']
                .mean()
                .to_dict()
            )

            # Danh sách userId (string) để search nhanh
            self.all_user_ids = [str(uid) for uid in self.user_counts.keys()]

            self.is_ready = True
            print(f">> [UserManager] Ready! Managed {len(self.user_counts)} users.")

        except Exception as e:
            print(f"ERROR loading UserManager: {e}")
            self.is_ready = False

    # USER SEARCH (PHỤC VỤ UI)
    def search_user(self, keyword: str = "", limit: int = 10) -> List[Dict]:
        """
        Tìm kiếm User theo ID (string matching).

        - Không nhập keyword → Trả về Top Active Users
        - Có keyword → Search theo userId
        """
        if not self.is_ready:
            return []

        # CASE 1: TOP ACTIVE USERS
        if not keyword or str(keyword).strip() == "":
            top_users = sorted(
                self.user_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            return [
                {
                    "userId": int(uid),
                    "count": int(cnt),
                    "mean": round(self.user_means.get(uid, 0.0), 1),
                    "type": "top_active"
                }
                for uid, cnt in top_users
            ]

        # CASE 2: SEARCH BY KEYWORD
        keyword = str(keyword).strip()
        results = []

        for uid, cnt in self.user_counts.items():
            if keyword in str(uid):
                results.append({
                    "userId": int(uid),
                    "count": int(cnt),
                    "mean": round(self.user_means.get(uid, 0.0), 1),
                    "type": "search_result"
                })
                if len(results) >= limit:
                    break

        # Ưu tiên user active hơn
        return sorted(results, key=lambda x: x['count'], reverse=True)

    # USER PROFILE (USER PERSONA)
    def get_user_profile(self, user_id: int) -> Dict:
        """
        Lấy thông tin chi tiết User (User Persona).
        Đây là LOGIC CHUẨN DUY NHẤT cho toàn hệ thống.
        """
        if not self.is_ready:
            return {}

        count = self.user_counts.get(user_id, 0)
        if count == 0:
            return {"id": user_id, "error": "User not found"}

        avg_rating = self.user_means.get(user_id, 0.0)

        # 1. LỊCH SỬ XEM
        user_history = self.ratings[self.ratings['userId'] == user_id]

        # 2. TOP MOVIES (USER RATED)
        top_rows = user_history.sort_values(
            by='rating',
            ascending=False
        ).head(5)

        top_movies = []
        for _, row in top_rows.iterrows():
            mid = row['movieId']
            top_movies.append({
                "title": self.movie_title_map.get(mid, f"Movie {mid}"),
                "rating": round(float(row['rating']), 1)
            })

        # 3. TOP GENRES (ADAPTIVE THRESHOLD)
        # Tránh bias user chấm khó / dễ
        threshold = max(4.0, avg_rating)

        high_rated_ids = user_history[
            user_history['rating'] >= threshold
        ]['movieId'].values

        # Fallback nếu user ít phim điểm cao
        if len(high_rated_ids) < 3:
            high_rated_ids = user_history['movieId'].values

        genre_counter = Counter()
        for mid in high_rated_ids:
            genres = self.movie_genre_map.get(mid, "").split('|')
            genre_counter.update(
                g for g in genres if g not in ['', '(no genres listed)']
            )

        top_genres = [g[0] for g in genre_counter.most_common(3)]

        # 4. INTERACTION LEVEL (HEURISTIC)
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
            "top_genres": top_genres,
            "top_movies": top_movies,
            "interaction_level": level
        }

# =====================================================
# DRIVER TEST (OPTIONAL)
# =====================================================
if __name__ == "__main__":
    um = UserManager()

    print("\n📍 Top Active Users:")
    for u in um.search_user(limit=5):
        print(u)

    print("\n📍 Search '41':")
    for u in um.search_user("41", limit=5):
        print(u)

    test_user = list(um.user_counts.keys())[0]
    print(f"\n📍 Profile User {test_user}:")
    profile = um.get_user_profile(test_user)
    for k, v in profile.items():
        print(f"{k}: {v}")
