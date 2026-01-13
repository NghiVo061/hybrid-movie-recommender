import pandas as pd
import joblib
import os
import numpy as np
from typing import List, Dict

class CollaborativeModel:
    def __init__(self, data_dir: str = None):
        """
        Backend Model cho User-Based Collaborative Filtering.
        """
        # Logic tìm đường dẫn (Fallback)
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')

        self.data_dir = data_dir
        self.common_dir = os.path.join(os.path.dirname(data_dir), 'common')

        self.movies = None
        self.user_item_matrix = None
        self.user_sim_matrix = None
        self.user_mean_ratings = None
        self.is_ready = False

        self._load_data()

    def _load_data(self):
        """Hàm nội bộ để load dữ liệu."""
        print(f">> [Collaborative-Backend] Loading data from {self.data_dir}...")
        try:
            movies_path = os.path.join(self.common_dir, 'movies_cleaned.csv')
            matrix_path = os.path.join(self.data_dir, 'user_item_matrix_centered.csv')
            sim_path = os.path.join(self.data_dir, 'user_similarity_matrix.pkl')
            mean_path = os.path.join(self.data_dir, 'user_mean_ratings.csv')

            for p in [movies_path, matrix_path, sim_path, mean_path]:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing file: {p}")

            self.movies = pd.read_csv(movies_path)
            self.user_item_matrix = pd.read_csv(matrix_path, index_col=0)
            self.user_sim_matrix = joblib.load(sim_path)
            self.user_mean_ratings = pd.read_csv(mean_path, index_col=0)

            # Ép kiểu dữ liệu chuẩn xác (Int) để tránh lỗi index
            self.user_item_matrix.index = self.user_item_matrix.index.astype(int)
            self.user_item_matrix.columns = self.user_item_matrix.columns.astype(int)
            self.user_sim_matrix.index = self.user_sim_matrix.index.astype(int)
            self.user_sim_matrix.columns = self.user_sim_matrix.columns.astype(int)

            # Xử lý dữ liệu movies (fill na) để tránh lỗi khi merge
            for col in ['genres', 'tag', 'overview']:
                if col in self.movies.columns:
                    self.movies[col] = self.movies[col].fillna('')

            self.is_ready = True
            print(f">> [Collaborative-Backend] Ready! Loaded {len(self.user_sim_matrix)} users.")
        except Exception as e:
            print(f"ERROR loading CF model: {str(e)}")
            self.is_ready = False

    def get_similar_users(self, user_id: int, top_n: int = 5) -> List[Dict]:
        """Trả về danh sách các user có gu giống nhất."""
        if not self.is_ready or user_id not in self.user_sim_matrix.index:
            return []

        sim_scores = self.user_sim_matrix[user_id].drop(user_id)
        top_sim_users = sim_scores.sort_values(ascending=False).head(top_n)

        user_ratings = self.user_item_matrix.loc[user_id]
        user_liked_ids = set(user_ratings[user_ratings > 0].index)

        similar_users_data = []
        for sim_user_id, score in top_sim_users.items():
            sim_ratings = self.user_item_matrix.loc[sim_user_id]
            sim_liked_ids = set(sim_ratings[sim_ratings > 0].index)

            common_ids = list(user_liked_ids & sim_liked_ids)
            common_titles = self.movies[self.movies['movieId'].isin(common_ids)]['title'].head(5).tolist()

            similar_users_data.append({
                'id': int(sim_user_id),
                'similarity_score': float(score),
                'common_movies': common_titles,
                'common_count': len(common_ids)
            })

        return similar_users_data

    def recommend(self, user_id: int, top_k: int = 10, k_neighbors: int = 50) -> pd.DataFrame:
        if not self.is_ready or user_id not in self.user_item_matrix.index:
            return pd.DataFrame()

        sim_vector = self.user_sim_matrix[user_id].drop(user_id)
        top_k_users = sim_vector.nlargest(k_neighbors)
        if top_k_users.empty or top_k_users.max() <= 0:
            return pd.DataFrame()

        top_k_sim_values = top_k_users.values
        top_k_ratings = self.user_item_matrix.loc[top_k_users.index]

        # 1. Tử số: Dot product
        weighted_sum = np.dot(top_k_sim_values, top_k_ratings.values)

        # 2. Mẫu số: Chỉ cộng sim của những người đã đánh giá phim đó
        is_rated_matrix = (top_k_ratings != 0).astype(float)
        # Sử dụng np.abs của sim để đúng công thức Collaborative Filtering
        sum_of_weights = np.dot(np.abs(top_k_sim_values), is_rated_matrix.values)
        sum_of_weights = np.where(sum_of_weights == 0, 1e-9, sum_of_weights)

        # 3. Tính điểm dự đoán
        pred_deviations = weighted_sum / sum_of_weights
        user_mean = self.user_mean_ratings.loc[user_id, 'mean_rating']
        pred_scores = np.clip(pred_deviations + user_mean, 0.5, 5.0)

        # 4. Lọc phim chưa xem & Loại bỏ nhiễu
        user_ratings_current = self.user_item_matrix.loc[user_id]
        is_unrated = (user_ratings_current == 0).values
        # CỦA SỬA TẠI ĐÂY: valid_mask phải cùng size với số lượng phim (8983)
        valid_mask = is_unrated & (sum_of_weights > 0.001)

        all_movie_ids = self.user_item_matrix.columns
        final_scores = pd.Series(pred_scores, index=all_movie_ids)[valid_mask]

        # 5. Lấy Top N
        final_preds = final_scores.sort_values(ascending=False).head(top_k)
        if final_preds.empty: return pd.DataFrame()

        res_df = pd.DataFrame({'movieId': final_preds.index, 'predicted_rating': final_preds.values})
        final_result = pd.merge(res_df, self.movies, on='movieId', how='left')

        cols_map = {
            'movieId': 'movieId', 'title': 'title', 'genres': 'genres',
            'tag': 'tags', 'predicted_rating': 'score', 'rating': 'avg_rating', 'vote_count': 'votes'
        }
        available_cols = [c for c in cols_map.keys() if c in final_result.columns]
        result = final_result[available_cols].rename(columns=cols_map)
        if 'votes' in result.columns:
            result['votes'] = result['votes'].fillna(0).astype(int)

        return result.reset_index(drop=True)

    def predict(self, user_id: int, movie_id: int, k_neighbors: int = 50) -> float:
        """
        Dự đoán rating đồng bộ hoàn toàn với hàm recommend.
        Chiến lược: Lấy K hàng xóm giống User nhất, sau đó mới xem họ chấm phim đó thế nào.
        """
        if not self.is_ready or user_id not in self.user_sim_matrix.index:
            return 0.0

        # 1. Tìm K hàng xóm giống User nhất (Y hệt bước 1 của Recommend)
        sim_vector = self.user_sim_matrix[user_id].drop(user_id)
        top_k_users = sim_vector.nlargest(k_neighbors)

        if top_k_users.empty or top_k_users.max() <= 0:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        # 2. Kiểm tra xem movie_id có trong data không
        if movie_id not in self.user_item_matrix.columns:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        # 3. Lấy rating của K hàng xóm này cho phim movie_id
        # Lưu ý: Ta chỉ tính dựa trên những người TRONG NHÓM K đã xem phim này
        neighbor_ratings = self.user_item_matrix.loc[top_k_users.index, movie_id]

        # Chỉ lấy những người có chấm điểm (khác 0)
        mask = neighbor_ratings != 0
        relevant_sims = top_k_users[mask]
        relevant_ratings = neighbor_ratings[mask]

        if relevant_sims.empty:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        # 4. Tính toán Weighted Sum (Y hệt bước 2 của Recommend)
        # Điểm dự đoán = User_Mean + (Tổng(Sim * Deviation) / Tổng(Sim))
        weighted_sum = np.dot(relevant_sims.values, relevant_ratings.values)
        sum_of_weights = np.abs(relevant_sims.values).sum()

        if sum_of_weights == 0:
            return float(self.user_mean_ratings.loc[user_id, 'mean_rating'])

        predicted_deviation = weighted_sum / sum_of_weights
        user_mean = self.user_mean_ratings.loc[user_id, 'mean_rating']

        return float(np.clip(user_mean + predicted_deviation, 0.5, 5.0))

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Hàm bao đóng (wrapper) của recommend để đồng bộ Interface với CB Model.
        Phục vụ tính Precision/Recall cho Evaluation.
        """
        return self.recommend(user_id, top_k)

# =========================================================
# DRIVER CODE (Mô phỏng Frontend / UI)
# =========================================================
if __name__ == "__main__":
    # Khởi tạo Model
    model = CollaborativeModel()

    selected_uid = 414

    # --- 2. TEST SIMILAR USERS ---
    print(f"\n--- 2. TEST SIMILAR USERS for {selected_uid} ---")
    sims = model.get_similar_users(selected_uid)
    for s in sims:
        print(f"- User {s['id']} (Giống {s['similarity_score']:.2f}) | Chung {s['common_count']} phim: {', '.join(s['common_movies'])}")

    # --- 3. TEST RECOMMENDATIONS ---
    print(f"\n--- 3. TEST RECOMMENDATIONS for {selected_uid} ---")
    recs = model.recommend(selected_uid, top_k=5, k_neighbors=50)

    if not recs.empty:
        # In ra các cột quan trọng (Driver code cũ)
        print(recs[['title', 'score', 'avg_rating', 'votes']])
    else:
        print("Không có gợi ý nào.")

    # --- 4. TEST NEW EVALUATION FUNCTIONS (ADDED) ---
    print(f"\n--- 4. TEST EVALUATION FUNCTIONS (NEW) for {selected_uid} ---")

    # Test Predict (có k_neighbors)
    test_movie_id = 1 # Toy Story
    pred_score = model.predict(selected_uid, test_movie_id, k_neighbors=50)
    print(f"✅ [Predict] Predicted rating (K=50) for User {selected_uid} - Movie {test_movie_id}: {pred_score:.2f}")

    # Test Recommend For User (Wrapper)
    print(f"✅ [RecommendForUser] Top 5 Evaluation Recs:")
    eval_recs = model.recommend_for_user(selected_uid, top_k=5)
    if not eval_recs.empty:
        print(eval_recs[['movieId', 'title', 'score']].to_string(index=False))