import pandas as pd
import joblib
import os
import numpy as np

class ContentBasedModel:
    def __init__(self, data_dir: str = None, min_similarity: float = 0.05):
        # Logic tìm đường dẫn (Fallback nếu chạy riêng lẻ)
        if data_dir is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')
        
        self.data_dir = data_dir
        # Common luôn là thư mục anh em của data_dir (evaluation/production)
        self.common_dir = os.path.join(os.path.dirname(data_dir), 'common')

        self.min_similarity = min_similarity
        self.movies = None
        self.cosine_sim = None
        self.id_to_index = None
        self.ratings = None
        self.is_ready = False

        self._load_data()

    def _load_data(self):
        """Load dữ liệu phim và ma trận tương đồng."""
        print(f">> [Content-Based] Loading data from {self.common_dir}...")
        try:
            movies_path = os.path.join(self.common_dir, 'movies_cleaned.csv')
            cosine_path = os.path.join(self.common_dir, 'cosine_similarity_matrix.pkl')

            train_path = os.path.join(self.data_dir, 'train_data.csv')
            full_path = os.path.join(self.data_dir, 'ratings_cleaned.csv')
            ratings_path = train_path if os.path.exists(train_path) else full_path

            if not os.path.exists(movies_path) or not os.path.exists(cosine_path):
                raise FileNotFoundError("Missing data files.")

            self.movies = pd.read_csv(movies_path)
            self.cosine_sim = joblib.load(cosine_path).astype(np.float32)

            if os.path.exists(ratings_path):
                self.ratings = pd.read_csv(ratings_path)
            else:
                print(f"⚠️ Warning: No ratings file found at {ratings_path}.")

            self.id_to_index = pd.Series(self.movies.index, index=self.movies['movieId']).to_dict()

            for col in ['genres', 'tag', 'overview']:
                if col in self.movies.columns:
                    self.movies[col] = self.movies[col].fillna('')

            self.is_ready = True
            print(f">> [Content-Based] Ready! Loaded {len(self.movies)} movies.")

        except Exception as e:
            print(f"ERROR loading CB model: {str(e)}")
            self.is_ready = False

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Dự đoán rating một cách thận trọng.
        - Nếu có dữ liệu tương đồng: Trả về 0.5 -> 5.0
        - Nếu KHÔNG có dữ liệu tương đồng (Similarity=0): Trả về Trung bình của User.
        """
        if not self.is_ready or self.ratings is None or movie_id not in self.id_to_index:
            return 0.0

        user_data = self.ratings[self.ratings['userId'] == user_id]
        if user_data.empty: return 0.0

        # 1. Lấy 50 phim mới nhất (Snapshot sở thích gần đây)
        if 'timestamp' in user_data.columns:
            profile_data = user_data.sort_values(by='timestamp', ascending=False).head(50)
        else:
            profile_data = user_data.tail(50)

        # 2. Lọc phim hợp lệ
        mask = profile_data['movieId'].isin(self.id_to_index.keys())
        valid_profile = profile_data[mask]

        if valid_profile.empty:
            return float(user_data['rating'].mean())

        watched_ids = valid_profile['movieId'].values
        actual_ratings = valid_profile['rating'].values
        valid_indices = [self.id_to_index[m_id] for m_id in watched_ids]

        # 3. Tính toán độ tương đồng
        movie_idx = self.id_to_index[movie_id]
        sim_scores = self.cosine_sim[movie_idx][valid_indices]

        # 4. Lọc nhiễu (ngưỡng 0.05)
        sim_scores = np.where(sim_scores < self.min_similarity, 0, sim_scores)
        sum_sim = np.sum(sim_scores)

        # 5. XỬ LÝ FALLBACK (ĐỒNG BỘ)
        # Nếu tổng độ tương đồng = 0 (Máy không tìm thấy manh mối nào)
        if sum_sim <= 0:
            # Trả về điểm trung bình của chính User đó (Baseline an toàn nhất)
            # Không trả về 0 (vì rating min là 0.5), không trả về 0.5 (vì bị coi là ghét)
            return float(np.mean(actual_ratings))

        # Nếu có dữ liệu: Tính Weighted Average
        pred = np.dot(sim_scores, actual_ratings) / sum_sim

        # Clip kết quả trong khoảng hợp lệ 0.5 - 5.0
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Gợi ý phim: Chỉ gợi ý những phim máy tính 'Hiểu' (Score > 0).
        """
        if not self.is_ready or self.ratings is None:
            return pd.DataFrame()

        user_data = self.ratings[self.ratings['userId'] == user_id]
        if user_data.empty:
            return pd.DataFrame()

        # 1. ĐỒNG BỘ ĐẦU VÀO
        if 'timestamp' in user_data.columns:
            profile_data = user_data.sort_values(by='timestamp', ascending=False).head(50)
        else:
            profile_data = user_data.tail(50)

        mask = profile_data['movieId'].isin(self.id_to_index.keys())
        valid_profile = profile_data[mask]

        if valid_profile.empty:
            return pd.DataFrame()

        watched_indices = [self.id_to_index[m] for m in valid_profile['movieId'].values]
        actual_ratings = valid_profile['rating'].values

        # 2. TÍNH TOÁN MA TRẬN
        sim_subset = self.cosine_sim[watched_indices]
        sim_subset = np.where(sim_subset < self.min_similarity, 0, sim_subset)

        weighted_sum = np.dot(actual_ratings, sim_subset)
        sum_sim = np.sum(sim_subset, axis=0)

        # 3. XỬ LÝ ĐIỂM SỐ (QUAN TRỌNG ⭐)
        # Tạo mask cho những phim có độ tương đồng > 0
        has_sim_mask = sum_sim > 0
        
        # Tránh chia cho 0
        safe_sum_sim = np.where(has_sim_mask, sum_sim, 1.0)
        preds = weighted_sum / safe_sum_sim

        # Logic chuẩn:
        # - Nếu có tương đồng: Clip điểm từ 0.5 đến 5.0
        # - Nếu KHÔNG tương đồng: Giữ nguyên là 0.0 (để lát nữa lọc bỏ)
        preds = np.where(has_sim_mask, np.clip(preds, 0.5, 5.0), 0.0)

        # 4. LỌC BỎ PHIM ĐÃ XEM
        full_watched_ids = user_data['movieId'].values
        full_watched_indices = [self.id_to_index[m] for m in full_watched_ids if m in self.id_to_index]
        preds[full_watched_indices] = -1 # Đánh dấu đã xem

        # 5. LẤY ỨNG VIÊN (MỞ RỘNG BỘ ĐỆM)
        # Lấy gấp 20 lần số lượng cần thiết để tha hồ lọc rác
        n_candidates = top_k * 20 
        top_indices = np.argsort(preds)[-n_candidates:][::-1]

        # 6. CHỌN LỌC CUỐI CÙNG
        # Chỉ lấy những phim có điểm > 0 (Tức là máy có cơ sở để dự đoán)
        valid_indices = [idx for idx in top_indices if preds[idx] > 0]
        
        # Cắt lấy Top K
        valid_indices = valid_indices[:top_k]

        if not valid_indices:
            return pd.DataFrame()

        # 7. TẠO DATAFRAME KẾT QUẢ
        res_df = pd.DataFrame({
            'movieId': self.movies.iloc[valid_indices]['movieId'].values,
            'predicted_rating': preds[valid_indices]
        })

        # Merge thông tin
        final_result = pd.merge(res_df, self.movies, on='movieId', how='left')

        # 8. CHỐT CHẶN CUỐI CÙNG: LỌC RÁC (Last Line of Defense)
        # Đảm bảo không còn phim '(no genres listed)' nào lọt lưới
        if 'genres' in final_result.columns:
            final_result = final_result[
                ~final_result['genres'].str.contains('(no genres listed)', case=False, regex=False)
            ]
            
        # Nếu lọc xong bị thiếu phim thì chấp nhận trả về ít hơn K, còn hơn là trả về rác.
        final_result = final_result.head(top_k)

        # 9. Đổi tên cột cho khớp Frontend
        cols_map = {
            'movieId': 'movieId', 'title': 'title', 'genres': 'genres',
            'tag': 'tags', 'predicted_rating': 'score',
            'rating': 'avg_rating', 'vote_count': 'votes'
        }
        available_cols = [c for c in cols_map.keys() if c in final_result.columns]
        result = final_result[available_cols].rename(columns=cols_map)

        if 'votes' in result.columns:
            result['votes'] = result['votes'].fillna(0).astype(int)

        return result.reset_index(drop=True)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """Đồng bộ Interface cho Evaluation."""
        return self.recommend(user_id, top_k)

# =========================================================
# DRIVER CODE
# =========================================================
if __name__ == "__main__":
    # Khởi tạo Model
    cb_model = ContentBasedModel()

    # --- 2. TEST USER PROFILE ---
    selected_uid = 414

    # --- 3. TEST RECOMMENDATIONS ---
    print(f"\n--- 3. TEST RECOMMENDATIONS for {selected_uid} ---")
    import time
    start_time = time.time()
    recs = cb_model.recommend(selected_uid, top_k=5)
    print(f"⏱️ Time: {time.time() - start_time:.4f}s")

    if not recs.empty:
        print(recs[['title', 'score', 'avg_rating', 'votes']])
    else:
        print("Không có gợi ý nào.")

    # --- 4. TEST EVALUATION FUNCTIONS ---
    print(f"\n--- 4. TEST EVALUATION FUNCTIONS for {selected_uid} ---")
    test_movie_id = 1 # Toy Story
    pred_score = cb_model.predict(selected_uid, test_movie_id)
    print(f"✅ [Predict] Predicted rating for User {selected_uid} - Movie {test_movie_id}: {pred_score:.2f}")

    print(f"✅ [RecommendForUser] Top 5 Evaluation Recs:")
    eval_recs = cb_model.recommend_for_user(selected_uid, top_k=5)
    if not eval_recs.empty:
        print(eval_recs[['movieId', 'title', 'score']].to_string(index=False))