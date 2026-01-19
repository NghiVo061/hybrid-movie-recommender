import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from models.ContentBased import ContentBasedModel
from models.CollaborativeFiltering import CollaborativeModel
from models.UserManager import UserManager

class AdaptiveHybridModel:
    def __init__(self, data_dir: str = None):
        """
        Mô hình Lai ghép Thích nghi (Adaptive Weighted Hybrid).
        """
        print(">> [Hybrid] Initializing Adaptive Hybrid System...")

        # 1. Tìm vị trí file Hybrid.py
        current_file = os.path.abspath(__file__) 
        models_dir = os.path.dirname(current_file) 
        project_root = os.path.dirname(models_dir) 
        
        # 2. Thiết lập đường dẫn data tuyệt đối
        if data_dir is None:
            # Mặc định trỏ vào evaluation
            self.data_dir = os.path.join(project_root, 'data', 'processed', 'evaluation')
        else:
            # Nếu người dùng truyền vào, đảm bảo nối từ root nếu là đường dẫn tương đối
            if not os.path.isabs(data_dir):
                self.data_dir = os.path.join(project_root, data_dir)
            else:
                self.data_dir = data_dir

        print(f">> [System] Data Path: {self.data_dir}")
        # ----------------------------------

        # 3. Truyền đường dẫn tuyệt đối này cho các con
        self.cb_model = ContentBasedModel(data_dir=self.data_dir)
        self.cf_model = CollaborativeModel(data_dir=self.data_dir)
        self.user_manager = UserManager(data_dir=self.data_dir)

        self.movies = self.cb_model.movies
        self.is_ready = (
            self.cb_model.is_ready and
            self.cf_model.is_ready and
            self.user_manager.is_ready
        )

        self.max_interact_count = 1
        if self.is_ready and self.user_manager.user_counts:
            real_max = max(self.user_manager.user_counts.values())
            self.max_interact_count = min(real_max, 300)
            print(f">> [Hybrid] Ready! Max Interactions (Capped): {self.max_interact_count}")
        else:
            print(">> [Hybrid] ERROR: Sub-models failed.")

    def calculate_adaptive_weight(self, user_id: int) -> float:
        """
        Tính trọng số Alpha theo LOGARITHMIC SCALING.
        """
        user_count = self.user_manager.user_counts.get(user_id, 0)

        if user_count <= 0:
            return 0.0

        # Công thức Logarithmic
        # Alpha tăng nhanh lúc đầu, chậm dần về sau
        alpha = np.log(1 + user_count) / np.log(1 + self.max_interact_count)

        return float(np.clip(alpha, 0.0, 0.95))

    def get_popular_recommendations(self, top_k=10):
        """
        Fallback: Trả về phim phổ biến nhất nếu User mới tinh.
        (GIỮ NGUYÊN KHÔNG ĐỔI)
        """
        if self.movies is None or self.movies.empty:
            return pd.DataFrame()

        # Sắp xếp theo lượng vote (độ phổ biến)
        if 'vote_count' in self.movies.columns:
            # Lấy phim có vote cao và điểm ổn > 3.0
            pop_movies = self.movies[self.movies['rating'] > 3.0].sort_values(by='vote_count', ascending=False).head(top_k)
        else:
            pop_movies = self.movies.head(top_k)

        pop_movies['score'] = pop_movies['rating'] 
        cols_map = {'tag': 'tags', 'rating': 'avg_rating', 'vote_count': 'votes'}
        return pop_movies.rename(columns=cols_map).reset_index(drop=True)

    def recommend(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Hàm gợi ý Hybrid: Đã sửa lại logic để lấy điểm thật (Predict) 
        thay vì điền 0, và đảm bảo trả về Title/Genres.
        """
        if not self.is_ready: return pd.DataFrame()

        # --- 0. XỬ LÝ COLD START (User mới tinh) ---
        if user_id not in self.user_manager.user_counts:
            return self.get_popular_recommendations(top_k)

        # --- 1. TÍNH ALPHA ---
        alpha = self.calculate_adaptive_weight(user_id)
        
        # --- 2. LẤY ỨNG VIÊN (Candidate Generation) ---
        # Lấy danh sách ứng viên rộng (gấp 5 lần cần thiết để merge)
        candidate_k = top_k * 5
        
        df_cb = self.cb_model.recommend(user_id, top_k=candidate_k)
        df_cf = self.cf_model.recommend(user_id, top_k=candidate_k, k_neighbors=50)

        # Nếu cả 2 đều rỗng -> Trả về rỗng
        if df_cb.empty and df_cf.empty: return pd.DataFrame()

        cb_scores = df_cb[['movieId', 'score']].rename(columns={'score': 'score_cb'}) if not df_cb.empty else pd.DataFrame(columns=['movieId', 'score_cb'])
        cf_scores = df_cf[['movieId', 'score']].rename(columns={'score': 'score_cf'}) if not df_cf.empty else pd.DataFrame(columns=['movieId', 'score_cf'])

        # --- 3. MERGE OUTER (Gộp danh sách) ---
        # Giữ lại NaN để biết giá trị nào bị thiếu
        merged = pd.merge(cb_scores, cf_scores, on='movieId', how='outer')

        # --- 4. DỰ ĐOÁN BÙ (FILL MISSING SCORES) ---
        def fill_missing_scores(row):
            mid = int(row['movieId'])
            s_cb = row['score_cb']
            s_cf = row['score_cf']
            
            # Nếu thiếu điểm CB -> Gọi CB Model dự đoán
            if pd.isna(s_cb):
                try: s_cb = self.cb_model.predict(user_id, mid)
                except: s_cb = 0.0
            
            # Nếu thiếu điểm CF -> Gọi CF Model dự đoán
            if pd.isna(s_cf):
                try: s_cf = self.cf_model.predict(user_id, mid, k_neighbors=50)
                except: s_cf = 0.0
            
            return pd.Series([s_cb, s_cf])

        # Áp dụng hàm điền khuyết
        merged[['score_cb', 'score_cf']] = merged.apply(fill_missing_scores, axis=1)

        # --- 5. TÍNH FINAL SCORE ---
        # Lúc này cả 2 cột đều đã có số thật, áp dụng công thức trọng số
        merged['final_score'] = (alpha * merged['score_cf']) + ((1 - alpha) * merged['score_cb'])

        # --- 6. SẮP XẾP & LẤY TOP K ---
        merged = merged.sort_values(by='final_score', ascending=False).head(top_k)

        # --- 7. GẮN METADATA (Title, Genres...) ---
        final_ids = merged['movieId'].values
        
        # Lấy thông tin gốc từ self.movies
        meta_info = self.movies[self.movies['movieId'].isin(final_ids)].copy()
        
        # Merge lại để lấy thông tin phim
        final_result = pd.merge(
            merged[['movieId', 'final_score', 'score_cb', 'score_cf']], 
            meta_info, 
            on='movieId', 
            how='left'
        )

        # --- 8. FORMAT & OUTPUT ---
        cols_map = {
            'final_score': 'score',
            'rating': 'avg_rating', 
            'vote_count': 'votes',
            'tag': 'tags'
        }
        final_result = final_result.rename(columns=cols_map)

        # Xử lý số liệu Votes
        if 'votes' in final_result.columns:
            final_result['votes'] = final_result['votes'].fillna(0).astype(int)

        desired_order = [
            'movieId', 'title', 'genres',       # Thông tin cơ bản
            'score', 'score_cb', 'score_cf',    # Điểm số
            'avg_rating', 'votes', 'tags'       # Thông tin bổ trợ
        ]
        
        # Chỉ lấy những cột thực sự tồn tại trong kết quả
        final_cols = [c for c in desired_order if c in final_result.columns]

        return final_result[final_cols].reset_index(drop=True)

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        return self.recommend(user_id, top_k)

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Dự đoán điểm số.
        """
        if not self.is_ready: return 0.0

        # Check tồn tại để tránh lỗi
        if user_id not in self.user_manager.user_counts:
             return 0.0 # Hoặc global mean

        alpha = self.calculate_adaptive_weight(user_id)
        cf_pred = self.cf_model.predict(user_id, movie_id, k_neighbors=50)
        cb_pred = self.cb_model.predict(user_id, movie_id)

        final_pred = (alpha * cf_pred) + ((1 - alpha) * cb_pred)
        return float(np.clip(final_pred, 0.5, 5.0))

    # --- CÁC HÀM UI HELPER ---
    def search_user(self, keyword, limit=10):
        return self.user_manager.search_user(keyword, limit)

    def get_user_profile(self, user_id):
        return self.user_manager.get_user_profile(user_id)

# =========================================================
# DRIVER CODE (TEST TOÀN DIỆN HỆ THỐNG HYBRID)
# =========================================================
if __name__ == "__main__":
    # 1. KHỞI TẠO MÔ HÌNH
    print("\n" + "="*60)
    print("🚀 KHỞI TẠO HỆ THỐNG ADAPTIVE HYBRID (OPTIMIZED V1)...")
    print("="*60)
    hybrid = AdaptiveHybridModel()
    
    if not hybrid.is_ready:
        print("❌ Lỗi: Hệ thống chưa sẵn sàng. Kiểm tra lại đường dẫn dữ liệu.")
        sys.exit()

    # =========================================================
    # KỊCH BẢN 1: KIỂM TRA UI & USER MANAGER (DELEGATION)
    # =========================================================
    print("\n" + "-"*60)
    print("Scenario 1: KIỂM TRA UI HELPER (SEARCH & PROFILE)")
    print("-" * 60)
    
    # A. Test Search (Top Active)
    print("🔎 1.1. Top Active Users (Không nhập keyword):")
    top_users = hybrid.search_user("", limit=3)
    for u in top_users:
        print(f"   - User {u['userId']} | Ratings: {u['count']} | Mean: {u['mean']} ⭐")
        
    # B. Test Search (Keyword cụ thể)
    keyword = "41"
    print(f"\n🔎 1.2. Tìm kiếm User có ID chứa '{keyword}':")
    search_res = hybrid.search_user(keyword, limit=3)
    for u in search_res:
        print(f"   - User {u['userId']} | Ratings: {u['count']}")

    # Lấy ra 1 User để test (Ưu tiên ID 414 nếu tìm thấy)
    target_user_id = 414
    if not any(u['userId'] == target_user_id for u in search_res):
        target_user_id = search_res[0]['userId'] if search_res else 1

    # C. Test Profile
    print(f"\n👤 1.3. Lấy Profile chi tiết cho User {target_user_id}:")
    profile = hybrid.get_user_profile(target_user_id)
    if "error" not in profile:
        print(f"   - ID: {profile['id']}")
        print(f"   - Level: {profile.get('interaction_level', 'N/A')}")
        print(f"   - Top Genres: {', '.join(profile['top_genres'])}")
        print(f"   - Top Movies: {[m['title'] for m in profile['top_movies']]}")
    else:
        print(f"   - Error: {profile['error']}")

    # =========================================================
    # KỊCH BẢN 2: KIỂM TRA LOGIC TRỌNG SỐ THÍCH NGHI (ALPHA)
    # =========================================================
    print("\n" + "-"*60)
    print("Scenario 2: KIỂM TRA ADAPTIVE WEIGHTING (ALPHA)")
    print("-" * 60)
    
    # Tìm một user ít tương tác (Newbie) để so sánh
    # Lấy min count từ UserManager
    all_counts = hybrid.user_manager.user_counts
    low_interact_user = min(all_counts, key=all_counts.get) if all_counts else target_user_id
    
    users_to_test = [target_user_id, low_interact_user]
    
    print(f"{'User ID':<10} | {'Ratings':<10} | {'Alpha':<10} | {'Chiến thuật'}")
    print("-" * 60)
    
    for uid in users_to_test:
        count = all_counts.get(uid, 0)
        alpha = hybrid.calculate_adaptive_weight(uid)
        strategy = "Tin vào CF (Cộng đồng)" if alpha > 0.5 else "Tin vào CB (Nội dung)"
        print(f"{uid:<10} | {count:<10} | {alpha:.4f}     | {strategy}")

    # =========================================================
    # KỊCH BẢN 3: GỢI Ý & PHÂN TÍCH ĐIỂM SỐ (RECOMMENDATION)
    # =========================================================
    print("\n" + "-"*60)
    print(f"Scenario 3: CHẠY GỢI Ý CHO USER {target_user_id}")
    print("-" * 60)
    
    recs = hybrid.recommend(target_user_id, top_k=5)
    
    if not recs.empty:
        # Hiển thị bảng chi tiết để debug xem điểm số đến từ đâu
        # score_cb: Điểm nội dung, score_cf: Điểm cộng đồng, score: Điểm tổng hợp
        cols = ['title', 'score', 'score_cb', 'score_cf', 'avg_rating']
        # Lọc cột nếu tồn tại để tránh lỗi print
        print_cols = [c for c in cols if c in recs.columns]
        print(recs[print_cols].to_string(index=False))
    else:
        print("⚠️ Không tìm thấy gợi ý nào.")

    # =========================================================
    # KỊCH BẢN 4: COLD-START (USER KHÔNG TỒN TẠI)
    # =========================================================
    print("\n" + "-"*60)
    print("Scenario 4: TEST FALLBACK (UNKNOWN USER)")
    print("-" * 60)
    
    unknown_id = 999999999
    print(f"❓ Đang gợi ý cho User ảo {unknown_id}...")
    
    # Hệ thống sẽ trả về phim phổ biến (Popular)
    pop_recs = hybrid.recommend(unknown_id, top_k=3)
    
    if not pop_recs.empty:
        print("✅ Hệ thống chuyển sang chế độ Popular Recommendation:")
        print(pop_recs[['title', 'avg_rating', 'votes']].to_string(index=False))
    else:
        print("❌ Lỗi: Không có fallback data.")

    # =========================================================
    # KỊCH BẢN 5: DỰ ĐOÁN ĐIỂM SỐ (PREDICT)
    # =========================================================
    print("\n" + "-"*60)
    print("Scenario 5: DỰ ĐOÁN RATING CỤ THỂ")
    print("-" * 60)
    
    test_movie_id = 1 # Toy Story
    pred_score = hybrid.predict(target_user_id, test_movie_id)
    
    print(f"🎥 User {target_user_id} dự kiến chấm phim 'Toy Story' (ID 1):")
    print(f"   => {pred_score:.2f} / 5.0 ⭐")

    print("\n" + "="*60)
    print("✅ TEST COMPLETED.")
    print("="*60)