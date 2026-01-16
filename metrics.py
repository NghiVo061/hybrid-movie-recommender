import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import time # Dùng để đo thời gian chạy

# --- FIX LỖI MODULE NOT FOUND ---
# Lấy đường dẫn thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy đường dẫn thư mục models
models_dir = os.path.join(current_dir, 'models')

# Thêm folder 'models' vào hệ thống tìm kiếm của Python
if models_dir not in sys.path:
    sys.path.append(models_dir)
# --------------------------------

# Import Model
try:
    from Hybrid import AdaptiveHybridModel
except ImportError:
    # Fallback đề phòng trường hợp file tên là hybrid.py (thường)
    from hybrid import AdaptiveHybridModel

# Thư mục lưu biểu đồ
OUTPUT_DIR = 'static/evaluation_charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_evaluation():
    print("--- BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ (FULL DATASET) ---")
    start_time_total = time.time()
    
    # 1. KHỞI TẠO MODEL
    data_path = os.path.join(current_dir, 'data', 'processed', 'evaluation')
    print(f">> Data path: {data_path}")
    
    hybrid = AdaptiveHybridModel(data_dir=data_path)
    
    if not hybrid.is_ready:
        print("❌ Lỗi: Hệ thống chưa sẵn sàng. Kiểm tra lại dữ liệu.")
        return

    # 2. LOAD DỮ LIỆU TEST
    test_file = os.path.join(data_path, 'test_data.csv')
    if not os.path.exists(test_file):
        print(f"❌ Không tìm thấy file: {test_file}")
        # Tạo dummy data nếu không thấy file (để tránh crash)
        test_data = pd.DataFrame({
            'userId': [414, 414, 414, 1, 1],
            'movieId': [1, 2, 3, 1, 5],
            'rating': [5.0, 4.0, 3.0, 4.0, 2.0]
        })
    else:
        test_data = pd.read_csv(test_file)
        # Chuẩn hóa tên cột
        test_data.rename(columns={'user_id': 'userId', 'movie_id': 'movieId'}, inplace=True)
    
    print(f"✅ Đã load {len(test_data)} dòng dữ liệu kiểm thử.")

    # ==========================================
    # NHIỆM VỤ 1: DỰ ĐOÁN LỖI (ERROR METRICS - RMSE & MAE)
    # ==========================================
    print("\n[Task 1] Đang tính RMSE và MAE (Trên toàn bộ dữ liệu Test)...")
    
    predictions = {'CB': [], 'CF': [], 'Hybrid': [], 'Actual': []}

    # Chạy loop qua toàn bộ test_data
    for idx, row in test_data.iterrows():
        try:
            # Map tên cột
            u_val = row.get('userId') if 'userId' in row else row.get('user_id')
            m_val = row.get('movieId') if 'movieId' in row else row.get('movie_id')
            
            u = int(u_val)
            m = int(m_val)
            r_true = float(row['rating'])
            
            p_cb = hybrid.cb_model.predict(u, m)
            p_cf = hybrid.cf_model.predict(u, m, k_neighbors=50) 
            p_hybrid = hybrid.predict(u, m) 
            
            predictions['CB'].append(p_cb)
            predictions['CF'].append(p_cf)
            predictions['Hybrid'].append(p_hybrid)
            predictions['Actual'].append(r_true)
        except Exception as e:
            continue 

    results_error = {}
    models_list = ['CB', 'CF', 'Hybrid']
    
    for model in models_list:
        if len(predictions[model]) == 0:
            print(f"⚠️ Model {model} không trả về dự đoán nào.")
            rmse, mae = 0, 0
        else:
            rmse = np.sqrt(mean_squared_error(predictions['Actual'], predictions[model]))
            mae = mean_absolute_error(predictions['Actual'], predictions[model])
        
        results_error[model] = {'RMSE': rmse, 'MAE': mae}
        print(f"   👉 Model {model}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # ==========================================
    # NHIỆM VỤ 2: ĐÁNH GIÁ XẾP HẠNG (PRECISION & RECALL)
    # ==========================================
    print("\n[Task 2] Đang tính Precision@10 và Recall@10...")
    
    k = 10
    precisions = {'CB': [], 'CF': [], 'Hybrid': []}
    recalls = {'CB': [], 'CF': [], 'Hybrid': []}
    
    relevant_data = test_data[test_data['rating'] >= 4.0]
    uid_col = 'userId' if 'userId' in relevant_data.columns else 'user_id'
    mid_col = 'movieId' if 'movieId' in relevant_data.columns else 'movie_id'
    
    # Gom nhóm dữ liệu
    test_user_movies = relevant_data.groupby(uid_col)[mid_col].apply(list).to_dict()
    
    # --- THAY ĐỔI: LẤY TẤT CẢ USER (FULL DATA) ---
    users_to_test = list(test_user_movies.keys())
    total_users_test = len(users_to_test)
    print(f"⏳ Đang chạy ranking cho {total_users_test} users (Vui lòng đợi)...")
    
    for i, u in enumerate(users_to_test):
        # In tiến độ mỗi 50 user để biết code không bị treo
        if (i + 1) % 50 == 0:
            print(f"   ... Đang xử lý User thứ {i + 1}/{total_users_test}")

        ground_truth = set(test_user_movies[u])
        if len(ground_truth) == 0: continue

        try:
            # Lấy danh sách gợi ý từ các model
            recs_cb = hybrid.cb_model.recommend_for_user(u, top_k=k)
            recs_cf = hybrid.cf_model.recommend_for_user(u, top_k=k)
            recs_hybrid = hybrid.recommend_for_user(u, top_k=k)
        except Exception:
            continue
        
        # Hàm tính Precision và Recall
        def calculate_metrics(recs_df, truth_set):
            if recs_df is None or recs_df.empty: return 0.0, 0.0
            col_id = 'movieId' if 'movieId' in recs_df.columns else 'movie_id'
            if col_id not in recs_df.columns: return 0.0, 0.0
            
            rec_ids = set(recs_df[col_id].values)
            hits = len(rec_ids & truth_set)
            prec = hits / k
            rec = hits / len(truth_set) if len(truth_set) > 0 else 0
            return prec, rec

        p, r = calculate_metrics(recs_cb, ground_truth)
        precisions['CB'].append(p); recalls['CB'].append(r)
        
        p, r = calculate_metrics(recs_cf, ground_truth)
        precisions['CF'].append(p); recalls['CF'].append(r)
        
        p, r = calculate_metrics(recs_hybrid, ground_truth)
        precisions['Hybrid'].append(p); recalls['Hybrid'].append(r)

    results_ranking = {}
    for model in models_list:
        avg_p = np.mean(precisions[model]) if precisions[model] else 0
        avg_r = np.mean(recalls[model]) if recalls[model] else 0
        results_ranking[model] = {'Precision@10': avg_p, 'Recall@10': avg_r}
        print(f"   👉 Model {model}: Precision@10 = {avg_p:.4f}, Recall@10 = {avg_r:.4f}")

    # ==========================================
    # NHIỆM VỤ 3: PHÂN TÍCH TÍNH THÍCH NGHI (ALPHA)
    # ==========================================
    print("\n[Task 3] Đang phân tích Alpha (Toàn bộ User trong hệ thống)...")
    user_interactions = []
    alpha_values = []
    
    # --- THAY ĐỔI: LẤY TẤT CẢ USER ---
    all_users = list(hybrid.user_manager.user_counts.keys())
    print(f"⏳ Đang tính Alpha cho {len(all_users)} users...")
    
    for u in all_users:
        count = hybrid.user_manager.user_counts.get(u, 0)
        alpha = hybrid.calculate_adaptive_weight(u)
        user_interactions.append(count)
        alpha_values.append(alpha)

    # ==========================================
    # NHIỆM VỤ 4: VẼ VÀ LƯU BIỂU ĐỒ (VISUALIZATION)
    # ==========================================
    print("\n[Task 4] Đang vẽ và lưu biểu đồ...")
    sns.set_style("whitegrid")

    def plot_comparison(metric_name, values, title, filename, color_palette, higher_is_better=True):
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x=models_list, y=values, palette=color_palette, hue=models_list, legend=False)
        
        direction = "Cao hơn là tốt hơn" if higher_is_better else "Thấp hơn là tốt hơn"
        plt.title(f'{title} ({direction})')
        plt.ylabel(metric_name)
        
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        
        save_path = f'{OUTPUT_DIR}/{filename}'
        plt.savefig(save_path)
        plt.close()
        print(f"   Saved: {save_path}")

    # 1. RMSE
    rmse_vals = [results_error[m]['RMSE'] for m in models_list]
    plot_comparison('RMSE', rmse_vals, 'So sánh Sai số RMSE', 
                   'rmse_comparison.png', 'Reds_d', higher_is_better=False)

    # 2. MAE
    mae_vals = [results_error[m]['MAE'] for m in models_list]
    plot_comparison('MAE', mae_vals, 'So sánh Sai số MAE', 
                   'mae_comparison.png', 'Purples_d', higher_is_better=False)

    # 3. Precision
    prec_vals = [results_ranking[m]['Precision@10'] for m in models_list]
    plot_comparison('Precision@10', prec_vals, 'So sánh Precision@10', 
                   'precision_comparison.png', 'Greens_d', higher_is_better=True)

    # 4. Recall
    rec_vals = [results_ranking[m]['Recall@10'] for m in models_list]
    plot_comparison('Recall@10', rec_vals, 'So sánh Recall@10', 
                   'recall_comparison.png', 'Blues_d', higher_is_better=True)

    # 5. Scatter Plot (Alpha)
    if user_interactions:
        plt.figure(figsize=(10, 6))
        plt.scatter(user_interactions, alpha_values, alpha=0.6, c=alpha_values, cmap='coolwarm')
        plt.colorbar(label='Giá trị Alpha')
        plt.title(f'Alpha thích nghi (N={len(all_users)} users)')
        plt.xlabel('Số phim đã xem (Log Scale)')
        plt.ylabel('Trọng số Alpha (Thiên về CF)')
        plt.xscale('log')
        plt.savefig(f'{OUTPUT_DIR}/alpha_adaptive_analysis.png')
        plt.close()
        print(f"   Saved: {OUTPUT_DIR}/alpha_adaptive_analysis.png")

    end_time = time.time()
    duration = end_time - start_time_total
    print(f"\n✅ HOÀN TẤT! Tổng thời gian chạy: {duration:.2f} giây.")
    print(f"📂 Kiểm tra biểu đồ tại: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_evaluation()