import streamlit as st
import pandas as pd
import os
import sys
import numpy as np

# =========================================================
# CONFIG & IMPORT
# =========================================================
sys.path.append(os.getcwd())
from models.Hybrid import AdaptiveHybridModel

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)


st.markdown("""
<style>
/* ===== HARD REMOVE SIDEBAR COLLAPSE BUTTON (SVG TITLE FIX) ===== */
button[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Một số version dùng thẻ khác */
div[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Chặn luôn SVG title hover */
svg title {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
# =========================================================
# CUSTOM SIDEBAR CSS
# =========================================================
st.markdown("""
<style>
/* ===== SIDEBAR CONTAINER ===== */
section[data-testid="stSidebar"] {
     background-color: #2b2f36;




    border-right: 1px solid #1e293b;
}

/* ===== SIDEBAR TEXT ===== */
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
    font-family: "Inter", system-ui, sans-serif;
}

/* ===== SIDEBAR TITLE ===== */
section[data-testid="stSidebar"] h1 {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

/* ===== RADIO GROUP ===== */
div[role="radiogroup"] {
    gap: 8px;
}

/* ===== RADIO ITEM ===== */
div[role="radiogroup"] label {
     background-color: #2b2f36;
    border: 2px solid #4b5563;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 6px;
    font-weight: 500;
    transition: all 0.25s ease;
}

/* Hover */
div[role="radiogroup"] label:hover {
    background: #020617;
    border-color: #6366f1;
    transform: translateX(2px);
}

/* Checked */
div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    border: none;
    color: white !important;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
}

/* ===== SCROLLBAR ===== */
section[data-testid="stSidebar"] ::-webkit-scrollbar {
    width: 6px;
}

section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #4f46e5, #6366f1);
    border-radius: 10px;
}

/* ===== FOOTER HIDE ===== */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)




# =========================================================
# INIT MODEL
# =========================================================
@st.cache_resource
def load_hybrid():
    return AdaptiveHybridModel(data_dir="data/processed/production")

try:
    hybrid = load_hybrid()
except Exception as e:
    st.error(f"❌ Lỗi khởi tạo hệ thống: {e}")
    st.stop()

# =========================================================
# SESSION STATE
# =========================================================
if "session_userId" not in st.session_state:
    st.session_state.session_userId = None
    


# Hàm hiển thị dataframe an toàn
def safe_display(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()

# =========================================================
# HELPER: FORMAT DATAFRAME
# =========================================================
# Code app.py sau khi đã tiền xử lý dữ liệu chuẩn
def format_result_df(df):
    if df.empty: return df
    
    # Reset index
    df.index = range(1, len(df) + 1)
    
    # Hàm xử lý an toàn cho từng ô
    def safe_format(x):
        # 1. ƯU TIÊN: Kiểm tra nếu là List/Array (Tab 3 Collaborative)
        # Phải kiểm tra cái này trước để tránh lỗi "Ambiguous truth value"
        if isinstance(x, (list, tuple, np.ndarray)):
            # Nối các phần tử trong list lại bằng dấu chấm tròn
            return " • ".join([str(item) for item in x])
        
        # 2. Kiểm tra nếu là Null/NaN (Dùng pd.isna an toàn cho scalar)
        if pd.isna(x) or str(x).strip() == "":
            return ""
            
        # 3. Xử lý String (Tab 2 & 4 - Dữ liệu đã tiền xử lý bằng dấu |)
        text = str(x)
        if '|' in text:
            # Tách bằng dấu gạch đứng, nối lại bằng dấu chấm tròn
            return " • ".join(text.split('|'))
            
        return text

    # Áp dụng
    for col in ['genres', 'tags', 'common_movies']: 
        if col in df.columns:
            df[col] = df[col].apply(safe_format)
            
    return df

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎬 Movie Recommender")


tab = st.sidebar.radio(
    "📂 Chức năng hệ thống",
    [
        "👤 Quản lý Người dùng (User)",
        "📚 Lọc theo Nội dung (Content-Based)",
        "👥 Lọc cộng đồng (Collaborative)",
        "🧠 Gợi ý Lai (Adaptive Hybrid)",
        "📊 Báo cáo Đánh giá (Evaluation)"
    ]
)

# =========================================================
# TAB 1 – USER MANAGER
# =========================================================
if tab == "👤 Quản lý Người dùng (User)":
    st.title("👤 Quản lý & Tìm kiếm Người dùng")
    st.markdown("---")

    if st.session_state.session_userId is None and "uid" in st.query_params:
        try: st.session_state.session_userId = int(st.query_params["uid"])
        except ValueError: pass

    # 1. Prepare Data
    raw_users = hybrid.search_user("", limit=2000)
    user_options = {u['userId']: f"User {u['userId']} ({u['count']} ratings)" for u in raw_users}
    
    # 2. Find Index
    default_index = None
    current_uid = st.session_state.session_userId
    list_keys = list(user_options.keys())
    
    if current_uid is not None and current_uid in user_options:
        try: default_index = list_keys.index(current_uid)
        except ValueError: default_index = None

    # 3. UI
    # Định nghĩa tỷ lệ cột để dùng lại cho khớp nhau
    col_ratio = [5, 0.5, 1.5] 

    c_input, c_clear, c_dummy = st.columns(col_ratio, vertical_alignment="bottom")
    with c_input:
        uid_selected = st.selectbox(
            "🔍 Chọn hoặc Nhập User ID:",
            options=list_keys,
            format_func=lambda x: user_options[x],
            index=default_index,
            placeholder="Gõ ID...",
            help="Gõ số để tìm kiếm ID."
        )

    with c_clear:
        if st.button("🗑️", help="Xóa chọn & Reset"):
            st.session_state.session_userId = None
            if "uid" in st.query_params: del st.query_params["uid"]
            st.rerun()

    if uid_selected is not None and uid_selected != st.session_state.session_userId:
        st.session_state.session_userId = int(uid_selected)
        st.query_params["uid"] = str(uid_selected)
        st.rerun()

    if st.session_state.session_userId is not None:
        uid_display = st.session_state.session_userId
        profile = hybrid.get_user_profile(uid_display)

        if "error" not in profile:
            # === CHỈNH SỬA TẠI ĐÂY ===
            # Tạo cột giống hệt bên trên để thanh success căn thẳng hàng với input
            c_msg, c_void, c_void2 = st.columns(col_ratio)
            with c_msg:
                st.success(f"✅ Đang chọn: User {uid_display}")
            # =========================

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("🎬 Phim đã xem", profile.get("total_watched", 0))
            c2.metric("⭐ Điểm trung bình", profile.get("avg_rating", 0.0))
            c3.metric("📊 Phân loại", profile.get("interaction_level", "N/A"))
            
            st.markdown("---")
            cl, cr = st.columns([2,8])
            with cl:
                st.markdown("**🎭 Thể loại yêu thích:**")
                genres = profile.get("top_genres", [])
                if genres: st.write(", ".join([f"**{g}**" for g in genres]))
                else: st.text("Chưa có dữ liệu.")
            with cr:
                st.markdown("**🏆 Top phim đánh giá cao:**")
                top_movies = profile.get("top_movies", [])
                if top_movies: st.dataframe(pd.DataFrame(top_movies), hide_index=True, use_container_width=True)
                else: st.text("Chưa có dữ liệu.")
        else:
            st.warning(f"⚠️ User ID {uid_display} không tồn tại.")

# =========================================================
# TAB 2 – CONTENT BASED (CÓ SCROLLBAR)
# =========================================================
elif tab == "📚 Lọc theo Nội dung (Content-Based)":
    st.title("📚 Content-Based Filtering")
    st.caption("Gợi ý dựa trên sự tương đồng nội dung phim (Genres, Tags)")
    st.markdown("---")

    uid = st.session_state.session_userId

    if uid is None:
        st.warning("⚠️ Vui lòng quay lại tab 'User Manager' để chọn User trước.")
    else:
        st.markdown(f"**Kết quả gợi ý cho User {uid}:**")
        recs = hybrid.cb_model.recommend(uid, top_k=10)
        
        if recs.empty:
            st.info("ℹ️ Không có gợi ý.")
        else:
            cols_to_show = ["title", "genres", "tags", "score"]
            df_show = safe_display(recs, cols_to_show)
            df_show = format_result_df(df_show)

            df_show = df_show.rename(columns={
                "title": "Tên Phim", "genres": "Thể loại", "tags": "Từ khóa", "score": "Điểm dự đoán"
            })
            
            # QUAN TRỌNG: use_container_width=False để hiện thanh cuộn ngang nếu nội dung dài
            st.dataframe(
                df_show,
                use_container_width=True,
                column_config={
                    "Tên Phim": st.column_config.TextColumn(width="medium"),
                    "Thể loại": st.column_config.TextColumn(width="medium"),
                    "Tags (Từ khóa)": st.column_config.TextColumn(width="large"), # Cột Tags rất dài, cần width large
                    "Điểm dự đoán": st.column_config.NumberColumn(format="%.2f")
                }
            )

# =========================================================
# TAB 3 – COLLABORATIVE (ĐÃ SỬA LỖI HIỂN THỊ HÀNG XÓM ẢO)
# =========================================================
elif tab == "👥 Lọc cộng đồng (Collaborative)":
    st.title("👥 Collaborative Filtering")
    st.caption("Gợi ý dựa trên người dùng tương đồng (User-Based KNN)")
    st.markdown("---")

    uid = st.session_state.session_userId

    if uid is None:
        st.warning("⚠️ Vui lòng quay lại tab 'Quản lý Người dùng (User)' để chọn User trước.")
    else:
        # 1. Gợi ý phim
        st.subheader(f"🎯 Phim đề xuất cho User {uid}")
        
        # Gọi model
        recs = hybrid.cf_model.recommend(uid, top_k=10)
        
        if recs.empty:
            # === TRƯỜNG HỢP KHÔNG CÓ GỢI Ý ===
            st.info("ℹ️ Không tìm thấy gợi ý phù hợp.")
            st.caption("Nguyên nhân: User mới chưa có đủ đánh giá hoặc không tìm thấy người dùng nào có gu tương đồng (KNN Distance quá xa).")
            # Dừng lại tại đây, không hiển thị phần Similar Users bên dưới nữa
            
        else:
            # === TRƯỜNG HỢP CÓ GỢI Ý (HIỂN THỊ CẢ PHIM VÀ NGƯỜI DÙNG) ===
            
            # A. Hiển thị bảng phim
            cols_to_show = ["title", "genres", "score"]
            df_show = safe_display(recs, cols_to_show)
            df_show = format_result_df(df_show)

            df_show = df_show.rename(columns={
                "title": "Tên Phim", "genres": "Thể loại", "score": "Điểm Dự Đoán"
            })

            st.dataframe(
                df_show,
                use_container_width=True,
                column_config={
                    "Tên Phim": st.column_config.TextColumn(width="medium"),
                    "Thể loại": st.column_config.TextColumn(width="medium"),
                    "Điểm Dự Đoán": st.column_config.NumberColumn(format="%.2f")
                }
            )
        
            st.divider()
            
            # B. Tìm người tương đồng (CHỈ HIỆN KHI CÓ RECS)
            st.subheader("👥 Top Người dùng có Gu giống bạn")
            st.caption("Những người dùng này đã đóng góp vào kết quả gợi ý ở trên.")
            
            sim_users = hybrid.cf_model.get_similar_users(uid, top_n=10)
            
            if sim_users:
                df_sim = pd.DataFrame(sim_users)
                
                # Xử lý hiển thị %
                if 'similarity_score' in df_sim.columns:
                    df_sim['similarity_score'] = df_sim['similarity_score'] * 100
                    
                df_sim = format_result_df(df_sim) 

                # Đổi tên cột
                rename_map = {
                    'id': 'User ID', 
                    'similarity_score': 'Độ tương đồng (%)',
                    'common_count': 'Số lượng phim chung', 
                    'common_movies': 'Danh sách phim chung (Sample)'
                }
                cols_to_rename = {k: v for k, v in rename_map.items() if k in df_sim.columns}
                df_sim.rename(columns=cols_to_rename, inplace=True)
                
                st.dataframe(
                    df_sim, 
                    use_container_width=True,
                    column_config={
                        "User ID": st.column_config.NumberColumn(format="%d"),
                        "Độ tương đồng (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Số lượng phim chung": st.column_config.NumberColumn(format="%d 🎬"),
                        "Danh sách phim chung (Sample)": st.column_config.TextColumn(width="large")
                    }
                )
            else:
                st.text("Không thể trích xuất danh sách người dùng tương đồng.")

# =========================================================
# TAB 4 – ADAPTIVE HYBRID (FINAL FIXED VERSION)
# =========================================================
elif tab == "🧠 Gợi ý Lai (Adaptive Hybrid)":
    st.title("🧠 Adaptive Hybrid System")
    st.caption("Kết hợp thông minh giữa CB và CF dựa trên độ tin cậy dữ liệu.")
    st.markdown("---")

    uid = st.session_state.session_userId

    if uid is None:
        st.warning("⚠️ Vui lòng quay lại tab 'Quản lý Người dùng (User)' để chọn User trước.")
    else:
        # ---------------------------------------------------------
        # BƯỚC 1: TÍNH TOÁN TRỌNG SỐ (ALPHA) & KIỂM TRA CF
        # ---------------------------------------------------------
        
        # 1.1 Tính Alpha lý thuyết dựa trên số lượng rating
        raw_alpha = hybrid.calculate_adaptive_weight(uid)
        
        # 1.2 [QUAN TRỌNG] Kiểm tra thực tế: CF có chạy được không?
        # Nếu CF trả về rỗng, ta phải coi như CF thất bại hoàn toàn.
        try:
            cf_check = hybrid.cf_model.recommend(uid, top_k=5)
            is_cf_failed = cf_check.empty
        except Exception:
            is_cf_failed = True

        # 1.3 Quyết định Alpha cuối cùng dùng cho UI
        if is_cf_failed:
            alpha = 0.0  # Ép về 0 (Content-Based only)
            reason_msg = "Không tìm thấy người dùng tương đồng ➔ Chuyển về 100% Content-Based"
            bar_color_cf = "#d3d3d3" # Màu xám (Disable)
        else:
            alpha = raw_alpha
            rating_count = hybrid.user_manager.user_counts.get(uid, 0)
            reason_msg = f"✅ Dựa trên {rating_count} lượt đánh giá của User"
            bar_color_cf = "#EF553B" # Màu cam (Active)

        # Tính phần trăm để vẽ biểu đồ
        pct_cf = alpha * 100
        pct_cb = (1 - alpha) * 100

        # ---------------------------------------------------------
        # BƯỚC 2: HIỂN THỊ THANH TRỌNG SỐ (ADAPTIVE WEIGHT BAR)
        # ---------------------------------------------------------
        st.subheader("⚖️ Cơ chế Trọng số Thích nghi (Adaptive Weight)")
        
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 2, 1])

            with c1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="margin:0; color: #00CC96;">{pct_cb:.1f}%</h3>
                    <p style="font-size: 0.9em; color: gray;">🧩 Content-Based</p>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                # Vẽ thanh Bar HTML
                st.markdown(f"""
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 0.8em;">
                        <span>Nội dung</span>
                        <span style="font-weight: bold;">Alpha: {alpha:.3f}</span>
                        <span>Cộng đồng</span>
                    </div>
                    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; height: 15px; overflow: hidden; display: flex;">
                        <div style="width: {pct_cb}%; background-color: #00CC96; height: 100%;"></div>
                        <div style="width: {pct_cf}%; background-color: {bar_color_cf}; height: 100%;"></div>
                    </div>
                    <div style="text-align: center; font-size: 0.75em; color: { 'red' if is_cf_failed else 'gray' }; margin-top: 5px;">
                        <i>{reason_msg}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                cf_text_color = "#EF553B" if not is_cf_failed else "#b0b0b0"
                st.markdown(f"""
                <div style="text-align: center;">
                    <h3 style="margin:0; color: {cf_text_color};">{pct_cf:.1f}%</h3>
                    <p style="font-size: 0.9em; color: gray;">👥 Collaborative</p>
                </div>
                """, unsafe_allow_html=True)

        # ---------------------------------------------------------
        # BƯỚC 3: TẠO GỢI Ý (RECOMMENDATION LOGIC)
        # ---------------------------------------------------------
        st.subheader("🎬 Kết quả Gợi ý Cuối cùng")
        
        recs = pd.DataFrame() 

        with st.spinner("Đang tổng hợp kết quả đa mô hình..."):
            if is_cf_failed:
                # === TRƯỜNG HỢP 1: FALLBACK VỀ CONTENT-BASED ===
                # Gọi trực tiếp CB model để đảm bảo kết quả GIỐNG HỆT Tab 2
                recs = hybrid.cb_model.recommend(uid, top_k=10)
                
                if not recs.empty:
                    # Tạo các cột giả lập để hiển thị đúng định dạng Hybrid
                    recs["score_cb"] = recs["score"]  # Điểm CB chính là điểm gốc
                    recs["score_cf"] = 0.0            # Điểm CF bằng 0
                    # "score" giữ nguyên là điểm CB
            else:
                # === TRƯỜNG HỢP 2: CHẠY HYBRID BÌNH THƯỜNG ===
                recs = hybrid.recommend(uid, top_k=10)
        
        # ---------------------------------------------------------
        # BƯỚC 4: HIỂN THỊ KẾT QUẢ
        # ---------------------------------------------------------
        if recs.empty:
            st.warning("Không tìm thấy gợi ý phù hợp.")
        else:
            # Danh sách cột cần hiển thị
            cols_to_show = ["title", "genres", "score", "score_cb", "score_cf"]
            
            # Hàm safe_display lọc cột an toàn
            df_show = safe_display(recs, cols_to_show)
            
            # Format dữ liệu (chuyển list thành string, index lại từ 1)
            df_show = format_result_df(df_show)
            
            # Đổi tên cột sang tiếng Việt
            df_show = df_show.rename(columns={
                "title": "Tên Phim", 
                "genres": "Thể loại",
                "score": "Điểm Hybrid", 
                "score_cb": "Điểm CB",
                "score_cf": "Điểm CF"
            })

            # Hiển thị DataFrame
            st.dataframe(
                df_show,
                use_container_width=True,
                column_config={
                    "Tên Phim": st.column_config.TextColumn(width="medium"),
                    "Thể loại": st.column_config.TextColumn(width="medium"),
                    "Điểm Hybrid": st.column_config.NumberColumn(format="%.2f"),
                    "Điểm CB": st.column_config.NumberColumn(format="%.2f"),
                    "Điểm CF": st.column_config.NumberColumn(format="%.2f")
                }
            )

# =========================================================
# TAB 5 – EVALUATION
# =========================================================
elif tab == "📊 Báo cáo Đánh giá (Evaluation)":
    st.title("📊 Kết quả Đánh giá Thực nghiệm")
    st.caption("Các biểu đồ này được load từ thư mục 'static/evaluation_charts'.")
    st.markdown("---")

    CHARTS_DIR = "static/evaluation_charts"
    chart_files = [
        ("rmse_comparison.png", "So sánh RMSE (Thấp hơn là tốt hơn)"),
        ("mae_comparison.png", "So sánh MAE (Thấp hơn là tốt hơn)"),
        ("precision_comparison.png", "So sánh Precision@10 (Cao hơn là tốt hơn)"),
        ("recall_comparison.png", "So sánh Recall@10 (Cao hơn là tốt hơn)"),
        ("alpha_adaptive_analysis.png", "Phân tích sự thích nghi của Alpha")
    ]
    
    if os.path.exists(CHARTS_DIR):
        cols = st.columns(2)
        found_any = False
        for i, (filename, caption) in enumerate(chart_files):
            path = os.path.join(CHARTS_DIR, filename)
            if os.path.exists(path):
                found_any = True
                with cols[i % 2]:
                    st.image(path, caption=caption, use_container_width=True)
                    st.divider()
        if not found_any: st.warning("⚠️ Không có ảnh báo cáo.")
    else:
        st.warning("⚠️ Chưa tìm thấy thư mục biểu đồ.")
