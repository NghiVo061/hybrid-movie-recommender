import streamlit as st
import pandas as pd

from models.Hybrid import AdaptiveHybridModel

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🎬 Hybrid Movie Recommender",
    layout="wide"
)

st.title("🎬 HỆ THỐNG GỢI Ý PHIM CÁ NHÂN HÓA (HYBRID)")

# =====================================================
# LOAD MODEL – THEO ĐỀ BÀI
# =====================================================
@st.cache_resource
def load_model():
    return AdaptiveHybridModel(
        data_dir="data/processed/evaluation"
    )

hybrid = load_model()

# =====================================================
# SESSION STATE
# =====================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# =====================================================
# SIDEBAR – USER SEARCH
# =====================================================
st.sidebar.header("🔎 User Search")

keyword = st.sidebar.text_input("Nhập User ID (có thể để trống)")
limit = st.sidebar.slider("Số user hiển thị", 3, 20, 10)

users = hybrid.search_user(keyword.strip(), limit)
user_ids = [u["userId"] for u in users] if users else []

if not user_ids:
    st.sidebar.warning("⚠️ Không tìm thấy user")
else:
    selected_user = st.sidebar.selectbox("Chọn User", user_ids)
    st.session_state.user_id = selected_user

top_k = st.sidebar.slider("Top-N phim gợi ý", 5, 20, 10)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Quản lý User",
    "🎬 Content-Based",
    "👥 Collaborative",
    "🧠 Adaptive Hybrid",
    "📊 Evaluation"
])

# =====================================================
# TAB 1 – USER MANAGEMENT
# =====================================================
with tab1:
    st.subheader("👤 User Profile")

    if st.session_state.user_id is None:
        st.info("👈 Hãy chọn User ở Sidebar để bắt đầu")
    else:
        profile = hybrid.get_user_profile(st.session_state.user_id)

        if "error" in profile:
            st.warning(profile["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("User ID", profile["id"])
            c2.metric("Tổng phim đã xem", profile["total_watched"])
            c3.metric("Điểm trung bình", profile["avg_rating"])

            st.markdown("### 🎭 Persona – Top Genres")
            st.write(" • ".join(profile["top_genres"]))

            st.markdown("### ⭐ Top Movies đã thích")
            st.dataframe(
                pd.DataFrame(profile["top_movies"]),
                use_container_width=True
            )

# =====================================================
# TAB 2 – CONTENT BASED
# =====================================================
with tab2:
    st.subheader("🎬 Content-Based Recommendation")

    if st.session_state.user_id is None:
        st.info("Vui lòng chọn User trước")
    else:
        recs = hybrid.cb_model.recommend(
            st.session_state.user_id,
            top_k=top_k
        )

        if recs.empty:
            st.warning("Không có gợi ý Content-Based")
        else:
            st.dataframe(recs, use_container_width=True)

# =====================================================
# TAB 3 – COLLABORATIVE FILTERING
# =====================================================
with tab3:
    st.subheader("👥 Collaborative Filtering")

    if st.session_state.user_id is None:
        st.info("Vui lòng chọn User trước")
    else:
        recs = hybrid.cf_model.recommend(
            st.session_state.user_id,
            top_k=top_k
        )

        if recs.empty:
            st.warning("Không có gợi ý Collaborative")
        else:
            st.dataframe(recs, use_container_width=True)

        st.markdown("### 🤝 Những người dùng có cùng gu")
        sim_users = hybrid.cf_model.get_similar_users(
            st.session_state.user_id
        )

        if sim_users:
            st.dataframe(pd.DataFrame(sim_users))
        else:
            st.info("Không tìm thấy user tương đồng")

# =====================================================
# TAB 4 – ADAPTIVE HYBRID
# =====================================================
with tab4:
    st.subheader("🧠 Adaptive Hybrid Recommendation")

    if st.session_state.user_id is None:
        st.info("Vui lòng chọn User trước")
    else:
        profile = hybrid.get_user_profile(st.session_state.user_id)

        # ---------- COLD START ----------
        if profile["total_watched"] == 0:
            st.warning(
                "👋 Chào mừng bạn! Đây là những phim phổ biến nhất để bắt đầu."
            )
            pop = hybrid.get_popular_recommendations(top_k)
            pop["title"] = pop["title"] + " 🔥 HOT"
            st.dataframe(pop, use_container_width=True)

        # ---------- NORMAL USER ----------
        else:
            alpha = hybrid.calculate_adaptive_weight(
                st.session_state.user_id
            )

            cf_pct = int(alpha * 100)
            cb_pct = 100 - cf_pct

            st.markdown("### ⚖️ Cơ chế Trọng số Hybrid")

            # ===== THANH XANH – CAM =====
            st.markdown(
                f"""
                <div style="width:100%; background:#eee; border-radius:12px; overflow:hidden; height:30px;">
                    <div style="
                        width:{cf_pct}%;
                        background:#2ecc71;
                        height:30px;
                        float:left;
                        text-align:center;
                        color:white;
                        font-weight:bold;
                        line-height:30px;
                    ">
                        CF {cf_pct}%
                    </div>
                    <div style="
                        width:{cb_pct}%;
                        background:#e67e22;
                        height:30px;
                        float:left;
                        text-align:center;
                        color:white;
                        font-weight:bold;
                        line-height:30px;
                    ">
                        CB {cb_pct}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if alpha < 0.4:
                st.info("👶 Người dùng mới – ưu tiên **Content-Based**")
            elif alpha <= 0.7:
                st.info("⚖️ Kết hợp cân bằng giữa **CB & CF**")
            else:
                st.info("🏆 Người dùng kỳ cựu – ưu tiên **Collaborative Filtering**")

            st.caption(
                f"User đã đánh giá {profile['total_watched']} phim — α = {alpha:.3f}"
            )

            if st.button("🚀 GỢI Ý PHIM"):
                recs = hybrid.recommend(
                    st.session_state.user_id,
                    top_k
                )
                st.dataframe(recs, use_container_width=True)

# =====================================================
# TAB 5 – EVALUATION
# =====================================================
with tab5:
    st.subheader("📊 Evaluation Metrics")
    st.info(
        "Tab này dùng để hiển thị RMSE, Precision@K, Recall@K "
        "(load từ file Evaluation)."
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Hybrid Recommender System | "
    "Content-Based + Collaborative Filtering | Streamlit"
)
