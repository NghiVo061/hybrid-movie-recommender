import streamlit as st
import pandas as pd
from models.Hybrid import AdaptiveHybridModel

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🎬 Adaptive Hybrid Movie Recommender",
    layout="wide"
)

# =========================================================
# INIT MODEL (DUY NHẤT 1 INSTANCE)
# =========================================================
@st.cache_resource
def load_hybrid():
    return AdaptiveHybridModel(
        data_dir="data/processed/production"
    )

hybrid = load_hybrid()

# =========================================================
# SESSION STATE
# =========================================================
if "session_userId" not in st.session_state:
    st.session_state.session_userId = None

# =========================================================
# HELPER
# =========================================================
def safe_display(df, cols):
    return df[[c for c in cols if c in df.columns]]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎬 Movie Recommender")

tab = st.sidebar.radio(
    "📂 Chức năng",
    [
        "👤 User Manager",
        "📚 Content-Based",
        "👥 Collaborative Filtering",
        "🧠 Adaptive Hybrid",
        "📊 Evaluation"
    ]
)

# =========================================================
# TAB 1 – USER MANAGER (AUTOCOMPLETE – FIXED)
# =========================================================
if tab == "👤 User Manager":
    st.title("👤 User Manager")

    # ---------- LOAD USER LIST (ĐÚNG THEO USERMANAGER) ----------
    users = hybrid.search_user("", limit=5000)

    user_ids = [u["userId"] for u in users]
    user_map = {
        u["userId"]: f"User {u['userId']} | {u['count']} ratings"
        for u in users
    }

    # ---------- AUTOCOMPLETE SELECT ----------
    uid = st.selectbox(
        "🔍 Nhập User ID",
        options=user_ids,
        index=None,
        placeholder="Gõ User ID để tìm...",
        format_func=lambda x: user_map.get(x, f"User {x}")
    )

    if uid is None:
        st.info("Vui lòng chọn User ID")
        st.stop()

    st.session_state.session_userId = int(uid)

    # ---------- PROFILE ----------
    profile = hybrid.get_user_profile(uid)

    if "error" not in profile:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🎬 Movies", profile["total_watched"])
        c2.metric("⭐ Avg Rating", round(profile["avg_rating"], 2))
        c3.metric("📊 Level", profile["interaction_level"])

        st.markdown("### 🎭 Top Genres")
        st.write(", ".join(profile["top_genres"]))

        st.markdown("### 🏆 Top Movies")
        st.dataframe(
            pd.DataFrame(profile["top_movies"]),
            use_container_width=True
        )

# =========================================================
# TAB 2 – CONTENT BASED
# =========================================================
elif tab == "📚 Content-Based":
    st.title("📚 Content-Based Recommendation")

    uid = st.session_state.session_userId
    if uid is None:
        st.warning("⚠️ Chọn User trước")
    else:
        recs = hybrid.cb_model.recommend(uid, top_k=10)
        if recs.empty:
            st.warning("Không có gợi ý.")
        else:
            st.dataframe(
                safe_display(
                    recs,
                    ["title", "genres", "score", "avg_rating", "votes"]
                ),
                use_container_width=True
            )

# =========================================================
# TAB 3 – COLLABORATIVE FILTERING
# =========================================================
elif tab == "👥 Collaborative Filtering":
    st.title("👥 Collaborative Filtering")

    uid = st.session_state.session_userId
    if uid is None:
        st.warning("⚠️ Chọn User trước")
    else:
        recs = hybrid.cf_model.recommend(uid, top_k=10)

        if recs.empty:
            st.warning("Không có gợi ý.")
        else:
            st.subheader("🎯 Phim được cộng đồng đề xuất")
            st.dataframe(
                safe_display(recs, ["title", "genres", "score"]),
                use_container_width=True
            )

        # ---------- SIMILAR USERS ----------
        st.subheader("👥 Những người dùng có cùng gu (Top 10)")

        sim_matrix = hybrid.cf_model.user_sim_matrix
        if uid in sim_matrix.index:
            sim_series = (
                sim_matrix.loc[uid]
                .drop(uid)
                .sort_values(ascending=False)
                .head(10)
            )

            df_sim = pd.DataFrame({
                "User ID": sim_series.index.astype(int),
                "Similarity": sim_series.values.round(3)
            })

            st.dataframe(df_sim, use_container_width=True)
        else:
            st.info("Không tìm thấy user tương đồng.")

# =========================================================
# TAB 4 – ADAPTIVE HYBRID
# =========================================================
elif tab == "🧠 Adaptive Hybrid":
    st.title("🧠 Hệ thống Gợi ý Lai Thích nghi")

    uid = st.session_state.session_userId
    if uid is None:
        st.warning("⚠️ Chọn User trước")
    else:
        profile = hybrid.get_user_profile(uid)

        # ---------- COLD START ----------
        if profile.get("total_watched", 0) == 0:
            st.info("👋 Chào mừng bạn! Đây là những bộ phim phổ biến nhất để bạn bắt đầu.")
            recs = hybrid.get_popular_recommendations(top_k=10)
            recs["tag"] = "🔥 POPULAR"
            st.dataframe(
                safe_display(recs, ["title", "genres", "avg_rating", "votes", "tag"]),
                use_container_width=True
            )

        else:
            alpha = hybrid.calculate_adaptive_weight(uid)
            cf_pct = int(alpha * 100)
            cb_pct = 100 - cf_pct

            st.subheader("⚖️ Trọng số Hybrid")

            c1, c2 = st.columns([3, 2])

            with c1:
                st.markdown("🔵 **Collaborative Filtering (Cộng đồng)**")
                st.progress(alpha)

                st.markdown("🟠 **Content-Based (Sở thích cá nhân)**")
                st.progress(1 - alpha)

            with c2:
                st.metric("CF (%)", f"{cf_pct}%")
                st.metric("CB (%)", f"{cb_pct}%")

            # ---------- PERSONA ----------
            if alpha < 0.4:
                st.info("👤 Người mới – tập trung vào sở thích cá nhân.")
            elif alpha <= 0.7:
                st.info("👤 Người dùng cân bằng – kết hợp cá nhân & cộng đồng.")
            else:
                st.success("👤 Thành viên kỳ cựu – ưu tiên cộng đồng.")

            # ---------- HYBRID RECOMMEND ----------
            st.subheader("🎬 Phim đề xuất")
            recs = hybrid.recommend(uid, top_k=10)

            if recs.empty:
                st.warning("Không có gợi ý.")
            else:
                st.dataframe(
                    safe_display(
                        recs,
                        ["title", "genres", "score", "score_cb", "score_cf", "avg_rating", "votes"]
                    ),
                    use_container_width=True
                )

# =========================================================
# TAB 5 – EVALUATION
# =========================================================
elif tab == "📊 Evaluation":
    st.title("📊 Evaluation")

    uploaded = st.file_uploader(
        "Upload file JSON hoặc ảnh",
        type=["json", "png", "jpg"]
    )

    if uploaded:
        if uploaded.name.endswith(".json"):
            st.json(pd.read_json(uploaded))
        else:
            st.image(uploaded, use_container_width=True)
