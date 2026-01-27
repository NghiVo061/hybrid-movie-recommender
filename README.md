# 🚀 Hướng dẫn Cài đặt & Chạy Dự án

Để hệ thống hoạt động, bạn vui lòng thực hiện đúng **3 bước** sau theo thứ tự:

### 1️⃣ Bước 1: Cài đặt thư viện
Mở terminal tại thư mục dự án và chạy lệnh:
```bash
pip install streamlit pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

### 2️⃣ Bước 2: Khởi tạo dữ liệu
Hệ thống cần chạy file notebook để tạo ra các file dữ liệu trước khi ứng dụng có thể hoạt động.
1. Mở file **`DataPreprocessing.ipynb`** (bằng VS Code hoặc Jupyter Notebook).
2. Nhấn nút **Run All** (Chạy toàn bộ các ô code).
3. Đợi khoảng 1-2 phút cho đến khi thấy thư mục `data/` xuất hiện.

### 3️⃣ Bước 3: Chạy ứng dụng Web
Sau khi Bước 2 hoàn tất, chạy lệnh sau tại terminal:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở tại: `http://localhost:8501`

---
*Lưu ý: Nếu bạn bỏ qua Bước 2, chương trình sẽ báo lỗi "File not found".*