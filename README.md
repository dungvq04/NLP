# 📰 Fake News Detection — English & Vietnamese  
**Machine Learning + Transformer (PhoBERT) + Explainable AI**

Dự án này xây dựng hệ thống phân loại tin thật – tin giả sử dụng nhiều phương pháp khác nhau:
- Mô hình học máy cổ điển (TF-IDF + Logistic Regression / XGBoost / Random Forest / Naive Bayes)
- Mô hình Transformer chuyên tiếng Việt: **PhoBERT-base-v2**
- Kỹ thuật giải thích mô hình (Explainable AI) bằng **LIME + spaCy NER**
- Bộ dữ liệu đa ngữ (Anh + Việt), kích thước ~47.000 mẫu

---
data Fake_csv: https://drive.google.com/file/d/1r166EWs9PBznby__0dSeaKnVYrCfYiDD/view?usp=sharing
data True_csv: https://drive.google.com/file/d/1ddHOomIlA6L0RHoBI9-C0Qd6uALjrMSP/view?usp=sharing

---

# 📌 1. Mục tiêu dự án
- Xây dựng pipeline phân loại tin giả hoàn chỉnh.
- So sánh mô hình cổ điển và mô hình Transformer hiện đại.
- Phân tích đặc trưng văn bản (EDA) và trực quan hóa.
- Ứng dụng XAI để giải thích dự đoán, tăng tính minh bạch.
- Tạo cơ sở cho triển khai thực tế (phòng chống tin giả).

---

# 📦 2. Bộ dữ liệu sử dụng
### **Dữ liệu tiếng Anh**
Nguồn: Fake.csv & True.csv  
- Fake: 23,481 mẫu  
- True: 21,417 mẫu  

### **Dữ liệu tiếng Việt**  
Nguồn: ReINTEL/VLSP (train + dev + test)  
→ Khoảng 33.000 mẫu

### **Sau khi gộp và làm sạch**
- **Tổng cộng:** ~ 47.195 mẫu  
- **Label:**  
  - `0` – Tin thật  
  - `1` – Tin giả  

---

# 🧼 3. Tiền xử lý dữ liệu
Các bước chính:
- Kiểm tra trùng lặp, giá trị thiếu  
- Gộp cột `"title"` + `"text"` → `"Content"`  
- Làm sạch văn bản: regex, lowercase, xóa ký tự đặc biệt  
- Loại bỏ stopwords (NLTK)  
- Lemmatization (spaCy)  
- Tạo các đặc trưng thống kê:  
  - số ký tự  
  - số từ  
  - số câu  
- Biểu diễn văn bản bằng TF-IDF (1–3 gram, 15k đặc trưng)

---

# 📊 4. Khám phá dữ liệu (EDA)
- Tin giả có xu hướng **ngắn – ít từ – ít câu**  
- Tin thật dài, đa dạng, phân bố rộng  
- Đặc trưng độ dài là tín hiệu mạnh để phân loại  
- Chủ đề phân bố không đồng đều → thiên lệch về “politics”  

Trực quan hóa:
- Scatter plot (Characters – Words – Sentences)
- Histogram độ dài văn bản (Fake vs True)
- WordCloud
- Biểu đồ chủ đề (subject)

---

# 🤖 5. Các mô hình được huấn luyện
### **Mô hình học máy cổ điển (TF-IDF + ML)**
- Naive Bayes  
- Logistic Regression  
- Random Forest  
- XGBoost  

### **Mô hình Transformer**
- **PhoBERT-base-v2** fine-tune trên 12k mẫu đa ngữ  
- Batch 16 – Epoch 4 – LR=3e-5  

### **Explainable AI**
- LIME để giải thích dự đoán theo từng từ  
- spaCy NER để trích xuất thực thể quan trọng (PERSON, ORG, DATE...)

---

# 📈 6. Kết quả thực nghiệm
### **1) Mô hình cổ điển (TF-IDF)**  
Logistic Regression / Random Forest / XGBoost đều đạt:  
- Accuracy ≈ **98%**  
- F1-score ≈ **0.98**  
- Nhầm lẫn chủ yếu ở các bài viết ngắn hoặc mơ hồ

### **2) PhoBERT (Transformer)**
- Validation Accuracy ≈ **90.67%**  
- Hiểu được ngữ cảnh sâu hơn  
- Hoạt động tốt hơn TF-IDF khi xử lý văn bản tiếng Việt

### **3) Explainable AI**
- LIME hiển thị các từ ảnh hưởng mạnh nhất  
- spaCy NER giúp thấy các thực thể quan trọng  
→ Tăng tính minh bạch của hệ thống

---


