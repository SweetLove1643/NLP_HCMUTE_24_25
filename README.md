
# 🧠 NLP_HCMUTE_24_25

**Link GitHub:** [https://github.com/SweetLove1643/NLP_HCMUTE_24_25](https://github.com/SweetLove1643/NLP_HCMUTE_24_25)

Dự án tổng hợp cuối kỳ môn Xử lý ngôn ngữ tự nhiên tại HCMUTE. Ứng dụng các kỹ thuật NLP như thu thập dữ liệu, tăng cường dữ liệu, tiền xử lý, biểu diễn, phân loại, recommendation và chatbot — tất cả đều được triển khai qua giao diện web sử dụng **Streamlit**.

---

## 🚀 Features

### 1. 🗂️ Data Collection
- **Crawl dữ liệu** từ [VnExpress](https://vnexpress.net/)
- **Tải dữ liệu từ file** (.txt, .json, .csv, .docx)

### 2. 🔁 Data Augmentation (`nlpaug`)
- Thêm từ, thay từ, đổi vị trí
- Back translation (Helsinki)
- Thay thế từ đồng nghĩa, thực thể
- Tách từ, lỗi keyboard

### 3. 🧹 Preprocessing (`nltk`, `spellchecker`, `contractions`, ...)
- Stopwords removal, stemming (Porter, Snowball), lemmatization
- POS tagging, NER
- Spellchecking & contraction expansion
- NER visualization

### 4. 📊 Text Vectorization
- One-hot, BoW, N-Gram, TF-IDF
- Text2Vec, FastText
- BERT Tokenizer

### 5. 🤖 Classification
- Mô hình: `MultinomialNB`, `Logistic Regression`, `SVM`, `KNN`, `Decision Tree`
- Dữ liệu: `20newsgroups`, `IMDB`, `SST`, `Amazon Reviews`, `TREC`, `DBPedia`
- Kết hợp vectorizer + model → đánh giá bằng `accuracy`, hiển thị biểu đồ so sánh

### 6. 🎯 Recommendation System
- Dựa trên cosine similarity giữa văn bản (sử dụng Universal Sentence Encoder)
- Đưa ra gợi ý phim cho user theo độ tương đồng

### 7. 💬 Chatbot
- Tích hợp mô hình `gemini-1.5-flash` qua API
- Trả lời hội thoại thông minh theo nội dung người dùng nhập

---

## 📁 Project Structure

```
NLP_HCMUTE_24_25/
├── Application/               # 📌 Project tổng hợp (chạy bằng Streamlit)
│   ├── main.py                # Trang chính
│   ├── Chatbot.py             # Chatbot API
│   ├── CollectionData.py      # Thu thập dữ liệu
│   ├── DataAugment.py         # Tăng cường dữ liệu
│   ├── DataPreProcessing.py   # Tiền xử lý dữ liệu
│   ├── ModelClassfication.py  # Phân loại văn bản
│   ├── Recommendation.py      # Gợi ý dựa trên USE
│   ├── TextVectorizer.py      # Các phương pháp biểu diễn văn bản
│   └── requirements.txt       # Phụ thuộc
├── Week1/ → Week11/           # 📝 Bài tập từng tuần (không chi tiết)
├── README.md                  # File mô tả
```

---

## 📦 Datasets & Pretrained Artefacts

### 🔗 Datasets được sử dụng:

| Dataset            | Link tải từ Kaggle |
|--------------------|--------------------|
| **IMDB**           | [🔗 IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download) |
| **SST**            | [🔗 SST Dataset](https://www.kaggle.com/datasets/youssefaboelwafa/sst-dataset) |
| **Amazon Reviews** | [🔗 Amazon Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) |
| **TREC**           | [🔗 TREC Dataset](https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi) |
| **DBPedia**        | [🔗 DBPedia Dataset](https://www.kaggle.com/datasets/danofer/dbpedia-classes) |

### 🧠 Artefacts:
- Mô hình đã huấn luyện và vectorizer được lưu trong thư mục `models/` và `vectorizers/` (nếu có)

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/SweetLove1643/NLP_HCMUTE_24_25
cd NLP_HCMUTE_24_25/Application

# Tạo môi trường ảo (khuyến khích)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run main.py
```

---

## 📋 Dependencies (Excerpt)

- `streamlit`, `nltk`, `scikit-learn`, `nlpaug`
- `contractions`, `spacy`, `transformers`, `tensorflow`
- `sentence-transformers`, `googletrans`, `docx`, `seaborn`, `matplotlib`

🔧 Xem đầy đủ trong [requirements.txt](Application/requirements.txt)

---

## 🧪 How to Use
Truy cập giao diện web sau khi chạy `main.py`, chọn chức năng từ **Sidebar**:
- "Thu thập dữ liệu"
- "Tăng cường dữ liệu"
- "Tiền xử lí"
- "Biểu diễn"
- "Phân loại"
- "Recommendation"
- "Chatbot"

---

## 🤝 Contributing

Bạn có thể đóng góp bằng cách tạo **pull request**, hoặc mở **issue** để đề xuất tính năng mới hoặc báo lỗi.

---

## 📜 License

This project is licensed under the **MIT License** – see `LICENSE` for details.

---

## 🙏 Acknowledgements

- [nlpaug](https://github.com/makcedward/nlpaug)
- [Hugging Face Transformers](https://huggingface.co/)
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder)
- [Gemini API](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)

---

> Dự án thuộc về nhóm học phần Xử lý Ngôn ngữ Tự Nhiên – HCMUTE (2024–2025)
