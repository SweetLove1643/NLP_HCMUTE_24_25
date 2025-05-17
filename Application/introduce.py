import streamlit as st

def introduce():        
    st.markdown("""
        <style>
        .main {
            background-color: #f9f9f9;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 20px;
        }
        .info-box {
            background-color: #ecf0f1;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        .student-info, .intro-text {
            font-size: 1.1em;
            color: #34495e;
            text-align: left;
            line-height: 1.6;
        }
        .footer {
            margin-top: 50px;
            font-size: 0.9em;
            color: #7f8c8d;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Tiêu đề
    st.markdown('<div class="title">NLP Project - HCMUTE</div>', unsafe_allow_html=True)

    # Thông tin sinh viên
    st.markdown("""
    <div class="info-box">
        <p class="student-info"><strong>Tên sinh viên:</strong> Phan Văn Quân</p>
        <p class="student-info"><strong>MSSV:</strong> 22110214</p>
        <p class="student-info"><strong>Lớp:</strong> 22110CLAI</p>
        <p class="student-info"><strong>Môn học:</strong> Xử lý ngôn ngữ tự nhiên (NLP)</p>
    </div>
    """, unsafe_allow_html=True)

    # Giới thiệu dự án
    st.markdown("""
    <div class="info-box">
        <h2 style="text-align:center; color:#2c3e50;">🌟 Giới thiệu Dự án</h2>
        <div class="intro-text">
            <p><strong>Link GitHub:</strong> <a href="https://github.com/SweetLove1643/NLP_HCMUTE_24_25" target="_blank">https://github.com/SweetLove1643/NLP_HCMUTE_24_25</a></p>
            <p>Dự án tổng hợp cuối kỳ môn Xử lý ngôn ngữ tự nhiên tại HCMUTE. Ứng dụng các kỹ thuật NLP như thu thập dữ liệu, tăng cường dữ liệu, tiền xử lý, biểu diễn, phân loại, recommendation và chatbot — tất cả đều được triển khai qua giao diện web sử dụng <strong>Streamlit</strong>.</p>
            <ul>
                <li><strong>Thu thập dữ liệu</strong> từ web (vnexpress.net) và file (.txt, .csv, .json, .docx,...)</li>
                <li><strong>Tăng cường dữ liệu</strong>: nlpaug (thêm, thay, đổi vị trí từ), back translation (Helsinki), đồng nghĩa, lỗi keyboard,...</li>
                <li><strong>Tiền xử lý</strong>: stopword, stemming, lemmatization, POS tagging, spellchecking, NER</li>
                <li><strong>Biểu diễn văn bản</strong>: One-hot, TF-IDF, Bag-of-Words, Word2Vec, FastText, BERT</li>
                <li><strong>Phân loại văn bản</strong>: MultinomialNB, Logistic Regression, SVM, KNN, Decision Tree</li>
                <li><strong>Recommendation</strong>: cosine similarity + USE</li>
                <li><strong>Chatbot</strong>: gọi API gemini-1.5-flash</li>
            </ul>
            <p>Dữ liệu sử dụng:</p>
            <ul>
                <li><a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download" target="_blank">IMDB</a></li>
                <li><a href="https://www.kaggle.com/datasets/youssefaboelwafa/sst-dataset" target="_blank">SST</a></li>
                <li><a href="https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews" target="_blank">Amazon Reviews</a></li>
                <li><a href="https://www.kaggle.com/datasets/thedevastator/the-trec-question-classification-dataset-a-longi" target="_blank">TREC</a></li>
                <li><a href="https://www.kaggle.com/datasets/danofer/dbpedia-classes" target="_blank">DBPedia</a></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">© 2025 NLP Project by Phan Văn Quân - 22110214</div>', unsafe_allow_html=True)

