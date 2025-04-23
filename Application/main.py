# streamlit run c:/Users/Admin/Documents/Study/Projects/HK6/NLP/Application/main.py
import streamlit as st
from numpy import triu

import CollectionData as cl
import DataAugment as da
import DataPreProcessing as pp
import TextVectorizer as tv
import ModelClassfication as mc
import Chatbot
import Recommendation as rc

# st.set_page_config(page_title="22110124 - Phan Van Quan", page_icon=":sunglasses:")
# st.title("22110124 - Phan Văn Quân")
if "data_input" not in st.session_state:
    st.session_state.data_input = ""


st.markdown(
    """
    <style>
    .main { 
        background-color: #f0f0f0; /* Ví dụ: Thay đổi màu nền */
    }
    .stApp {
        font-family: 'Arial', sans-serif; /* Ví dụ: Thay đổi font chữ */
    }
    </style>
    """,
    unsafe_allow_html=True
)

menu_options = ["Thu thập dữ liệu", 
                "Tăng cường dữ liệu", 
                "Tiền xử lí dữ liệu", 
                "Biểu diễn dữ liệu", 
                "Phân loại dữ liệu", 
                "Recommendation", 
                "Chatbot"]

st.sidebar.title("Menu")
selected_option = st.sidebar.selectbox("Chọn chức năng", menu_options)

if selected_option == "Thu thập dữ liệu":
    st.header("Thu thập dữ liệu")
    option_collectiondata = ["Crawl web", "Txt"]
    tab1, tab2 = st.tabs(option_collectiondata)
    with tab1:
        st.subheader("Crawl")
        st.text_input("Url", disabled=True, value=cl.web_url)
        dataframe = cl.CollectionData()
        selected = st.dataframe(dataframe, use_container_width=True)
    with tab2:
        st.subheader("TXT")
        uploaded_file = st.file_uploader("Chọn file .txt", type=["txt"])
        if uploaded_file:
            content = cl.Inputtxt(uploaded_file)
            st.text_area("Dữ liệu từ file", content if content else "", height=300)
elif selected_option == "Tăng cường dữ liệu":
    augment_tab = ["Thêm từ", "Thay từ", "Đổi vị trí", "Back translate", "Thay thực thể", "Từ đồng nghĩa", "Tách từ", "Lỗi keyboard"]
    st.header("Tăng cường dữ liệu")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(augment_tab)
    with tab1:
        st.header("Thêm từ")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPInsert")
        if st.button("Tăng cường", key="NLPInsert1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Thêm từ", value=da.NLPInsert(aug_text), key="NLPInsert2")
    with tab2:
        st.header("Thay từ")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPSub")
        if st.button("Tăng cường", key="NLPSub1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Thay từ", value=da.NLPSubstitute(aug_text), key="NLPSub2")
    with tab3:
        st.header("Đổi vị trí")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPSwap")
        if st.button("Tăng cường", key="NLPSwap1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Đổi vị trí", value=da.NLPSwap(aug_text), key="NLPSwap2")
    with tab4:
        st.header("Back translate")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPBackTranslate")
        if st.button("Tăng cường", key="NLPBackTranslate1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Back translate", value=da.NLPBackTranslate(aug_text), key="NLPBackTranslate2")
    with tab5:
        st.header("Thay thực thể")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPReserved")
        if st.button("Tăng cường", key="NLPReserved1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Thay thực thể", value=da.NLPReserved(aug_text), key="NLPReserved2")
    with tab6:
        st.header("Từ đồng nghĩa")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPSynonym")
        if st.button("Tăng cường", key="NLPSynonym1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Từ đồng nghĩa", value=da.NLPSynonym(aug_text), key="NLPSynonym2")
    with tab7:
        st.header("Tách từ")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPSplit")
        if st.button("Tăng cường", key="NLPSplit1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Tách từ", value=da.NLPSplit(aug_text), key="NLPSplit2")
    with tab8:
        st.header("Lỗi keyboard")
        aug_text = st.text_area(label="Dữ liệu tăng cường",value=st.session_state.data_input, key="NLPKeyboard")
        if st.button("Tăng cường", key="NLPKeyboard1"):
            
            st.session_state.data_input = aug_text
            st.text_area("Lỗi keyboard", value=da.NLPKeyboard(aug_text), key="NLPKeyboard2")
elif selected_option == "Tiền xử lí dữ liệu":
    preprocessing_tab = ["Stopwords", "StemmingPorter", "StemmingSnowball", "Lemmatization", "PosTagging", "PosTaggingChart", "Spell Checker", "Ner", "Ner render"]
    st.header("Tiền xử lí dữ liệu")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(preprocessing_tab)
    with tab1:
        st.header("Loại bỏ Stopwords, dấu câu")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="Stopwords")
        if st.button("Tiền xử lí dữ liệu", key="Stopwords1"):
            st.session_state.data_input = preprocessing_text
            st.text_area("Loại bỏ Stopwords, dấu câu", value=pp.StopWord(preprocessing_text), key="Stopwords2")
    with tab2:
        st.header("StemmingPorter")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="StemmingPorter")
        if st.button("Tiền xử lí dữ liệu", key="StemmingPorter1"):
            st.session_state.data_input = preprocessing_text
            st.text_area("StemmingPorter", value=pp.StemmingPorter(preprocessing_text), key="StemmingPorter2")
    with tab3:
        st.header("Stemming Snowball")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="StemmingSnowball")
        if st.button("Tiền xử lí dữ liệu", key="StemmingSnowball1"):
            st.session_state.data_input = preprocessing_text
            st.text_area("Stemming Snowball", value=pp.StemmingSnowball(preprocessing_text), key="StemmingSnowball2")
    with tab4:
        st.header("Lemmatization")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="Lemmatization")
        if st.button("Tiền xử lí dữ liệu", key="Lemmatization1"):
            st.session_state.data_input = preprocessing_text
            st.dataframe(pp.Lemmatization(preprocessing_text), use_container_width=True, key="Lemmatization2")
    with tab5:
        st.header("PosTagging")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="PosTagging")
        if st.button("Tiền xử lí dữ liệu", key="PosTagging1"):
            st.session_state.data_input = preprocessing_text
            st.dataframe(pp.PosTagging(preprocessing_text), use_container_width=True, key="PosTagging2")
    with tab6:
        st.header("PosTaggingChart")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="PosTaggingChart")
        if st.button("Tiền xử lí dữ liệu", key="PosTaggingChart1"):
            st.session_state.data_input = preprocessing_text
            st.markdown(pp.PosTaggingChart(preprocessing_text), unsafe_allow_html=True)
    with tab7:
        st.header("Spell Checker")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="SpellChecker")
        if st.button("Tiền xử lí dữ liệu", key="SpellChecker1"):
            st.session_state.data_input = preprocessing_text
            st.text_area("Spell Checker (Sửa lại các từ viết sai)", value=pp.spellchecker(preprocessing_text), key="SpellChecker2")
    with tab8:
        st.header("Ner")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="Ner")
        if st.button("Tiền xử lí dữ liệu", key="Ner1"):
            st.session_state.data_input = preprocessing_text
            st.dataframe(pp.Ner(preprocessing_text), use_container_width=True, key="Ner2")
    with tab9:
        st.header("Ner render")
        preprocessing_text = st.text_area(label="Dữ liệu tiền xử lí",value=st.session_state.data_input, key="NerRender")
        if st.button("Tiền xử lí dữ liệu", key="NerRender1"):
            st.session_state.data_input = preprocessing_text
            st.markdown(pp.NerRender(preprocessing_text), unsafe_allow_html=True)
elif selected_option == "Biểu diễn dữ liệu":
    st.header("Biểu diễn dữ liệu")
    vectorizer_tab = ["One Hot Encoding", 
                      "Bag of Word (BoW)", 
                      "Bag of N-Grams", 
                      "TF-IDF (Term Frequency - Inverse Document Frequency)",
                      "Text2Vec",
                      "FastText",
                      "BertTokenizer"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(vectorizer_tab)
    with tab1:
        st.header("One Hot Encoding")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="OneHotEncoding")
        if st.button("Biểu diễn dữ liệu", key="OneHotEncoding1"):
            st.session_state.data_input = vectorizer_text
            matrix, data = tv.OnehotEncoding(vectorizer_text)
            st.subheader("Dữ liệu ban đầu")
            st.write(data)
            st.subheader("Matrix")
            st.dataframe(matrix, use_container_width=True, key="OneHotEncoding2")
    with tab2:
        st.header("Bag of Word (BoW)")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="BagofWord")
        if st.button("Biểu diễn dữ liệu", key="BagofWord1"):
            st.session_state.data_input = vectorizer_text
            matrix, index_token, sentences = tv.BagofWord(vectorizer_text)
            st.subheader("Dữ liệu ban đầu")
            st.write(sentences)
            st.subheader("Matrix")
            st.dataframe(matrix, use_container_width=True, key="BagofWord2")
            st.subheader("Index token")
            st.write(index_token)
    with tab3:
        st.header("Bag of N-Grams")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="BagofN_Gram")
        if st.button("Biểu diễn dữ liệu", key="BagofN_Gram1"):
            st.session_state.data_input = vectorizer_text
            matrix, index_token, sentences = tv.BagofN_Gram(vectorizer_text)
            st.subheader("Dữ liệu ban đầu")
            st.write(sentences)
            st.subheader("Matrix")
            st.dataframe(matrix, use_container_width=True, key="BagofN_Gram2")
            st.subheader("Index token")
            st.write(index_token)
    with tab4:
        st.header("TF-IDF (Term Frequency - Inverse Document Frequency)")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="TF-IDF")
        if st.button("Biểu diễn dữ liệu", key="TF-IDF1"):
            st.session_state.data_input = vectorizer_text
            matrix, index_token, sentences = tv.TF_IDF(vectorizer_text)
            st.subheader("Dữ liệu ban đầu")
            st.write(sentences)
            st.subheader("Matrix")
            st.dataframe(matrix, use_container_width=True, key="TF-IDF2")
            st.subheader("Index token")
            st.write(index_token)
    with tab5:
        st.header("Text2Vec")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="Text2Vec")
        if vectorizer_text:
            all_tokens = vectorizer_text.split()
            selected_token = st.selectbox("Chọn từ cần tìm vector", all_tokens, key="Text2Vec_token")
        if st.button("Biểu diễn dữ liệu", key="Text2Vec1"):
            st.session_state.data_input = vectorizer_text
            st.write(tv.Text2vec(vectorizer_text, selected_token), key="Text2Vec2")
    with tab6:
        st.header("FastText")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="FastText")
        if vectorizer_text:
            all_tokens = vectorizer_text.split()
            selected_token = st.selectbox("Chọn từ cần tìm vector", all_tokens, key="FastText_token")
        if st.button("Biểu diễn dữ liệu", key="FastText1"):
            st.session_state.data_input = vectorizer_text
            st.write(tv.fasttext(vectorizer_text, selected_token), key="FastText2")
    with tab7:
        st.header("BertTokenizer")
        vectorizer_text = st.text_area(label="Dữ liệu biểu diễn",value=st.session_state.data_input, key="BertTokenizer")
        if st.button("Biểu diễn dữ liệu", key="BertTokenizer1"):
            st.session_state.data_input = vectorizer_text
            st.write(tv.berttokenizer(vectorizer_text), key="BertTokenizer2")
elif selected_option == "Phân loại dữ liệu":
    option_data_select = ["fetch_20newsgroups", "IMDB", "SST", "Amazon Reviews", "TREC", "DBPedia"]
    option_vectorizer = ["TF-IDF Vectorizer",
                         "Bag Of Word",
                         "Bag Of Word N-GRAM",
                         "One Hot Encoder",
                         "Word2Vec",
                         "FastText",
                         "Bert Tokenizer"]
    st.header("Phân loại dữ liệu")
    classification_tab = ["MultinomialNB", "Logistics Regression", "SVM", "KNeighbors Classifier", "Decision Tree"]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(classification_tab)
    with tab1:
        st.header("MultinomialNB")
        datatype_selected = st.selectbox(label="Chọn nguồn dữ liệu", options=option_data_select, key="MultinomialNBdata")
        vectorizertype_selected = st.selectbox(label="Chọn kiểu biểu diễn văn bản", options=option_vectorizer, key="MultinomialNBvector")
        data_check = st.text_area(label="Nhập dữ liệu kiểm tra",value=st.session_state.data_input, key="MultinomialNBinput")
        if st.button("Kiểm tra", key="MultinomialNBbutton"):
            st.session_state.data_input = data_check
            acc, label_result = mc.multinomialnb(vectorizertype_selected, datatype_selected, st.session_state.data_input)
            st.write(f"Độ chính xác của mô hình: {acc}")
            st.write(f"Kết quả dự đoán: {label_result}")

    with tab2:
        st.header("Logistics Regression")
        datatype_selected = st.selectbox(label="Chọn nguồn dữ liệu", options=option_data_select, key="LogisticsRegressiondata")
        vectorizertype_selected = st.selectbox(label="Chọn kiểu biểu diễn văn bản", options=option_vectorizer, key="LogisticsRegressionvector")
        data_check = st.text_area(label="Nhập dữ liệu kiểm tra",value=st.session_state.data_input, key="LogisticsRegressioninput")
        if st.button("Kiểm tra", key="LogisticsRegressionbutton"):
            st.session_state.data_input = data_check
            acc, label_result = mc.logisticsregression(vectorizertype_selected, datatype_selected, st.session_state.data_input)
            st.write(f"Độ chính xác của mô hình: {acc}")
            st.write(f"Kết quả dự đoán: {label_result}")
    with tab3:
        st.header("SVM")
        datatype_selected = st.selectbox(label="Chọn nguồn dữ liệu", options=option_data_select, key="SVMdata")
        vectorizertype_selected = st.selectbox(label="Chọn kiểu biểu diễn văn bản", options=option_vectorizer, key="SVMvector")
        data_check = st.text_area(label="Nhập dữ liệu kiểm tra",value=st.session_state.data_input, key="SVMinput")
        if st.button("Kiểm tra", key="SVMbutton"):
            st.session_state.data_input = data_check
            acc, label_result = mc.svm(vectorizertype_selected, datatype_selected, st.session_state.data_input)
            st.write(f"Độ chính xác của mô hình: {acc}")
            st.write(f"Kết quả dự đoán: {label_result}")
    with tab4:
        st.header("KNeighbors Classifier")
        datatype_selected = st.selectbox(label="Chọn nguồn dữ liệu", options=option_data_select, key="KNeighborsClassifierdata")
        vectorizertype_selected = st.selectbox(label="Chọn kiểu biểu diễn văn bản", options=option_vectorizer, key="KNeighborsClassifiervector")
        data_check = st.text_area(label="Nhập dữ liệu kiểm tra",value=st.session_state.data_input, key="KNeighborsClassifierinput")
        if st.button("Kiểm tra", key="KNeighborsClassifierbutton"):
            st.session_state.data_input = data_check
            acc, label_result = mc.kneighborsclassifier(vectorizertype_selected, datatype_selected, st.session_state.data_input)
            st.write(f"Độ chính xác của mô hình: {acc}")
            st.write(f"Kết quả dự đoán: {label_result}")
    with tab5:
        st.header("Decision Tree")
        datatype_selected = st.selectbox(label="Chọn nguồn dữ liệu", options=option_data_select, key="DecisionTreedate")
        vectorizertype_selected = st.selectbox(label="Chọn kiểu biểu diễn văn bản", options=option_vectorizer, key="DecisionTreevector")
        data_check = st.text_area(label="Nhập dữ liệu kiểm tra",value=st.session_state.data_input, key="DecisionTreeinput")
        if st.button("Kiểm tra", key="DecisionTreebutton"):
            st.session_state.data_input = data_check
            acc, label_result = mc.decisiontree(vectorizertype_selected, datatype_selected, st.session_state.data_input)
            st.write(f"Độ chính xác của mô hình: {acc}")
            st.write(f"Kết quả dự đoán: {label_result}")
elif selected_option == "Recommendation":
    st.header("Recommendation")
    user_id = st.number_input("Nhập userid (1 -> 610)", min_value=1, max_value=610, value=1, key="user_id")
    if st.button("Gợi ý phim", key="recommendation_button"):
        recommendations = rc.recommendation_movies(user_id)
        st.subheader("Danh sách phim được gợi ý")
        st.dataframe(recommendations, use_container_width=True, key="recommendation_result", hide_index=True)
elif selected_option == "Chatbot":
    Chatbot.start_generative_ai()

        