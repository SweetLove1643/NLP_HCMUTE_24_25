# streamlit run c:/Users/Admin/Documents/Study/Projects/HK6/NLP/Application/main.py
import streamlit as st
from numpy import triu
import pandas as pd

import CollectionData as cl
import DataAugment as da
import DataPreProcessing as pp
import TextVectorizer as tv
import ModelClassfication as mc
import Chatbot
import Recommendation as rc
import matplotlib.pyplot as plt
import seaborn as sns

# st.set_page_config(page_title="22110124 - Phan Van Quan", page_icon=":sunglasses:")
# st.title("22110124 - Phan Văn Quân")
if "data_input" not in st.session_state:
    st.session_state.data_input = ""
if 'augmented_data' not in st.session_state:
    st.session_state.augmented_data = {}  # Lưu dữ liệu tăng cường theo key của tab
# Khởi tạo st.session_state.total_accuracy
if 'total_accuracy' not in st.session_state:
    st.session_state.total_accuracy = pd.DataFrame(
        columns=['Model', 'Vectorizer', 'DataSource', 'Accuracy']
    )


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
    option_collectiondata = ["Crawl web", "Tải dữ liệu từ tệp"]
    tab1, tab2 = st.tabs(option_collectiondata)
    with tab1:
        st.subheader("Crawl")
        st.text_input("Url", disabled=True, value=cl.web_url)
        dataframe = cl.CollectionData()
        selected = st.dataframe(dataframe, use_container_width=True, key="dataframe", hide_index=True)
    with tab2:
        st.subheader("Tải dữ liệu từ tệp")
        uploaded_file = st.file_uploader("Chọn file", type=["txt", "csv", "json", "docx"])
        if uploaded_file is not None:
            st.write("Tên tệp:", uploaded_file.name)
            
            # Đọc nội dung tệp ngay khi tải lên
            content = cl.read_file(uploaded_file)
            
            # Nút xem nội dung
            if st.button("Xem nội dung tệp"):
                if content is not None:
                    if uploaded_file.name.endswith('.csv'):
                        st.dataframe(content, use_container_width=True, key="csv")
                    elif uploaded_file.name.endswith('.json'):
                        st.json(content, expanded=True)
                    elif uploaded_file.name.endswith('.docx'):
                        st.text_area("Nội dung tệp DOCX", content, height=300, key="docx")
                    elif uploaded_file.name.endswith('.txt'):
                        st.text_area("Dữ liệu từ file", content, height=300, key="txt")
                else:
                    st.warning("Tệp không hợp lệ hoặc không thể đọc. Vui lòng kiểm tra định dạng.")

            # Nút lưu dữ liệu
            if st.button("Lưu dữ liệu"):
                if content is not None:
                    st.session_state.data_input = content
                    st.write("Dữ liệu đã lưu:")
                    # Hiển thị dữ liệu theo loại
                    if isinstance(content, pd.DataFrame):
                        st.dataframe(content, use_container_width=True, key="saved_csv")
                    elif isinstance(content, (dict, list)):
                        st.json(content, expanded=True, key="saved_json")
                    else:
                        st.text_area("Dữ liệu", content, height=300, key="saved_data")
                    st.success("Dữ liệu đã được lưu thành công!")
                else:
                    st.warning("Không có dữ liệu để lưu. Vui lòng tải lên tệp hợp lệ.")

        else:
            st.info("Vui lòng tải lên một tệp để tiếp tục.")
elif selected_option == "Tăng cường dữ liệu":
    augment_tab = ["Thêm từ", "Thay từ", "Đổi vị trí", "Back translate", "Thay thực thể", "Từ đồng nghĩa", "Tách từ", "Lỗi keyboard"]
    st.header("Tăng cường dữ liệu")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(augment_tab)
    with tab1:
        st.header("Thêm từ")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPInsert"
        )
        
        # Nút Tăng cường
        if st.button("Tăng cường", key="NLPInsert1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    # Lưu văn bản đã chỉnh sửa
                    st.session_state.data_input = aug_text
                    # Gọi hàm tăng cường
                    value = da.NLPInsert(aug_text)
                    # Debug: In giá trị để kiểm tra
                    st.write(f"Debug - Giá trị trả về từ NLPInsert: {value}")
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPInsert'] = value
                        st.text_area("Thêm từ", value=value, key="NLPInsert2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPInsert trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        # Nút Lưu dữ liệu
        if st.button("Lưu dữ liệu", key="NLPInsert3"):
            try:
                # Kiểm tra xem có dữ liệu tăng cường không
                if 'NLPInsert' in st.session_state.augmented_data and st.session_state.augmented_data['NLPInsert']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPInsert']
                    # Hiển thị dữ liệu đã lưu
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Thêm từ", value=st.session_state.data_input, key="NLPInsert4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")
    with tab2:
        st.header("Thay từ")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPSub"
        )
        
        # Nút Tăng cường
        if st.button("Tăng cường", key="NLPSub1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPSubstitute(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPSub'] = value
                        st.text_area("Thay từ", value=value, key="NLPSub2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPSubstitute trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        # Nút Lưu dữ liệu
        if st.button("Lưu dữ liệu", key="NLPSub3"):
            try:
                if 'NLPSub' in st.session_state.augmented_data and st.session_state.augmented_data['NLPSub']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPSub']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Thay từ", value=st.session_state.data_input, key="NLPSub4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab3:
        st.header("Đổi vị trí")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPSwap"
        )
        
        # Nút Tăng cường
        if st.button("Tăng cường", key="NLPSwap1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPSwap(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPSwap'] = value
                        st.text_area("Đổi vị trí", value=value, key="NLPSwap2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPSwap trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        # Nút Lưu dữ liệu
        if st.button("Lưu dữ liệu", key="NLPSwap3"):
            try:
                if 'NLPSwap' in st.session_state.augmented_data and st.session_state.augmented_data['NLPSwap']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPSwap']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Đổi vị trí", value=st.session_state.data_input, key="NLPSwap4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab4:
        st.header("Back translate")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPBackTranslate"
        )
        
        # Nút Tăng cường
        if st.button("Tăng cường", key="NLPBackTranslate1"):
            # try:
            if not aug_text.strip():
                st.warning("Vui lòng nhập dữ liệu để tăng cường.")
            else:
                st.session_state.data_input = aug_text
                value = da.NLPBackTranslate(aug_text)
                if value is not None and isinstance(value, str):
                    st.session_state.augmented_data['NLPBackTranslate'] = value
                    st.text_area("Back translate", value=value, key="NLPBackTranslate2")
                else:
                    st.error("Không thể tăng cường dữ liệu. Hàm NLPBackTranslate trả về None hoặc giá trị không hợp lệ.")
            # except Exception as e:
            #     st.error(f"Lỗi khi tăng cường: {e}")

        # Nút Lưu dữ liệu
        if st.button("Lưu dữ liệu", key="NLPBackTranslate3"):
            try:
                if 'NLPBackTranslate' in st.session_state.augmented_data and st.session_state.augmented_data['NLPBackTranslate']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPBackTranslate']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Back translate", value=st.session_state.data_input, key="NLPBackTranslate4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab5:
        st.header("Thay thực thể")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPReserved"
        )
        if st.button("Tăng cường", key="NLPReserved1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPReserved(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPReserved'] = value
                        st.text_area("Thay thực thể", value=value, key="NLPReserved2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPReserved trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        if st.button("Lưu dữ liệu", key="NLPReserved3"):
            try:
                if 'NLPReserved' in st.session_state.augmented_data and st.session_state.augmented_data['NLPReserved']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPReserved']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Thay thực thể", value=st.session_state.data_input, key="NLPReserved4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab6:
        st.header("Từ đồng nghĩa")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPSynonym"
        )
        if st.button("Tăng cường", key="NLPSynonym1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPSynonym(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPSynonym'] = value
                        st.text_area("Từ đồng nghĩa", value=value, key="NLPSynonym2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPSynonym trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        if st.button("Lưu dữ liệu", key="NLPSynonym3"):
            try:
                if 'NLPSynonym' in st.session_state.augmented_data and st.session_state.augmented_data['NLPSynonym']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPSynonym']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Từ đồng nghĩa", value=st.session_state.data_input, key="NLPSynonym4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab7:
        st.header("Tách từ")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPSplit"
        )
        if st.button("Tăng cường", key="NLPSplit1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPSplit(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPSplit'] = value
                        st.text_area("Tách từ", value=value, key="NLPSplit2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPSplit trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        if st.button("Lưu dữ liệu", key="NLPSplit3"):
            try:
                if 'NLPSplit' in st.session_state.augmented_data and st.session_state.augmented_data['NLPSplit']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPSplit']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Tách từ", value=st.session_state.data_input, key="NLPSplit4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab8:
        st.header("Lỗi keyboard")
        aug_text = st.text_area(
            label="Dữ liệu tăng cường",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NLPKeyboard"
        )
        if st.button("Tăng cường", key="NLPKeyboard1"):
            try:
                if not aug_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tăng cường.")
                else:
                    st.session_state.data_input = aug_text
                    value = da.NLPKeyboard(aug_text)
                    if value is not None and isinstance(value, str):
                        st.session_state.augmented_data['NLPKeyboard'] = value
                        st.text_area("Lỗi keyboard", value=value, key="NLPKeyboard2")
                    else:
                        st.error("Không thể tăng cường dữ liệu. Hàm NLPKeyboard trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tăng cường: {e}")

        if st.button("Lưu dữ liệu", key="NLPKeyboard3"):
            try:
                if 'NLPKeyboard' in st.session_state.augmented_data and st.session_state.augmented_data['NLPKeyboard']:
                    st.session_state.data_input = st.session_state.augmented_data['NLPKeyboard']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Lỗi keyboard", value=st.session_state.data_input, key="NLPKeyboard4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu tăng cường để lưu. Vui lòng nhấn 'Tăng cường' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")
elif selected_option == "Tiền xử lí dữ liệu":
    preprocessing_tab = ["Stopwords", "StemmingPorter", "StemmingSnowball", "Lemmatization", "PosTagging", "PosTaggingChart", "Spell Checker", "Ner", "Ner render"]
    st.header("Tiền xử lí dữ liệu")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(preprocessing_tab)
    with tab1:
        st.header("Loại bỏ Stopwords, dấu câu")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="Stopwords"
        )
        if st.button("Tiền xử lí dữ liệu", key="Stopwords1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.StopWord(preprocessing_text)
                    if processed_data is not None and isinstance(processed_data, str):
                        st.session_state.augmented_data['Stopwords'] = processed_data
                        st.text_area("Loại bỏ Stopwords, dấu câu", value=processed_data, key="Stopwords2")
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm StopWord trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")    

        if st.button("Lưu dữ liệu", key="StopwordsSave"):
            try:
                if 'Stopwords' in st.session_state.augmented_data and st.session_state.augmented_data['Stopwords']:
                    st.session_state.data_input = st.session_state.augmented_data['Stopwords']
                    if isinstance(st.session_state.data_input, str):
                        st.text_area("Loại bỏ Stopwords, dấu câu", value=st.session_state.data_input, key="Stopwords4")
                    elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                        st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                        st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

        with tab2:
            st.header("StemmingPorter")
            preprocessing_text = st.text_area(
                label="Dữ liệu tiền xử lí",
                value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
                key="StemmingPorter"
            )
            if st.button("Tiền xử lí dữ liệu", key="StemmingPorter1"):
                try:
                    if not preprocessing_text.strip():
                        st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                    else:
                        st.session_state.data_input = preprocessing_text
                        processed_data = pp.StemmingPorter(preprocessing_text)
                        if processed_data is not None and isinstance(processed_data, str):
                            st.session_state.augmented_data['StemmingPorter'] = processed_data
                            st.text_area("StemmingPorter", value=processed_data, key="StemmingPorter2")
                        else:
                            st.error("Không thể tiền xử lí dữ liệu. Hàm StemmingPorter trả về None hoặc giá trị không hợp lệ.")
                except Exception as e:
                    st.error(f"Lỗi khi tiền xử lí: {e}")

            if st.button("Lưu dữ liệu", key="StemmingPorterSave"):
                try:
                    if 'StemmingPorter' in st.session_state.augmented_data and st.session_state.augmented_data['StemmingPorter']:
                        st.session_state.data_input = st.session_state.augmented_data['StemmingPorter']
                        if isinstance(st.session_state.data_input, str):
                            st.text_area("StemmingPorter", value=st.session_state.data_input, key="StemmingPorter4")
                        elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                            st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                            st.success("Lưu dữ liệu thành công!")
                    else:
                        st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu: {e}")

        with tab3:
            st.header("Stemming Snowball")
            preprocessing_text = st.text_area(
                label="Dữ liệu tiền xử lí",
                value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
                key="StemmingSnowball"
            )
            if st.button("Tiền xử lí dữ liệu", key="StemmingSnowball1"):
                try:
                    if not preprocessing_text.strip():
                        st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                    else:
                        st.session_state.data_input = preprocessing_text
                        processed_data = pp.StemmingSnowball(preprocessing_text)
                        if processed_data is not None and isinstance(processed_data, str):
                            st.session_state.augmented_data['StemmingSnowball'] = processed_data
                            st.text_area("Stemming Snowball", value=processed_data, key="StemmingSnowball2")
                        else:
                            st.error("Không thể tiền xử lí dữ liệu. Hàm StemmingSnowball trả về None hoặc giá trị không hợp lệ.")
                except Exception as e:
                    st.error(f"Lỗi khi tiền xử lí: {e}")

            if st.button("Lưu dữ liệu", key="StemmingSnowballSave"):
                try:
                    if 'StemmingSnowball' in st.session_state.augmented_data and st.session_state.augmented_data['StemmingSnowball']:
                        st.session_state.data_input = st.session_state.augmented_data['StemmingSnowball']
                        if isinstance(st.session_state.data_input, str):
                            st.text_area("Stemming Snowball", value=st.session_state.data_input, key="StemmingSnowball4")
                        elif isinstance(st.session_state.data_input, (pd.DataFrame, dict, list)):
                            st.write("Dữ liệu đã lưu:", st.session_state.data_input)
                            st.success("Lưu dữ liệu thành công!")
                    else:
                        st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu: {e}")

        with tab4:
            st.header("Lemmatization")
            preprocessing_text = st.text_area(
                label="Dữ liệu tiền xử lí",
                value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
                key="Lemmatization"
            )
            if st.button("Tiền xử lí dữ liệu", key="Lemmatization1"):
                try:
                    if not preprocessing_text.strip():
                        st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                    else:
                        st.session_state.data_input = preprocessing_text
                        processed_data = pp.Lemmatization(preprocessing_text)
                        if processed_data is not None and isinstance(processed_data, pd.DataFrame):
                            st.session_state.augmented_data['Lemmatization'] = processed_data
                            st.dataframe(processed_data, use_container_width=True, key="Lemmatization2")
                        else:
                            st.error("Không thể tiền xử lí dữ liệu. Hàm Lemmatization trả về None hoặc giá trị không hợp lệ.")
                except Exception as e:
                    st.error(f"Lỗi khi tiền xử lí: {e}")

            if st.button("Lưu dữ liệu", key="LemmatizationSave"):
                try:
                    if 'Lemmatization' in st.session_state.augmented_data and isinstance(st.session_state.augmented_data['Lemmatization'], pd.DataFrame):
                        st.session_state.data_input = st.session_state.augmented_data['Lemmatization']
                        st.dataframe(st.session_state.data_input, use_container_width=True, key="Lemmatization4")
                        st.success("Lưu dữ liệu thành công!")
                    else:
                        st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu: {e}")
    with tab5:
        st.header("PosTagging")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="PosTagging"
        )
        if st.button("Tiền xử lí dữ liệu", key="PosTagging1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.PosTagging(preprocessing_text)
                    if processed_data is not None and isinstance(processed_data, pd.DataFrame):
                        st.session_state.augmented_data['PosTagging'] = processed_data
                        st.dataframe(processed_data, use_container_width=True, key="PosTagging2")
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm PosTagging trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")

        if st.button("Lưu dữ liệu", key="PosTaggingSave"):
            try:
                if 'PosTagging' in st.session_state.augmented_data and isinstance(st.session_state.augmented_data['PosTagging'], pd.DataFrame):
                    st.session_state.data_input = st.session_state.augmented_data['PosTagging']
                    st.dataframe(st.session_state.data_input, use_container_width=True, key="PosTagging4")
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab6:
        st.header("PosTaggingChart")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="PosTaggingChart"
        )
        if st.button("Tiền xử lí dữ liệu", key="PosTaggingChart1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.PosTaggingChart(preprocessing_text)
                    if processed_data:
                        st.markdown(processed_data, unsafe_allow_html=True)
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm PosTaggingChart trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")

    with tab7:
        st.header("Spell Checker")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="SpellChecker"
        )
        if st.button("Tiền xử lí dữ liệu", key="SpellChecker1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.spellchecker(preprocessing_text)
                    if processed_data is not None and isinstance(processed_data, str):
                        st.session_state.augmented_data['SpellChecker'] = processed_data
                        st.text_area("Spell Checker (Sửa lại các từ viết sai)", value=processed_data, key="SpellChecker2")
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm spellchecker trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")

        if st.button("Lưu dữ liệu", key="SpellCheckerSave"):
            try:
                if 'SpellChecker' in st.session_state.augmented_data and st.session_state.augmented_data['SpellChecker']:
                    st.session_state.data_input = st.session_state.augmented_data['SpellChecker']
                    st.text_area("Spell Checker (Sửa lại các từ viết sai)", value=st.session_state.data_input, key="SpellChecker4")
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab8:
        st.header("Ner")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="Ner"
        )
        if st.button("Tiền xử lí dữ liệu", key="Ner1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.Ner(preprocessing_text)
                    if processed_data is not None and isinstance(processed_data, pd.DataFrame):
                        st.session_state.augmented_data['Ner'] = processed_data
                        st.dataframe(processed_data, use_container_width=True, key="Ner2")
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm Ner trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")

        if st.button("Lưu dữ liệu", key="NerSave"):
            try:
                if 'Ner' in st.session_state.augmented_data and isinstance(st.session_state.augmented_data['Ner'], pd.DataFrame):
                    st.session_state.data_input = st.session_state.augmented_data['Ner']
                    st.dataframe(st.session_state.data_input, use_container_width=True, key="Ner4")
                    st.success("Lưu dữ liệu thành công!")
                else:
                    st.warning("Không có dữ liệu để lưu. Vui lòng nhấn 'Tiền xử lí dữ liệu' trước.")
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    with tab9:
        st.header("Ner render")
        preprocessing_text = st.text_area(
            label="Dữ liệu tiền xử lí",
            value=st.session_state.data_input if isinstance(st.session_state.data_input, str) else str(st.session_state.data_input),
            key="NerRender"
        )
        if st.button("Tiền xử lí dữ liệu", key="NerRender1"):
            try:
                if not preprocessing_text.strip():
                    st.warning("Vui lòng nhập dữ liệu để tiền xử lí.")
                else:
                    st.session_state.data_input = preprocessing_text
                    processed_data = pp.NerRender(preprocessing_text)
                    if processed_data:
                        st.markdown(processed_data, unsafe_allow_html=True)
                    else:
                        st.error("Không thể tiền xử lí dữ liệu. Hàm NerRender trả về None hoặc giá trị không hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lí: {e}")
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
    classification_tab = ["MultinomialNB", "Logistics Regression", "SVM", "KNeighbors Classifier", "Decision Tree", "Statistic Accuracy"]   
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(classification_tab)
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
            # Tạo bản ghi mới
            new_data = pd.DataFrame({
                'Model': ["MultinomialNB"],
                'Vectorizer': [vectorizertype_selected],
                'DataSource': [datatype_selected],
                'Accuracy': [float(acc)]
            })

            # Thêm vào st.session_state.total_accuracy
            st.session_state.total_accuracy = pd.concat(
                [st.session_state.total_accuracy, new_data],
                ignore_index=True
            )

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
            # Tạo bản ghi mới
            new_data = pd.DataFrame({
                'Model': ["Logistics Regression"],
                'Vectorizer': [vectorizertype_selected],
                'DataSource': [datatype_selected],
                'Accuracy': [float(acc)]
            })

            # Thêm vào st.session_state.total_accuracy
            st.session_state.total_accuracy = pd.concat(
                [st.session_state.total_accuracy, new_data],
                ignore_index=True
            )
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
            # Tạo bản ghi mới
            new_data = pd.DataFrame({
                'Model': ["SVM"],
                'Vectorizer': [vectorizertype_selected],
                'DataSource': [datatype_selected],
                'Accuracy': [float(acc)]
            })

            # Thêm vào st.session_state.total_accuracy
            st.session_state.total_accuracy = pd.concat(
                [st.session_state.total_accuracy, new_data],
                ignore_index=True
            )
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
            # Tạo bản ghi mới
            new_data = pd.DataFrame({
                'Model': ["KNeighbors Classifier"],
                'Vectorizer': [vectorizertype_selected],
                'DataSource': [datatype_selected],
                'Accuracy': [float(acc)]
            })

            # Thêm vào st.session_state.total_accuracy
            st.session_state.total_accuracy = pd.concat(
                [st.session_state.total_accuracy, new_data],
                ignore_index=True
            )
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
            # Tạo bản ghi mới
            new_data = pd.DataFrame({
                'Model': ["Decision Tree"],
                'Vectorizer': [vectorizertype_selected],
                'DataSource': [datatype_selected],
                'Accuracy': [float(acc)]
            })

            # Thêm vào st.session_state.total_accuracy
            st.session_state.total_accuracy = pd.concat(
                [st.session_state.total_accuracy, new_data],
                ignore_index=True
            )
    with tab6:
        st.subheader("Biểu đồ so sánh Accuracy")
        df = st.session_state.total_accuracy
        st.dataframe(df, use_container_width=True, key="total_accuracy", hide_index=True)
        
        # Kiểm tra DataFrame
        required_columns = ['Model', 'Vectorizer', 'DataSource', 'Accuracy']
        if df.empty or len(df.columns) == 0:
            st.warning("Chưa có dữ liệu để hiển thị biểu đồ.")
        elif not all(col in df.columns for col in required_columns):
            st.error("Dữ liệu thiếu các cột cần thiết: Model, Vectorizer, DataSource, Accuracy")
        else:
            try:
                # Thiết lập kiểu biểu đồ
                sns.set(style="whitegrid")
                # Vẽ Grouped Bar Plot với Subplots
                g = sns.catplot(
                    data=df,
                    x='Vectorizer',
                    y='Accuracy',
                    hue='Model',
                    col='DataSource',
                    kind='bar',
                    height=5,
                    aspect=1.2
                )
                # Tiêu đề và nhãn
                g.set_axis_labels("Vectorizer", "Accuracy")
                g.set_titles("Nguồn dữ liệu: {col_name}")
                g.fig.suptitle("So sánh Accuracy của các mô hình theo Vectorizer và Nguồn dữ liệu", y=1.05)
                # Hiển thị biểu đồ
                st.pyplot(g.fig, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ: {e}")
elif selected_option == "Recommendation":
    st.header("Recommendation")
    user_id = st.number_input("Nhập userid (1 -> 610)", min_value=1, max_value=610, value=1, key="user_id")
    if st.button("Gợi ý phim", key="recommendation_button"):
        recommendations = rc.recommendation_movies(user_id)
        st.subheader("Danh sách phim được gợi ý")
        st.dataframe(recommendations, use_container_width=True, key="recommendation_result", hide_index=True)
elif selected_option == "Chatbot":
    Chatbot.start_generative_ai()

        