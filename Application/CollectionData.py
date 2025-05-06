from bs4 import BeautifulSoup
import requests
import pandas as pd
import docx
import json
import pandas as pd
from io import BytesIO

## Thu thập dữ liệu
web_url = "https://vnexpress.net/"
def CollectionData():
    response = requests.get(web_url)
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')
    all_article = soup.find(class_="col-left col-small").find_all(class_="item-news item-news-common")

    
    title = []
    discription = []
    Urls = []

    for article in all_article:
        title_new = article.find("h3").find("a").get_text()
        discription_ = article.find("p").get_text()
        url = article.find("h3").find("a").get("href")

        title.append(title_new)
        discription.append(discription_)
        Urls.append(url)

    dic = {"Title":title, "Url":Urls, "Discription":discription}
    df = pd.DataFrame(dic)
    return df




def read_file(uploaded_file):
    """
    Đọc dữ liệu từ các tệp DOCX, JSON, CSV, hoặc TXT.
    """
    if uploaded_file is None:
        return None

    try:
        # Lấy tên tệp và phần mở rộng
        file_name = uploaded_file.name
        file_extension = file_name.lower().split('.')[-1]

        # Xử lý theo loại tệp
        if file_extension == 'txt':
            # Đọc tệp văn bản (giữ logic ban đầu)
            content = uploaded_file.read().decode('utf-8')
            return content

        elif file_extension == 'docx':
            # Đọc tệp DOCX
            doc = docx.Document(BytesIO(uploaded_file.read()))
            # Trích xuất văn bản từ các đoạn
            content = '\n'.join([para.text for para in doc.paragraphs if para.text])
            return content

        elif file_extension == 'json':
            # Đọc tệp JSON
            content = json.load(uploaded_file)
            return content

        elif file_extension == 'csv':
            # Đọc tệp CSV
            df = pd.read_csv(uploaded_file)
            return df

        else:
            # Phần mở rộng không được hỗ trợ
            return None

    except Exception as e:
        print(f"Lỗi khi đọc tệp: {e}")
        return None