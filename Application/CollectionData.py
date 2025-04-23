from bs4 import BeautifulSoup
import requests
import pandas as pd
from tkinter import filedialog

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


def Inputtxt(uploaded_file):
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8")
            return content
        except Exception as e:
            return None
    return None