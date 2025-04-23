import requests 
from bs4 import BeautifulSoup
import pandas as pd 
import urllib as ul
import csv

movie_name = []
Urls = []
OtherName = []
NewEp = []
Year = []
ratings = []

url = "https://hoathinh3d.live/phim-hoat-hinh-3d-le"

response = requests.get(url)

print(response.status_code)

html = response.content

soup = BeautifulSoup(html, "html.parser")

movies = soup.find(class_='halim_box').find_all(class_='halim-item')


for movie in movies:
    title = movie.a.h2.text
    movie_name.append(title)

    url = movie.find('a').get("href")
    Urls.append(url)

    movie_respones = requests.get(url)
    movie_soup = BeautifulSoup(movie_respones.content, "html.parser")
    
    OtherName.append(movie_soup.find('p', class_='org_title').text)
    NewEp.append(movie_soup.find('span', class_='new-ep').text)
    Year.append(movie_soup.find(class_='released').text)
    ratings.append(movie_soup.find('span', class_='score').text)
    
dic = {"Movie Name" : movie_name, "Url": Urls, "Other Name": OtherName, "New Ep":NewEp, "Year":Year, "Rate":ratings}
df = pd.DataFrame(dic)
print(df.head(25))
df.to_csv("movies.csv", index=False, encoding="utf-8-sig")
print("Dữ liệu đã được lưu vào 'movies.csv'")