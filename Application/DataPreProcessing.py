### Tiền xử lí dữ liệu

import contractions
from spellchecker import SpellChecker
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import spacy
from nltk.stem import PorterStemmer


def Tokenization(input): #Phan doan cau. Tra ve list cau
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    sentences = list(doc.sents)
    return sentences

def StopWord(input):
    nlp = spacy.load("en_core_web_sm")  
    doc = nlp(input)
    result = [token.text for token in doc if not token.is_stop and not token.is_punct]  # Loại bỏ stop words và dấu câu
    result = " ".join(result)
    return result

def StemmingPorter(input): # Chuyển về nguyên mẫu
    stemmer = PorterStemmer()
    doc = Tokenization(input)  
    stemmed_words = [] 
    
    for sentence in doc:  
        for word in sentence: 
            stemmed_words.append(stemmer.stem(word.text)) 
    return " ".join(stemmed_words)

def StemmingSnowball(input):
    stemmer = SnowballStemmer(language='english')
    doc = Tokenization(input)
    stemmed_word = []

    for sentence in doc:
        for word in sentence:
            stemmed_word.append(stemmer.stem(word.text))
    return " ".join(stemmed_word)
    
def Lemmatization(input): # Chuyển về nguyên mẫu + id word, tra ve table
    nlp = spacy.load("en_core_web_sm")
    V = []
    Lemmatization_result = []
    id_word = []
    for token in nlp(input):
        Lemmatization_result.append(token.lemma_)
        V.append(token)
        id_word.append(token.lemma)
    dic = {"Từ gốc":V, "Lemmatization":Lemmatization_result,"id_word":id_word}
    df = pd.DataFrame(dic)

    return df

def PosTagging(input): # Xác định loại từ, định danh từ
    nlp = spacy.load("en_core_web_sm")
    V = []
    PosTag = []
    id = []
    for word in nlp(input):
        V.append(word)
        PosTag.append(word.pos_)
        id.append(word.pos)
    
    dic = {"Từ gốc":V, "Loại từ":PosTag, "id_word":id}
    df = pd.DataFrame(dic)
    return df

def PosTaggingChart(input): #tao chart cho postagging, the hien mqh cac tu
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    return spacy.displacy.render(doc, style="dep")

def Contraction(input): # Sua tu viet tat
    return contractions.fix(input)

def spellchecker(input): # Sửa lại các từ viết sai
    spell = SpellChecker()
    word = input.split()
    result = []
    for token in word:
        if spell.correction(token) is None:
            result.append(token)
        else:
            result.append(spell.correction(token))
    return " ".join(result)

def Ner(input): #Gán nhán, phân loại, sử dụng spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    entity = []
    Detection = []
    for word in (doc.ents):
        entity.append(word.text)
        Detection.append(word.label_)
    dic = {"Thực thể":entity, "Gán nhãn":Detection}
    df = pd.DataFrame(dic)
    return df
def NerRender(input):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)
    return spacy.displacy.render(doc, style='ent')