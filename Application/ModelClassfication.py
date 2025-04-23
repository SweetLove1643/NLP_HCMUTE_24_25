import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models import FastText

from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def multinomialnb(vectorizer, data, input):
    if data == "fetch_20newsgroups":
        datasets = fetch_20newsgroups()
        x, y = datasets.data, datasets.target
    elif data == "IMDB":
        datasets = pd.read_csv("datasets/IMDBDataset.csv")
        x, y = datasets['review'], datasets['sentiment'].map({'positive': 1, 'negative': 0})
    elif data =="SST":
        datasets = pd.read_csv('datasets/SST.csv')
        x, y = datasets['sentence'], datasets['label']
    elif data == "Amazon Reviews":
        datasets = pd.read_csv('datasets/AmazonReviews.csv', nrows=20000, names=["label", "title", "review"])
        x, y = datasets['review'], datasets['label'] - 1
    elif data == "TREC":
        datasets = pd.read_csv('datasets/TREC.csv')
        x, y = datasets['text'] ,datasets['label-coarse']
    elif data == "DBPedia":
        datasets = pd.read_csv('datasets/DBPEDIA.csv')
        x, y = datasets['text'], datasets['l1']

    
    input = np.array([input])

    def document_to_vector(doc, model): # chuyển list token trong câu thành một mảng vecto tương ứng mỗi token
        words = word_tokenize(doc.lower())
        words_vector = [model.wv[word] for word in words if word in model.wv]
        if len(words_vector) == 0:
            return np.zeros(model.vector_size)
        return np.mean(words_vector, axis=0)
    
    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        # Lấy trung bình embeddings của tất cả token
        return np.mean(embeddings.numpy()[0], axis=0)


    if vectorizer == "TF-IDF Vectorizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        vectorizer = TfidfVectorizer(stop_words='english')
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "Bag Of Word":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if  vectorizer == "Bag Of Word N-GRAM":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45)
        vectorizer = CountVectorizer(ngram_range=(2,2))
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "One Hot Encoder":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
        vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train = np.array(x_train).reshape(-1, 1)
        x_train = vectorizer.fit_transform(x_train)
        x_test = np.array(x_test).reshape(-1, 1)
        x_test = vectorizer.transform(x_test)
        input = np.array(input).reshape(-1, 1)
        input = vectorizer.transform(input)
    if vectorizer == "Word2Vec":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        model = Word2Vec(sentences=x_tokenized, vector_size=100, window=5, min_count=2)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "FastText":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        model = FastText(sentences=x_tokenized, vector_size=100, window=5, min_count=2, min_n=3, max_n=6)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "Bert Tokenizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        x_train_vector = np.array([get_bert_embedding(doc) for doc in x_train])
        x_test_vector = np.array([get_bert_embedding(doc) for doc in x_test])
        input_vector = np.array([get_bert_embedding(input)])

        # Chuẩn hóa vì MultinomialNB yêu cầu không âm
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_vector)
        x_test = scaler.transform(x_test_vector)
        input = scaler.transform(input_vector)



    model = MultinomialNB()
    model.fit(x_train, y_train)     

    pred = model.predict(x_test)
    inputpredect = model.predict(input)

    metric = metrics.classification_report(y_test, pred, output_dict=True)

    accuracymodel = metric['accuracy']
    return accuracymodel, inputpredect[0]

def logisticsregression(vectorizer, data, input):
    if data == "fetch_20newsgroups":
        datasets = fetch_20newsgroups()
        x, y = datasets.data, datasets.target
    elif data == "IMDB":
        datasets = pd.read_csv('datasets/IMDBDataset.csv')
        x, y = datasets['review'], datasets['sentiment'].map({'positive': 1, 'negative': 0})
    elif data =="SST":
        datasets = pd.read_csv('datasets/SST.csv')
        x, y = datasets['sentence'], datasets['label']
    elif data == "Amazon Reviews":
        datasets = pd.read_csv('datasets/AmazonReviews.csv', nrows=20000, names=["label", "title", "review"])
        x, y = datasets['review'], datasets['label'] - 1
    elif data == "TREC":
        datasets = pd.read_csv('datasets/TREC.csv')
        x, y = datasets['text'] ,datasets['label-coarse']
    elif data == "DBPedia":
        datasets = pd.read_csv('datasets/DBPEDIA.csv')
        x, y = datasets['text'], datasets['l1']
    input = np.array([input])

    def document_to_vector(doc, model): # chuyển list token trong câu thành một mảng vecto tương ứng mỗi token
        words = word_tokenize(doc.lower())
        words_vector = [model.wv[word] for word in words if word in model.wv]
        if len(words_vector) == 0:
            return np.zeros(model.vector_size)
        return np.mean(words_vector, axis=0)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        # Lấy trung bình embeddings của tất cả token
        return np.mean(embeddings.numpy()[0], axis=0)


    if vectorizer == "TF-IDF Vectorizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        vectorizer = TfidfVectorizer(stop_words='english')
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "Bag Of Word":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if  vectorizer == "Bag Of Word N-GRAM":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45)
        vectorizer = CountVectorizer(ngram_range=(2,2))
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "One Hot Encoder":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
        vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train = np.array(x_train).reshape(-1, 1)
        x_train = vectorizer.fit_transform(x_train)
        x_test = np.array(x_test).reshape(-1, 1)
        x_test = vectorizer.transform(x_test)
        input = np.array(input).reshape(-1, 1)
        input = vectorizer.transform(input)
    if vectorizer == "Word2Vec":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        model = Word2Vec(sentences=x_tokenized, vector_size=100, window=5, min_count=2)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "FastText":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        model = FastText(sentences=x_tokenized, vector_size=100, window=5, min_count=2, min_n=3, max_n=6)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "Bert Tokenizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        x_train_vector = np.array([get_bert_embedding(doc) for doc in x_train])
        x_test_vector = np.array([get_bert_embedding(doc) for doc in x_test])
        input_vector = np.array([get_bert_embedding(input)])

        # Chuẩn hóa vì MultinomialNB yêu cầu không âm
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_vector)
        x_test = scaler.transform(x_test_vector)
        input = scaler.transform(input_vector)



    model = LogisticRegression()
    model.fit(x_train, y_train)     

    pred = model.predict(x_test)
    inputpredect = model.predict(input)

    metric = metrics.classification_report(y_test, pred, output_dict=True)

    accuracymodel = metric['accuracy']
    return accuracymodel, inputpredect[0]

def svm(vectorizer, data, input):
    if data == "fetch_20newsgroups":
        datasets = fetch_20newsgroups()
        x, y = datasets.data, datasets.target
    elif data == "IMDB":
        datasets = pd.read_csv('datasets/IMDBDataset.csv')
        x, y = datasets['review'], datasets['sentiment'].map({'positive': 1, 'negative': 0})
    elif data =="SST":
        datasets = pd.read_csv('datasets/SST.csv')
        x, y = datasets['sentence'], datasets['label']
    elif data == "Amazon Reviews":
        datasets = pd.read_csv('datasets/AmazonReviews.csv', nrows=20000, names=["label", "title", "review"])
        x, y = datasets['review'], datasets['label'] - 1
    elif data == "TREC":
        datasets = pd.read_csv('datasets/TREC.csv')
        x, y = datasets['text'] ,datasets['label-coarse']
    elif data == "DBPedia":
        datasets = pd.read_csv('datasets/DBPEDIA.csv')
        x, y = datasets['text'], datasets['l1']
    input = np.array([input])

    def document_to_vector(doc, model): # chuyển list token trong câu thành một mảng vecto tương ứng mỗi token
        words = word_tokenize(doc.lower())
        words_vector = [model.wv[word] for word in words if word in model.wv]
        if len(words_vector) == 0:
            return np.zeros(model.vector_size)
        return np.mean(words_vector, axis=0)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        # Lấy trung bình embeddings của tất cả token
        return np.mean(embeddings.numpy()[0], axis=0)


    if vectorizer == "TF-IDF Vectorizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        vectorizer = TfidfVectorizer(stop_words='english')
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "Bag Of Word":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if  vectorizer == "Bag Of Word N-GRAM":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45)
        vectorizer = CountVectorizer(ngram_range=(2,2))
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "One Hot Encoder":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
        vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train = np.array(x_train).reshape(-1, 1)
        x_train = vectorizer.fit_transform(x_train)
        x_test = np.array(x_test).reshape(-1, 1)
        x_test = vectorizer.transform(x_test)
        input = np.array(input).reshape(-1, 1)
        input = vectorizer.transform(input)
    if vectorizer == "Word2Vec":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        model = Word2Vec(sentences=x_tokenized, vector_size=100, window=5, min_count=2)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "FastText":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        model = FastText(sentences=x_tokenized, vector_size=100, window=5, min_count=2, min_n=3, max_n=6)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "Bert Tokenizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        x_train_vector = np.array([get_bert_embedding(doc) for doc in x_train])
        x_test_vector = np.array([get_bert_embedding(doc) for doc in x_test])
        input_vector = np.array([get_bert_embedding(input)])

        # Chuẩn hóa vì MultinomialNB yêu cầu không âm
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_vector)
        x_test = scaler.transform(x_test_vector)
        input = scaler.transform(input_vector)


    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(x_train, y_train)     

    pred = model.predict(x_test)
    inputpredect = model.predict(input)

    metric = metrics.classification_report(y_test, pred, output_dict=True)

    accuracymodel = metric['accuracy']
    return accuracymodel, inputpredect[0]

def kneighborsclassifier(vectorizer, data, input):
    if data == "fetch_20newsgroups":
        datasets = fetch_20newsgroups()
        x, y = datasets.data, datasets.target
    elif data == "IMDB":
        datasets = pd.read_csv('datasets/IMDBDataset.csv')
        x, y = datasets['review'], datasets['sentiment'].map({'positive': 1, 'negative': 0})
    elif data =="SST":
        datasets = pd.read_csv('datasets/SST.csv')
        x, y = datasets['sentence'], datasets['label']
    elif data == "Amazon Reviews":
        datasets = pd.read_csv('datasets/AmazonReviews.csv', nrows=20000, names=["label", "title", "review"])
        x, y = datasets['review'], datasets['label'] - 1
    elif data == "TREC":
        datasets = pd.read_csv('datasets/TREC.csv')
        x, y = datasets['text'] ,datasets['label-coarse']
    elif data == "DBPedia":
        datasets = pd.read_csv('datasets/DBPEDIA.csv')
        x, y = datasets['text'], datasets['l1']
    input = np.array([input])

    def document_to_vector(doc, model): # chuyển list token trong câu thành một mảng vecto tương ứng mỗi token
        words = word_tokenize(doc.lower())
        words_vector = [model.wv[word] for word in words if word in model.wv]
        if len(words_vector) == 0:
            return np.zeros(model.vector_size)
        return np.mean(words_vector, axis=0)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        # Lấy trung bình embeddings của tất cả token
        return np.mean(embeddings.numpy()[0], axis=0)


    if vectorizer == "TF-IDF Vectorizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        vectorizer = TfidfVectorizer(stop_words='english')
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "Bag Of Word":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if  vectorizer == "Bag Of Word N-GRAM":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45)
        vectorizer = CountVectorizer(ngram_range=(2,2))
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "One Hot Encoder":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
        vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train = np.array(x_train).reshape(-1, 1)
        x_train = vectorizer.fit_transform(x_train)
        x_test = np.array(x_test).reshape(-1, 1)
        x_test = vectorizer.transform(x_test)
        input = np.array(input).reshape(-1, 1)
        input = vectorizer.transform(input)
    if vectorizer == "Word2Vec":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        model = Word2Vec(sentences=x_tokenized, vector_size=100, window=5, min_count=2)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "FastText":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        model = FastText(sentences=x_tokenized, vector_size=100, window=5, min_count=2, min_n=3, max_n=6)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "Bert Tokenizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        x_train_vector = np.array([get_bert_embedding(doc) for doc in x_train])
        x_test_vector = np.array([get_bert_embedding(doc) for doc in x_test])
        input_vector = np.array([get_bert_embedding(input)])

        # Chuẩn hóa vì MultinomialNB yêu cầu không âm
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_vector)
        x_test = scaler.transform(x_test_vector)
        input = scaler.transform(input_vector)



    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)     

    pred = model.predict(x_test)
    inputpredect = model.predict(input)

    metric = metrics.classification_report(y_test, pred, output_dict=True)

    accuracymodel = metric['accuracy']
    return accuracymodel, inputpredect[0]

def decisiontree(vectorizer, data, input):
    if data == "fetch_20newsgroups":
        datasets = fetch_20newsgroups()
        x, y = datasets.data, datasets.target
    elif data == "IMDB":
        datasets = pd.read_csv('datasets/IMDBDataset.csv')
        x, y = datasets['review'], datasets['sentiment'].map({'positive': 1, 'negative': 0})
    elif data =="SST":
        datasets = pd.read_csv('datasets/SST.csv')
        x, y = datasets['sentence'], datasets['label']
    elif data == "Amazon Reviews":
        datasets = pd.read_csv('datasets/AmazonReviews.csv', nrows=20000, names=["label", "title", "review"])
        x, y = datasets['review'], datasets['label'] - 1
    elif data == "TREC":
        datasets = pd.read_csv('datasets/TREC.csv')
        x, y = datasets['text'] ,datasets['label-coarse']
    elif data == "DBPedia":
        datasets = pd.read_csv('datasets/DBPEDIA.csv')
        x, y = datasets['text'], datasets['l1']
    input = np.array([input])

    def document_to_vector(doc, model): # chuyển list token trong câu thành một mảng vecto tương ứng mỗi token
        words = word_tokenize(doc.lower())
        words_vector = [model.wv[word] for word in words if word in model.wv]
        if len(words_vector) == 0:
            return np.zeros(model.vector_size)
        return np.mean(words_vector, axis=0)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        # Lấy trung bình embeddings của tất cả token
        return np.mean(embeddings.numpy()[0], axis=0)


    if vectorizer == "TF-IDF Vectorizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        vectorizer = TfidfVectorizer(stop_words='english')
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "Bag Of Word":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if  vectorizer == "Bag Of Word N-GRAM":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=45)
        vectorizer = CountVectorizer(ngram_range=(2,2))
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)
        input = vectorizer.transform(input)
    if vectorizer == "One Hot Encoder":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)
        vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        x_train = np.array(x_train).reshape(-1, 1)
        x_train = vectorizer.fit_transform(x_train)
        x_test = np.array(x_test).reshape(-1, 1)
        x_test = vectorizer.transform(x_test)
        input = np.array(input).reshape(-1, 1)
        input = vectorizer.transform(input)
    if vectorizer == "Word2Vec":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=45)
        model = Word2Vec(sentences=x_tokenized, vector_size=100, window=5, min_count=2)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "FastText":
        x_tokenized = [word_tokenize(doc.lower()) for doc in x] # lấy từng row trong x -> tách ra từng từ -> chuyển chữ thường
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        model = FastText(sentences=x_tokenized, vector_size=100, window=5, min_count=2, min_n=3, max_n=6)

        x_train_vector = np.array([document_to_vector(doc, model) for doc in x_train])
        x_test_vector = np.array([document_to_vector(doc, model) for doc in x_test])
        input_vector = np.array([document_to_vector(input[0], model)])

        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train_vector)
        x_test = scalar.transform(x_test_vector)
        input = scalar.transform(input_vector)
    if vectorizer == "Bert Tokenizer":
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=45)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        x_train_vector = np.array([get_bert_embedding(doc) for doc in x_train])
        x_test_vector = np.array([get_bert_embedding(doc) for doc in x_test])
        input_vector = np.array([get_bert_embedding(input)])

        # Chuẩn hóa vì MultinomialNB yêu cầu không âm
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_vector)
        x_test = scaler.transform(x_test_vector)
        input = scaler.transform(input_vector)



    model = DecisionTreeClassifier(max_depth=100)
    model.fit(x_train, y_train)     

    pred = model.predict(x_test)
    inputpredect = model.predict(input)

    metric = metrics.classification_report(y_test, pred, output_dict=True)

    accuracymodel = metric['accuracy']
    return accuracymodel, inputpredect[0]



