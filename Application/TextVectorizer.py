### Text Vectorization
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import FastText
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

def OnehotEncoding(input):
    corpor = []
    nlp = spacy.load("en_core_web_sm")
    sentences = input.split('.')

    for sentence in sentences:
        sen = [(token.strip()) for token in sentence.split(',')]
        corpor.extend(sen)

    data = []
    for sentence in corpor:
        doc = nlp(sentence)
        token = [(token.text) for token in doc if not token.is_stop]
        data.extend(token)

    new_data = np.array(data).reshape(-1, 1)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    result = onehot_encoder.fit_transform(new_data)
    return result, data

def BagofWord(input):
    sentences = []
    input = input.split('.')

    for senten in input:
        sen = senten.split(',')
        for i in sen:
            sentences.append(i)
    for i in sentences:
        if i == '':
            sentences.remove(i)

    bagofword = CountVectorizer()
    result = bagofword.fit_transform(sentences)
    index_token = bagofword.get_feature_names_out()
    return result.toarray(), index_token, sentences

def BagofN_Gram(input):
    sentences = []
    input = input.split('.')

    for senten in input:
        sen = senten.split(',')
        for i in sen:
            sentences.append(i)
    for i in sentences:
        if i == '':
            sentences.remove(i)
    bagofN_gram = CountVectorizer(ngram_range=(2, 2))
    result = bagofN_gram.fit_transform(sentences)
    index_token = bagofN_gram.get_feature_names_out()
    return result.toarray(), index_token, sentences

def TF_IDF(input):
    sentences = []
    input = input.split('.')

    for senten in input:
        sen = senten.split(',')
        for i in sen:
            sentences.append(i)
    for i in sentences:
        if i == '':
            sentences.remove(i)
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(sentences)
    index_token = tfidf.get_feature_names_out()

    return result.toarray(), index_token, sentences

def Text2vec(input, token):
    nlp = spacy.load("en_core_web_sm")
    sentences = []
    input = input.split('.')
    
    for senten in input:
        sen = senten.split(',') 
        for s in sen:
            sentences.append([token.text for token in nlp(s)])
    for i in sentences:
        if i == '':
            sentences.remove(i)
        

    model = Word2Vec(sentences,vector_size=100, window=5, min_count=1)
    model.train(sentences, total_examples=len(sentences), epochs=10)
    return model.wv[token]

def fasttext(input, token):

    sentences = []
    input = input.split('.')
    
    for senten in input:
        sen = senten.split(',')
        for i in sen:
            sentences.append(i)
    for i in sentences:
        if i == '':
            sentences.remove(i)

    model = FastText(sentences,vector_size=100, window=5, min_count=1)
    model.train(sentences, total_examples=len(sentences), epochs=10)

    return model.wv[token]

def berttokenizer(input):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
      
    inputs = tokenizer(input, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)    
        embedding  = output.last_hidden_state
    return embedding