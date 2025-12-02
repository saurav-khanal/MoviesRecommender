import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer


cv = CountVectorizer(max_features=5000, stop_words='english')
ps = PorterStemmer()

#import the data
movies_df= pd.read_csv("data.csv")

#data pre-processing

movies_df = movies_df[["id", "title", "genres", "overview", "keywords"]]
movies_df.isnull().sum()
movies_df.dropna(inplace=True)
movies_df.duplicated().sum()
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies_df["genres"] = movies_df["genres"].apply(
    lambda x: [i.strip().replace(" ", "") for i in x]
)
movies_df["keywords"] = movies_df["keywords"].apply(
    lambda x: [i.strip().replace(" ", "") for i in x]
)

movies_df["overview"]=movies_df["overview"].apply(lambda x:x.split())

movies_df["genres"] = movies_df["genres"].apply(lambda x: [i.strip().replace(" ", "") for i in x])
movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [i.strip().replace(" ", "") for i in x])


movies_df['tags']= movies_df["genres"] + movies_df["overview"] + movies_df["keywords"]

new_df = movies_df[["id", "title", "tags"]]
new_df.loc[:, "tags"] = new_df["tags"].apply(lambda x: " ".join(x))

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df.loc[:, "tags"] = new_df["tags"].apply(stem)
vectors = cv.fit_transform(new_df["tags"]).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in new_df["title"].values:
        print(f"Movie '{movie}' not found.")
        return
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
