import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
#import the data
movies_df= pd.read_csv("TMDB_movie_dataset.csv")

#data pre-processing

movies_df = movies_df[["id", "title", "genres", "overview", "keywords"]]
movies_df.isnull().sum()
movies_df.dropna(inplace=True)
movies_df.duplicated().sum()
movies_df["overview"]=movies_df["overview"].apply(lambda x:x.split())

movies_df["genres"] = movies_df["genres"].apply(
    lambda x: [i.strip().replace(" ", "") for i in x.split(",")]
)
movies_df["keywords"] = movies_df["keywords"].apply(
    lambda x: [i.strip().replace(" ", "") for i in x.split(",")]
)

movies_df['tags']= movies_df["genres"] + movies_df["overview"] + movies_df["keywords"]

new_df = movies_df[["id", "title", "tags"]]
new_df.loc[:, "tags"] = new_df["tags"].apply(lambda x: " ".join(x))
print(new_df["tags"][0])
