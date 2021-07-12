from django.shortcuts import render
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB




category_data = pd.read_csv("idx2category.csv")
idx2category = {row.k: row.v for idx, row in category_data.iterrows()}  #iterrows １行ごとに取り出すmethod

with open("rdmf.pickle", mode="rb") as f:
    model = pickle.load(f)


def index(request):
    if request.method == "GET":
        return render(
            request,
            "nlp/home.html"
        )
    else:
        title = [request.POST["title"]]
        print("title:", title)
        name = title
        result = model.predict(title)[0]
        print("result:", result)
        pred = idx2category[result]
        return render(
            request,
            "nlp/home.html",
            {"title": pred, "name": name}
        )
