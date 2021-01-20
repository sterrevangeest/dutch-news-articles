import pandas as pd
import csv
import spacy
import json
from spacy import displacy
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nl_core_news_sm
nlp = nl_core_news_sm.load()

df = pd.read_csv('dutch-news-articles.csv')
categories = df['category'].unique()


def export_named_entities(categories):

    for category in categories:
        df_category = df.loc[df['category'] == category]
        all_named_entities = df_category['content'].apply(
            get_all_named_entities)
        sum_named_entities = count_named_entities(all_named_entities)
        create_json(category, sum_named_entities)


def get_all_named_entities(row):
    doc = nlp(row)
    items = []
    for entity in doc.ents:
        if entity.label_ != 'CARDINAL' and entity.label_ != 'DATE' and entity.label_ != 'QUANTITY' and entity.label_ != 'TIME' and entity.label_ != 'ORDINAL' and entity.label_ != 'PERCENT' and entity.label_ != 'MONEY':
            items.append(entity.text)
    print(items)
    return items


def count_named_entities(all_named_entities):
    sum_entities = []
    for items in all_named_entities:
        for item in items:
            sum_entities.append(item)
    sum_entities = dict(Counter(sum_entities).most_common(110)).keys()
    return sum_entities


def create_json(category, sum_named_entities):
    print(category, sum_named_entities)
    result = dict([(item, idx) for idx, item in enumerate(sum_named_entities)])
    with open("./named-entities/" + category + ".json", "w") as out_file:
        json.dump(result, out_file, ensure_ascii=False)


export_named_entities(categories)
