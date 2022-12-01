import os
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
#import tensorflow as tf
#from transformers import BertTokenizer
#from transformers import TFBertForSequenceClassification
#from official.nlp import optimization

import ftfy
from bs4 import BeautifulSoup
import nltk
from stopwordsiso import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

"""# Data preparation

### Steps

* Remove html tags from product attributes
* Remove mispelling 
* Since the descriptive text does not contain negative verbs that could change the semantics, the last step is to remove stopwords
"""

#Funzione per rimuovere i tag html
def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")
    for data in soup(['style', 'script']):
        data.decompose()

    return ' '.join(soup.stripped_strings)
#Converte una lista in una stringa
def listToString(s):
    str1 = " "
    return (str1.join(s))
#Pulizia del DataSet
def cleanDataset(product_df):
    #Array contenente le stopwords delle lingue del dataset
    stop_words = stopwords(["ja","en","es"])
    tokenizer = RegexpTokenizer(r'\w+')
    for index in tqdm(range(0, len(product_df))):
        for col in range(2,7):
            # Eseguo azioni solo sui valori not NaN
            if not (pd.isnull(product_df.iloc[index, col])):
                # Rimuovo i tag html
                if (bool(BeautifulSoup(product_df.iloc[index, col], "html.parser").find())):
                    product_df.iloc[index, col] = remove_tags(product_df.iloc[index, col])
                # Rimuovo i mispelling
                product_df.iloc[index, col] = ftfy.fix_text(product_df.iloc[index, col])
                # Rimuovo le stopwords
                word_tokens = tokenizer.tokenize(product_df.iloc[index, col])
                product_df.iloc[index, col] = listToString([w for w in word_tokens if not w.lower() in stop_words])

    return product_df

"""# Load Dataset"""

train_df = pd.read_csv('import/train/train_task3_to_clean_200k.csv')

"""## Data Cleaning"""

train_df = cleanDataset(train_df)
train_df.info()

"""## Saving cleaning dataset to csv"""

train_df.to_csv('import/train/train_task3_cleaned_200k.csv', index=False)