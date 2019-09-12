import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

dataset = pd.read_csv('train.csv',encoding = 'latin-1')
#dataset1= pd.read_csv('train.csv')

dataset['tweet'][0]

clean_tweets=[]
for i in range(31962):
    tweet=re.sub('@[\w]*',' ',dataset['tweet'][i])
    tweet=re.sub('[^a-zA-z#]',' ',tweet)

    tweet=tweet.lower()
    tweet=tweet.split()

    tweet=[ps.stem(token) for token in tweet if not token in stopwords.words('english')]
    tweet=' '.join(tweet)
    clean_tweets.append(tweet)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=3000)
X1= cv.fit_transform(clean_tweets)
X1=X1.toarray()
 
 y= dataset['label'].values
 
 print(cv.get_feature_names())
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
lr.score(X_test,y_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)

from sklearn.naive_bayes import N
import graphviz

