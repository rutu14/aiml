import pickle
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
#from spellchecker import SpellChecker
#from textblob import TextBlob
import re

df = pd.read_csv('intent.csv')
df = df[pd.notnull(df['intent'])]
#print(df.head(10))

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
#spell = SpellChecker()

def clean_text(text):

    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    #text = ' '.join(spell.correction(word) for word in text.split())
    #text = ' '.join(str(TextBlob(str(word)).correct()) for word in text.split())
    return text

df['Questions'] = df['Questions'].apply(clean_text)
#data['Questions'] = data['Questions'].apply(clean_text)
#print(df['Questions'])

X_train = df.Questions
y_train = df.intent
#X_test = data.Questions
#y_test = data.intent
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)),
               ])

logreg.fit(X_train, y_train)
'''

y_pred = logreg.predict(X_test)

print(y_pred)
'''


lg_filename='log.pickle'
lg_pkl=open(lg_filename,'wb')
pickle.dump(logreg,lg_pkl)
lg_pkl.close()
