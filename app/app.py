from flask import Flask, request, render_template
import nltk
import string
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_distances
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('omw-1.4')
nltk.download('vader_lexicon')


# app = Flask(__name__)
app = Flask(__name__, template_folder='templates')

df=pd.read_csv('./dialogs.txt',names=('Query','Response'),sep=('\t'))

Text=df['Query']

sid = SentimentIntensityAnalyzer()
for sentence in Text:
     print(sentence)
        
     ss = sid.polarity_scores(sentence)
     for k in ss:
         print('{0}: {1}, ' .format(k, ss[k]), end='')
     print()

analyzer = SentimentIntensityAnalyzer()
df['rating'] = Text.apply(analyzer.polarity_scores)
df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)

punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
df['Query'] = df['Query'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
df['Response'] = df['Response'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

tfidf = TfidfVectorizer()
factors = tfidf.fit_transform(df['Query']).toarray()
tfidf.get_feature_names()


lemmatizer = WordNetLemmatizer()

query = 'who are you ?'
def chatbot(query):
    # step:-1 clean
    query = lemmatizer.lemmatize(query)
    # step:-2 word embedding - transform
    query_vector = tfidf.transform([query]).toarray()
    # step-3: cosine similarity
    similar_score = 1 -cosine_distances(factors,query_vector)
    index = similar_score.argmax() # take max index position
    # searching or matching question
    matching_question = df.loc[index]['Query']
    response = df.loc[index]['Response']
    pos_score = df.loc[index]['pos']
    neg_score = df.loc[index]['neg']
    neu_score = df.loc[index]['neu']
    confidence = similar_score[index][0]
    chat_dict = {'match':matching_question,
                'response':response,
                'score':confidence,
                'pos':pos_score,
                'neg':neg_score,
                'neu':neu_score}
    return chat_dict


# Set up the main route
@app.route('/', methods=['GET', 'POST'])

 # @app.route("/")
def main():
         if request.method == 'GET':
            return (render_template('index.html'))

         if request.method == 'POST':
            m_name = request.form['query']
            title = m_name.title()
            print("New Query" + title)
            response = chatbot(title)
            print( response['pos'] )
            resp = response['response']
            pos_rep = response['pos']
            neg_rep = response['neg']
            neu_rep = response['neu']
          #  resp = response()
          #  print (resp.response)
            
          #  return response['response']
            return render_template('index.html',title=title, resp= resp, positive = pos_rep, negative = neg_rep, neutral = neu_rep )

       #  return getattr(response, str(response.response)) 

    # return "<p>Hello, World!</p>"

# @app.route("/index")
# def main():
#     if flask.request.method == 'GET':
#         return(flask.render_template('index.html'))

if __name__ == '__main__':
     app.run()