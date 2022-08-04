import yfinance as yf
import streamlit as st
from dateutil.relativedelta import relativedelta
from datetime import date
from plotly import graph_objs as go  # for interactive graphs
import pandas as pd
import nltk
from nltk import ngrams 
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 
import numpy
from nltk.draw import dispersion_plot

#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')
#nltk.download('averaged_perceptron_tagger')
#nltk.download("maxent_ne_chunker")
#nltk.download("words")
#nltk.download("Punkt")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Local CSS Sheet
local_css("style.css")


def load_data(ticker):
    START = date.today() - relativedelta(days=10)
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)  
    data.reset_index(inplace=True)
    return data


# Global variables
st.title("Stock Web-App")
data = load_data('GME')
#file = open('wallstreetbet.txt')
#wallstreetbets_data = file.read(encoding='uft-8')

def main():
    pass

def plot_raw_data():
    st.subheader("""Daily **opening and closing price** for  Gamestop """)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Closing Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Opening Price'))
    fig.layout.update(title_text="Stock History for Gamestop", xaxis_rangeslider_visible=True, width=1000, height=900)
    fig.update_yaxes(dtick=0.3) # change size of the y-axis steps 
    
    st.plotly_chart(fig)

#______________________________________________________________________________________________________________________________________________________________________________________#
# NLP Part
def preprocessing():


    #open text file you want to analyze
    f = open('wallstreetbet.txt', 'r', encoding='utf8')
    raw = f.read()

    #tokenize by words and make into nltk text
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    return text

def get_cleared_text(text):
    cleared = filter_punctuation(text)
    cleared = filter_stopwords(cleared)

    return cleared


def dispersion_plot_vanilla(nltk_text):
    words = ["good", "bad", "buy", "sell"]
    plt.ion()
    dispersion_plot(nltk_text, words)
    plt.ioff()
    plt.savefig('dispersion_plot.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    st.title("""**Dispersion Plot**""")
    st.subheader("""**helpful to determine the location of a word in a sequence of text sentences.**""")
    st.image('dispersion_plot.png')



def dispersion_plotting(nltk_text):
    #words to filter for
    words = ["good", "bad", "buy", "sell"]

    #step 1: iterate over all of the nltk_text and
    #step 2: compare with the given words and then save offset
    points = [(x, y) for x in range(len(nltk_text))
              for y in range(len(words)) if nltk_text[x] == words[y]]

    #zip aggregates 0 or more iteratables into a tuple
    if points:
        x, y = zip(*points)
    else:
        x = y = ()

    plt.plot(x, y, "rx", scalex=1)
    plt.yticks(range(len(words)), words, color="g")
    plt.xticks()
    plt.ylim(-1, len(words))
    plt.title("Lexical Dispersion Plot")
    plt.xlabel("Word Offset")
    plt.savefig('disp_plot')
    plt.show()


def filter_punctuation(nltk_text):
    text = [word.lower() for word in nltk_text if word.isalpha()]
    return text

def filter_stopwords(list_to_be_cleared):
    stop_words = set(stopwords.words("english"))

    # empty list für das Ergebnis
    filtered_list = []

    for word in list_to_be_cleared:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    
    return filtered_list



def frequency_dist_dict(cleared_list):
    frequency_dist = FreqDist(cleared_list)

    #dictionaries cant be sorted so its getting sorted as a list and then cast back into a dict
    od = dict(sorted(frequency_dist.items(), key=lambda item: item[1], reverse=True))

    #dictionaries cant be sliced so conversion to list in order to slice the first 20 instances (can be changed)
    first_twenty = list(od.items())[:20]

    #conversion back into a dict
    final_dict = {}
    final_dict.update(first_twenty)
    
    keyList = list(final_dict.keys())
    valueList = list(final_dict.values())
    
    st.title("""**Frequency Analysis**""")
    st.subheader("""**Dispersion Plot**""")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=keyList,y= valueList,name='Frequency of occurring words '))
    st.plotly_chart(fig)

    return final_dict   
    

def collocations(cleared_list):

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in cleared_list]
    new_text = nltk.Text(lemmatized_words)
    new_text.collocations()



def sentiment_anaylsis(cleaned_list):
    sia = SentimentIntensityAnalyzer()
    number = 0
    pos = 0
    neg = 0
    neu = 0
    with open('wallstreetbetsentiment.txt', 'w', encoding='utf-8') as f:
        for string in cleaned_list:
            s = sia.polarity_scores(string)
            pos = pos + s['pos']
            neg = neg + s['neg']
            neu = neu + s['neu']
            number = number +1

        pos_avg = pos / number
        neg_avg = neg / number
        neu_avg = neu/number
        #print("Positive = " + str(pos_avg))
        #print("Negative = " + str(neg_avg))
        #print("Neutral =  " + str(neu_avg))
        
        st.title(""" **Sentiment Analysis**""")
        st.subheader('Sentiment analysis can help you determine the ratio of positive to negative engagements about a specific topic. You can analyze bodies of text, such as comments, tweets, and product reviews, to obtain insights from your audience.')
        st.write("Positive = " + str(pos_avg))
        st.write("Negative = " + str(neg_avg))
        st.write("Neutral =  " + str(neu_avg))

        if(pos_avg > neg_avg >neu_avg):
            st.write("Overall Sentiment is positive")
        elif(neg_avg> pos_avg > neg_avg):
            st.write("Overall Sentiment is negative")
        elif(neu_avg > neg_avg > pos_avg):
            st.write("Overall Sentiment is neutral")
        else:
            st.write("No overall sentiment available")
            

#______________________________________________________________________________________________________________________________________________________________________________________#


def main():
    plot_raw_data()
    text = preprocessing()
    cleared = get_cleared_text(text)
    collocations(cleared)
    frequency_dist_dict(cleared)
    dispersion_plot_vanilla(text)
    sentiment_anaylsis(cleared)

if __name__ == "__main__":
    main()
