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

nltk.download('punkt')

# https://github.com/luigibr1/Streamlit-StockSearchWebApp/blob/master/web_app_v3.py
# https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py
# https://www.ritchieng.com/pandas-multi-criteria-filtering/

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Local CSS Sheet
local_css("style.css")

START = date.today() - relativedelta(days=7)
TODAY = date.today().strftime("%Y-%m-%d")


def load_data(ticker):
    data = yf.download(ticker, START, TODAY)  # returns a panda dataframe
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


def dispersion_plot(nltk_text):
    nltk_text.dispersion_plot(["good", "bad", "buy", "sell"])

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



def frequency_dist(cleared_list):
    frequencydist = FreqDist(cleared_list)
    frequencydist.plot(20, cumulative=True)

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
        print("Positive = " + str(pos_avg))
        print("Negative = " + str(neg_avg))
        print("Neutral =  " + str(neu_avg))


#______________________________________________________________________________________________________________________________________________________________________________________#


def main():
    plot_raw_data()
    text = preprocessing()
    cleared = get_cleared_text(text)
    collocations(cleared)
    
    #st.write(get_cleared_text(text))
    

if __name__ == "__main__":
    main()