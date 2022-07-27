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
file = open('wallstreetbet.txt')
wallstreetbets_data = file.read()

def main():
    pass

def plot_raw_data():
    st.subheader("""Daily **opening and closing price** for  Gamestop """)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Closing Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Opening Price'))
    fig.layout.update(title_text="Stock History for Gamestop", xaxis_rangeslider_visible=True, width=800, height=800)
    fig.update_yaxes(dtick=0.5) # change size of the y-axis steps 
    
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


def wordcloud():
    
    st.title("Wordcloud")
    stop_words = set(stopwords.words("english"))
    concat_quotes = ' '.join([i for i in wallstreetbets_data.text_without_stopwords.astype(str)])

    t=stylecloud.gen_stylecloud(  # file_path='SJ-Speech.txt',

                                text=concat_quotes,

                                icon_name='fas fa-apple-alt',

                                background_color='black',

                                output_name='apple.png',

                                collocations=False,

                                custom_stopwords=stop_words)

    st.image(t)

#______________________________________________________________________________________________________________________________________________________________________________________#


def main():
    plot_raw_data()
    wordcloud()
    #st.write(get_cleared_text(text))
    

if __name__ == "__main__":
    main()