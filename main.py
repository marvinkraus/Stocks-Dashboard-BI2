import yfinance as yf
import streamlit as st
from dateutil.relativedelta import relativedelta
from datetime import date
from plotly import graph_objs as go 
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
from wordcloud import WordCloud
from nltk.corpus.reader import wordnet

#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('vader_lexicon')
#nltk.download('averaged_perceptron_tagger')
#nltk.download("maxent_ne_chunker")
#nltk.download("words")
#nltk.download("Punkt")

st.set_page_config(layout="wide",page_title='STOCK WEB APP', page_icon='🤑')
st.set_option('deprecation.showPyplotGlobalUse', False)



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Local CSS Sheet
local_css("style.css")
padding = 200

@st.cache(suppress_st_warning=True)
def load_data(ticker):
    START = date.today() - relativedelta(days=10)
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)  
    data.reset_index(inplace=True)
    return data


# Global variables
#webApp_title = '<p style="font-family:helvetica; color:white; font-size: 80px;"> <center>🤑 Stock Web-App 🤑</center></p>'
webApp_title =  '<h1 style="text-align: center;"><span style="color: #ffffff;"><strong>🤑 Stock Web-App 🤑</strong></span></h1>'
st.markdown(webApp_title, unsafe_allow_html=True)
data = load_data('TSLA')
#file = open('wallstreetbet.txt')
#wallstreetbets_data = file.read(encoding='uft-8')

def main():
    pass


def plot_raw_data():
    st.subheader("""Daily **opening and closing price** for  Tesla """)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Closing Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Opening Price'))
    fig.layout.update(xaxis_rangeslider_visible=True, width=1200, height=900,plot_bgcolor = '#484d6d', paper_bgcolor ='#282A36')
    fig.update_yaxes(dtick=5) # change size of the y-axis steps 
    fig.update_xaxes(showline=True, linewidth=3, linecolor='white', gridcolor='white')
    fig.update_yaxes(showline=True, linewidth=3, linecolor='white', gridcolor='white')
    fig.layout.xaxis.color = 'white'
    fig.layout.yaxis.color = 'white'
    fig.update_layout(legend_font_color="white")
    fig.update_traces(line_width=3)
    #fig.update_layout(paper_bgcolor="white")
    st.plotly_chart(fig,use_container_width = True)

#______________________________________________________________________________________________________________________________________________________________________________________#
# NLP Part

def preprocessing():
    #open text file you want to analyze
    f = open('tesla_news.txt', 'r', encoding='utf8') # tesla.txt und die ganzen Analysen machen für das dashboard und dann in kapitel in 9 die ergebnisse einfügen 
    raw = f.read()

    #tokenize by words and make into nltk text
    tokens = nltk.word_tokenize(raw)
    text = nltk.Text(tokens)

    return text


def get_cleared_text(text):
    cleared = filter_punctuation(text)
    cleared = filter_stopwords(cleared)
    cleared = lemmantize_text(cleared)

    return cleared


def dispersion_plot_vanilla(nltk_text):
    st.subheader("Dispersion Plot")
    col1, col2, col3 = st.columns([1,1.5,1])

    with col1:
        st.write("")

    with col2:
        words = ["good", "bad", "buy", "sell"]
        plt.ion()
        dispersion_plot(nltk_text, words)
        plt.ioff()
        plt.savefig('dispersion_plot.png')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        st.image('dispersion_plot.png')
        st.title('')
        st.title('')
        st.title('')
        


    with col3:
        st.write("")
   


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
    plt.show()
    st.title('')
    st.title('')
    st.title('')


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
    
    st.subheader("Frequency Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=keyList,y= valueList,name='Frequency of occurring words '))
    fig.layout.update(height=900,font=dict(
        family="Helvetica",
        size=17,
        color="black",)
    )
    fig.layout.update(plot_bgcolor = '#fafafb', paper_bgcolor ='#fafafb')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black')
    fig.layout.xaxis.color = 'black'
    fig.layout.yaxis.color = 'black'
    fig.update_layout(legend_font_color="white")
    fig.update_xaxes(tickangle=55)

    st.plotly_chart(fig,use_container_width = True)
    st.title('')
    st.title('')
    st.title('')
    # hier eine Wordcloud erstellen --> ausprobieren 

    return final_dict   
 

def show_wordcloud(cleared):
    #f = open('wallstreetbet.txt', 'r', encoding='utf8')
    #raw = f.read()
    data = " ".join(cleared)
    wordcloud = WordCloud(
        background_color='black',
        max_words=50,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show() 
    st.subheader('Frequency Analysis - Wordcloud')
    st.pyplot()
    plt.savefig('wordcloud.png',
            dpi = 300)
    st.title('')
    st.title('')
    st.title('')


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
        
        st.subheader('Sentiment Analysis')
        labels = 'Positive', 'Negative', 'Neutral'
        sizes = [pos_avg,neg_avg,neu_avg]
        explode = (0, 0, 0.2)  

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)


#maps nltk Part of Speech tags to wordnet tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


#lemmantizes a text
def lemmantize_text(cleared_list):

    cl = []
    tags = nltk.pos_tag(cleared_list)
    lemmatizer = WordNetLemmatizer()
    for word, pos in tags:
        if(get_wordnet_pos(pos) is not None):
            cl.append(lemmatizer.lemmatize(word, get_wordnet_pos(pos)))
        else:
            cl.append(word)
    return cl     

#______________________________________________________________________________________________________________________________________________________________________________________#

def main():
    plot_raw_data()
    text = preprocessing()
    cleared = get_cleared_text(text)
    webTitle_NLP =  '<h2 style="text-align: center;"><span style="color: #ffffff;"><strong>Overall sentiment in reddit based on comments in the last day</strong></span></h2>'
    st.markdown(webTitle_NLP, unsafe_allow_html=True)
    st.title('')
    frequency_dist_dict(cleared)
    dispersion_plot_vanilla(text)
    show_wordcloud(cleared)
    #collocations(cleared)
    sentiment_anaylsis(cleared)
    
    

if __name__ == "__main__":
    main()
