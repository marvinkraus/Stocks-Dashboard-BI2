import yfinance as yf
import streamlit as st
from dateutil.relativedelta import relativedelta

from datetime import date
from plotly import graph_objs as go  # for interactive graphs
import pandas as pd


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


def main():
    plot_raw_data()

if __name__ == "__main__":
    main()