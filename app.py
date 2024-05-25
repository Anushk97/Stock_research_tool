import streamlit as st
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import pandas as pd
from datetime import date, timedelta
import openai
import plotly.graph_objs as go
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


st.title('Stock Research help!')

# API_KEY = st.secrets["api"]["iex_key"]
API_BASE_URL = "https://cloud.iexapis.com/stable/"


def get_stock_data(symbol, time_range="5y"):
    params = {
        "token": "pk_53ca8663164549eaaed782b5f815961f"
    }

    response = requests.get(API_BASE_URL + f"stock/{symbol}/chart/{time_range}", params=params)
    data = response.json()

    if "error" in data:
        st.error(f"Error: {data['error']}")
        return None

    stock_data = pd.DataFrame(data)
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data.set_index("date", inplace=True)
    stock_data = stock_data[["open", "high", "low", "close", "volume"]]
    stock_data.columns = ["Open", "High", "Low", "Close", "Volume"]
    return stock_data

def calculate_price_difference(stock_data):
    latest_price = stock_data.iloc[-1]["Close"]
    previous_year_price = stock_data.iloc[-252]["Close"] if len(stock_data) > 252 else stock_data.iloc[0]["Close"]
    price_difference = latest_price - previous_year_price
    percentage_difference = (price_difference / previous_year_price) * 100
    return price_difference, percentage_difference

def app():
    popular_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM"]
    symbol = st.sidebar.selectbox("Select a stock symbol:", popular_symbols, index=None)
    st.sidebar.write('OR')
    symbol2 = st.sidebar.text_input("Enter a custom symbol:")
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    
    if symbol:
        stock = symbol
    else:
        stock = symbol2
    
    if stock:
        st.title("📈 Stock Dashboard")
        stock_data = get_stock_data(stock)

        if stock_data is not None:
            price_difference, percentage_difference = calculate_price_difference(stock_data)
            latest_close_price = stock_data.iloc[-1]["Close"]
            max_52_week_high = stock_data["High"].tail(252).max()
            min_52_week_low = stock_data["Low"].tail(252).min()
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Close Price", f"${latest_close_price:.2f}")
        with col2:
            st.metric("Price Difference (YoY)", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%")
        with col3:
            st.metric("52-Week High", f"${max_52_week_high:.2f}")
        with col4:
            st.metric("52-Week Low", f"${min_52_week_low:.2f}")

        st.subheader("Candlestick Chart")
        candlestick_chart = go.Figure(data=[go.Candlestick(x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'])])
        candlestick_chart.update_layout(title=f"{symbol} Candlestick Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(candlestick_chart, use_container_width=True)
    


# user_input = st.text_input('Enter stock ticker: ')
        st.title("🌱 Sentiment Analysis")
        
        news = {}

        try:
            url = f'https://finviz.com/quote.ashx?t={stock}&p=d'
            request = Request(url=url, headers={'user-agent': 'news_scraper'})
            response = urlopen(request)

            html = BeautifulSoup(response, features='html.parser')
            finviz_news_table = html.find(id='news-table')
            news[stock] = finviz_news_table

            # filter and store news_parsed
            news_parsed = []
            for stock, news_item in news.items():
                for row in news_item.findAll('tr'):
                    try:
                        headline = row.a.getText()
                        source = row.span.getText()
                        news_parsed.append([stock, headline])
                    except:
                        pass

            # convert to a dataframe for data analysis
            df = pd.DataFrame(news_parsed, columns=['Stock', 'Headline'])
            # df = df[:50]

            combine_text = ''.join(df['Headline'].tolist())
            # print(combine_text)

            # openai.api_key = st.secrets['api']['openai_key']
            openai.api_key = openai_api_key

            completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Please analyse the sentiment of the {combine_text}. Give advice like a financial advisor looking at the news in the current market. Keep the response full with good keywords. Don't include the keyword sentiment. Come up a final result of bullish or bearish."
                },

            ],
            )
            # print(completion.choices[0].message.content)

            response = completion.choices[0].message.content
            st.write(response)
        except:
            st.write('')
        
        st.title("💭 Word Cloud")
        wordcloud = WordCloud().generate(combine_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        
        st.pyplot(fig)


if __name__ == "__main__":
    app()