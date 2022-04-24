import cv2
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
#from sqlalchemy import create_engine
#from config import DBConfig
import datetime
#import tweepy as tw
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.express as px

from plotly.subplots import make_subplots

# import plotly.graph_objects as go from wordcloud


#-------------------- #utility Functions --------------------
def sentiment_label(sentiment):
    sentiment_label = ""
    if  -0.2 < sentiment <= 0.2: sentiment_label = "Neutral"
    elif sentiment <= -0.2: sentiment_label = "Negative"
    elif sentiment > 0.2: sentiment_label = "Positive"
    return sentiment_label
#
# @st.cache(allow_output_mutation=True, show_spinner=False)
# def get_con():
#     return create_engine('postgresql+psycopg2://DB_USER:DB_PWORD@localhost/tweet'.format(DBConfig.USER, DBConfig.PWORD, DBConfig.HOST),
#                          convert_unicode=True)


# @st.cache(allow_output_mutation=True, show_spinner=False, ttl=5*60)
# def get_data():
#     timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
#     df = pd.read_sql_table('tweets', get_con())
#     df['Sentiment_label']= df['sentiment'].apply(sentiment_label)
#     df = df.rename(columns={'body': 'Tweet', 'tweet_date': 'Timestamp',
#                             'followers': 'Followers', 'sentiment': 'Sentiment',
#                             'keyword': 'Subject'})
#     return df, timestamp


@st.cache(show_spinner=False)
def filter_by_date(df, start_date, end_date):
    df_filtered = df.loc[(df.Timestamp.dt.date >= start_date) & (df.Timestamp.dt.date <= end_date)]
    return df_filtered


@st.cache(show_spinner=False)
def filter_by_subject(df, subjects):
    return df[df.Subject.isin(subjects)]


@st.cache(show_spinner=False)
def count_plot_data(df, freq):
    plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).id.count().unstack(level=0, fill_value=0)
    plot_df.index.rename('Date', inplace=True)
    plot_df = plot_df.rename_axis(None, axis='columns')
    return plot_df


@st.cache(show_spinner=False)
def sentiment_plot_data(df, freq):
    plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).Sentiment.mean().unstack(level=0, fill_value=0)
    plot_df.index.rename('Date', inplace=True)
    plot_df = plot_df.rename_axis(None, axis='columns')
    return plot_df

#----------------------- Application ------------------------
#-------------------Influenctional Tweets vs Recent Tweets -------------------
st.set_page_config(layout="wide", page_title='Twitter Sentiment p')

timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
data =  pd.read_csv('TweetData')
data['Timestamp']= pd.to_datetime(data.Timestamp, format='%Y-%m-%d %H:%M:%S')
data['Sentiment_label']= data['Sentiment'].apply(sentiment_label)
st.header('Twitter Sentiment Retreival')
st.write('Total tweet count: {}'.format(data.shape[0]))
st.write('Data last loaded {}'.format(timestamp))

col1, col2 = st.columns(2)

date_options = data.Timestamp.dt.date.unique()
start_date_option = st.sidebar.selectbox('Select Start Date', date_options, index=0)
end_date_option = st.sidebar.selectbox('Select End Date', date_options, index=len(date_options)-1)

keywords = data.Subject.unique()
keyword_options = st.sidebar.multiselect(label='Subjects to Include:', options=keywords.tolist(), default=keywords.tolist())

data_subjects = data[data.Subject.isin(keyword_options)]
data_daily = filter_by_date(data_subjects, start_date_option, end_date_option)

top_daily_tweets = data_daily.sort_values(['Followers'], ascending=False).head(10)

col1.subheader('Influential Tweets')
col1.dataframe(top_daily_tweets[['Tweet', 'Timestamp', 'Followers', 'Subject']].reset_index(drop=True), 1000, 400)

col2.subheader('Recent Tweets')
col2.dataframe(data_daily[['Tweet', 'Timestamp', 'Followers', 'Subject']].sort_values(['Timestamp'], ascending=False).
               reset_index(drop=True).head(10))
#---------------------- Sentiment Plots ---------------------
plot_freq_options = {
    'Hourly': 'H',
    'Four Hourly': '4H',
    'Daily': 'D'
}
plot_freq_box = st.sidebar.selectbox(label='Plot Frequency:', options=list(plot_freq_options.keys()), index=0)
plot_freq = plot_freq_options[plot_freq_box]

col1.subheader('Tweet Volumes')
plotdata = count_plot_data(data_daily, plot_freq)
col1.line_chart(plotdata)

col2.subheader('Sentiment')
plotdata2 = sentiment_plot_data(data_daily, plot_freq)
col2.line_chart(plotdata2)

st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
subject = st.sidebar.selectbox('Subject Selection', keyword_options, key='1')
sentiment_count = data['Sentiment_label'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
if not st.sidebar.checkbox("Hide", True):
    st.markdown("### Number of tweets by sentiment")
    if select == 'Bar plot':
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.dataframe(data)
# ---------------------- Word Cloud --------------------
def cloud(image, text, max_word, max_font, random):
    stopwords = set(STOPWORDS)
    stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
                      'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
                      'put', 'seem', 'asked', 'made', 'half', 'much',
                      'certainly', 'might', 'came','https','co','Russia','Russian','Ukraine','t','democracy','Sudan'])

    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word, mask=image,
                   stopwords=stopwords, max_font_size=max_font, random_state=random)

    # generate word cloud
    wc.generate(text)

    # create coloring from image
    image_colors = ImageColorGenerator(image)

    # show the figure
    plt.figure(figsize=(100, 100))
    fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 2]})
    axes[0].imshow(wc, interpolation="bilinear")
    # recolor wordcloud and show
    # we could also give color_func=image_colors directly in the constructor
    #axes[1].imshow(image, cmap=plt.cm.gray, interpolation="bilinear")

    for ax in axes:
        ax.set_axis_off()
    st.pyplot(fig)

st.write("#### Text Summarization with a WordCloud")
max_word = 200
max_font = 50
random = st.sidebar.slider("Random State", 30, 100, 42 )
image = 'image.png'
text = data.Tweet.str.cat(sep=' ')
image = cv2.imread(image)
# st.image(image, width=100, use_column_width=True)
st.write("### Word cloud")
st.write(cloud(image, text, max_word, max_font, random))

#--------------------- Sentiment Counters -------------------------

st.sidebar.subheader("Total number of tweets for each Sentiment")
Each_subject = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
Subject_sentiment_count = data.groupby('Subject')['Sentiment_label'].count().sort_values(ascending=False)
Subject_sentiment_count = pd.DataFrame({'Subject':Subject_sentiment_count.index, 'Tweets':Subject_sentiment_count.values.flatten()})
if not st.sidebar.checkbox("Close", True, key='2'):
    if Each_subject == 'Bar plot':
        st.subheader("Total number of tweets for each Subject")
        fig_1 = px.bar(Subject_sentiment_count, x='Subject', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_1)
    if Each_subject == 'Pie chart':
        st.subheader("Total number of tweets for each Subject")
        fig_2 = px.pie(Subject_sentiment_count, values='Tweets', names='Subject')
        st.plotly_chart(fig_2)

st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('Positive', 'Neutral', 'Negative'))
st.sidebar.markdown(data.query("Sentiment_label == @random_tweet")[["Tweet"]].sample(n=1).iat[0, 0])

#----------------------- Locations -------------------------
locations = pd.DataFrame(pd.eval(data_daily[data_daily['location'].notnull()].location), columns=['lon', 'lat'])
st.map(locations)
