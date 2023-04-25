import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('vader_lexicon')

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
df = pd.read_csv('Twitter_Data.csv')

# Create a function to analyze sentiment using TextBlob
def analyze_sentiment_textblob(clean_text):
    blob = TextBlob(clean_text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Create a function to analyze sentiment using Vader
def analyze_sentiment_vader(clean_text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(clean_text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Create a function to search for tweets that contain a specific keyword
def search_tweets(keyword, num_tweets):
    df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')

    if keyword:
        results = df[df['clean_text'].str.contains(keyword, na=False)]
    else:
        results = df

    if num_tweets > results.shape[0]:
        num_tweets = results.shape[0]

    results = results.sample(n=num_tweets)

    return results


# Create a function to display a pie chart or a bar chart of the sentiment distribution
def display_chart(data, chart_type):
    if 'sentiment_textblob' in data.columns:
        data = data[['clean_text', 'sentiment_textblob']].rename(columns={'sentiment_textblob': 'category'})
    elif 'sentiment_vader' in data.columns:
        data = data[['clean_text', 'sentiment_vader']].rename(columns={'sentiment_vader': 'category'})
    else:
        st.write('Error: input DataFrame must have a sentiment column')
        return

    if chart_type == 'Pie Chart':
        chart_data = data['category'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
        st.pyplot(fig1)
       
    elif chart_type == 'Bar Chart':
        chart_data = data['category'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.bar(chart_data.index, chart_data)
        st.pyplot(fig2)


# Create a streamlit app
def main():
    st.title("Twitter Sentiment Analysis")
    st.write("Enter a keyword to search for tweets:")
    keyword = st.text_input(label="Keyword", value="COVID")
    num_tweets = st.slider("Select the number of tweets you want to see:", min_value=1, max_value=100, value=10)
    chart_type = st.selectbox("Select the type of chart to display:", ["Pie Chart", "Bar Chart"])

    results = search_tweets(keyword, num_tweets)
    st.write(results[['clean_text']])

    if not results.empty:
        # Analyze sentiment using TextBlob
        results['sentiment_textblob'] = results['clean_text'].apply(analyze_sentiment_textblob)

    # Analyze sentiment using Vader
        results['sentiment_vader'] = results['clean_text'].apply(analyze_sentiment_vader)

    st.write("Sentiment Distribution:")
    display_chart(results, chart_type)

    st.write("Sentiment Analysis using TextBlob:")
    results['sentiment_textblob'] = results['clean_text'].apply(analyze_sentiment_textblob)
    st.write(results[['clean_text', 'sentiment_textblob']])

    st.write("Sentiment Analysis using Vader:")
    results['sentiment_vader'] = results['clean_text'].apply(analyze_sentiment_vader)
    st.write(results[['clean_text', 'sentiment_vader']])

    display_chart(results, chart_type)
    
    if chart_type == 'Bar Chart':
    
        st.write("Word Cloud of Most Frequent Words:")
        text = " ".join(review for review in results.clean_text)
        stopwords = set(STOPWORDS)
        stopwords.update([keyword])
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()    

if __name__ == '__main__':
    main()
















# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt

# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from textblob import TextBlob

# nltk.download('vader_lexicon')

# st.set_option('deprecation.showPyplotGlobalUse', False)

# # Load the dataset
# df = pd.read_csv('Twitter_Data.csv')

# # Create a function to analyze sentiment using TextBlob
# def analyze_sentiment_textblob(clean_text):
#     blob = TextBlob(clean_text)
#     polarity = blob.sentiment.polarity
#     if polarity > 0:
#         return 'Positive'
#     elif polarity < 0:
#         return 'Negative'
#     else:
#         return 'Neutral'

# # Create a function to analyze sentiment using Vader
# def analyze_sentiment_vader(clean_text):
#     analyzer = SentimentIntensityAnalyzer()
#     scores = analyzer.polarity_scores(clean_text)
#     if scores['compound'] >= 0.05:
#         return 'Positive'
#     elif scores['compound'] <= -0.05:
#         return 'Negative'
#     else:
#         return 'Neutral'

# # Create a function to search for tweets that contain a specific keyword
# def search_tweets(keyword, num_tweets):
#     df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')

#     if keyword:
#         results = df[df['clean_text'].str.contains(keyword, na=False)]
#     else:
#         results = df

#     if num_tweets > results.shape[0]:
#         num_tweets = results.shape[0]

#     results = results.sample(n=num_tweets)

#     return results


# # Create a function to display a pie chart or a bar chart of the sentiment distribution
# def display_chart(data, chart_type):
#     if 'sentiment_textblob' in data.columns:
#         data = data[['clean_text', 'sentiment_textblob']].rename(columns={'sentiment_textblob': 'category'})
#     elif 'sentiment_vader' in data.columns:
#         data = data[['clean_text', 'sentiment_vader']].rename(columns={'sentiment_vader': 'category'})
#     else:
#         st.write('Error: input DataFrame must have a sentiment column')
#         return

#     if chart_type == 'Pie Chart':
#         chart_data = data['category'].value_counts()
#         fig1, ax1 = plt.subplots()
#         ax1.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
#         st.pyplot(fig1)
#     elif chart_type == 'Bar Chart':
#         chart_data = data['category'].value_counts()
#         fig2, ax2 = plt.subplots()
#         ax2.bar(chart_data.index, chart_data)
#         st.pyplot(fig2)


# # Create a streamlit app
# def main():
#     st.title("Twitter Sentiment Analysis")
#     st.write("Enter a keyword to search for tweets:")
#     keyword = st.text_input(label="Keyword", value="COVID")
#     num_tweets = st.slider("Select the number of tweets you want to see:", min_value=1, max_value=100, value=10)
#     chart_type = st.selectbox("Select the type of chart to display:", ["Pie Chart", "Bar Chart"])

#     results = search_tweets(keyword, num_tweets)
#     st.write(results)

#     if not results.empty:
#         # Analyze sentiment using TextBlob and Vader
#         results['sentiment_textblob'] = results['clean_text']
# if __name__ == '__main__':
#     main()
# # ---------------------------------------------------
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import plotly.express as px
# # from plotly.subplots import make_subplots
# # import plotly.graph_objects as go
# # from wordcloud import WordCloud, STOPWORDS
# # import matplotlib.pyplot as plt

# # import nltk
# # from nltk.sentiment.vader import SentimentIntensityAnalyzer
# # from textblob import TextBlob

# # nltk.download('vader_lexicon')

# # st.set_option('deprecation.showPyplotGlobalUse', False)

# # # Load the dataset
# # df = pd.read_csv('Twitter_Data.csv')

# # # Create a function to analyze sentiment using TextBlob
# # def analyze_sentiment_textblob(clean_text):
# #     blob = TextBlob(clean_text)
# #     polarity = blob.sentiment.polarity
# #     if polarity > 0:
# #         return 'Positive'
# #     elif polarity < 0:
# #         return 'Negative'
# #     else:
# #         return 'Neutral'

# # # Create a function to analyze sentiment using Vader
# # def analyze_sentiment_vader(clean_text):
# #     analyzer = SentimentIntensityAnalyzer()
# #     scores = analyzer.polarity_scores(clean_text)
# #     if scores['compound'] >= 0.05:
# #         return 'Positive'
# #     elif scores['compound'] <= -0.05:
# #         return 'Negative'
# #     else:
# #         return 'Neutral'

# # # Create a function to search for tweets that contain a specific keyword
# # def search_tweets(keyword, num_tweets):
# #     df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')

# #     if keyword:
# #         results = df[df['clean_text'].str.contains(keyword, na=False)]
# #     else:
# #         results = df

# #     if num_tweets > results.shape[0]:
# #         num_tweets = results.shape[0]

# #     results = results.sample(n=num_tweets)

# #     return results


# # # Create a function to display a pie chart or a bar chart of the sentiment distribution
# # def display_chart(data, chart_type):
# #     if 'sentiment_textblob' in data.columns:
# #         data = data[['clean_text', 'sentiment_textblob']].rename(columns={'sentiment_textblob': 'category'})
# #     elif 'sentiment_vader' in data.columns:
# #         data = data[['clean_text', 'sentiment_vader']].rename(columns={'sentiment_vader': 'category'})
# #     else:
# #         st.write('Error: input DataFrame must have a sentiment column')
# #         return

# #     if chart_type == 'Pie Chart':
# #         chart_data = data['category'].value_counts()
# #         fig1, ax1 = plt.subplots()
# #         ax1.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
# #         st.pyplot(fig1)
# #     elif chart_type == 'Bar Chart':
# #         chart_data = data['category'].value_counts()
# #         fig2, ax2 = plt.subplots()
# #         ax2.bar(chart_data.index, chart_data)
# #         st.pyplot(fig2)


# # # Create a streamlit app
# # def main():
# #     st.title("Twitter Sentiment Analysis")
# #     st.write("Enter a keyword to search for tweets:")
# #     keyword = st.text_input(label="Keyword", value="COVID")
# #     num_tweets = st.slider("Select the number of tweets you want to see:", min_value=1, max_value=100, value=10)
# #     chart_type = st.selectbox("Select the type of chart to display:", ["Pie Chart", "Bar Chart"])

# #     results = search_tweets(keyword, num_tweets)
# #     st.write(results)

# #     if not results.empty:
# #         # Analyze sentiment using TextBlob and Vader
# #         results['sentiment_textblob'] = results['clean_text'].apply(analyze_sentiment_textblob)
# #         results['sentiment_vader'] = results['clean_text'].apply(analyze_sentiment_vader)
        
# #         # Display comparison of sentiment analysis using TextBlob and Vader
# #         st.write('Comparison of sentiment analysis using TextBlob and Vader:')
# #         fig3, ax3 = plt.subplots()
# #         ax3.hist([results['sentiment_textblob'], results['sentiment_vader']], color=['orange', 'green'], alpha=0.5, label=['TextBlob', 'Vader'])
# #         ax3.legend()
# #         st.pyplot(fig3)
        
# #         # Display sentiment distribution for each method
# #     st.write('Sentiment distribution using TextBlob:')
# #     display_chart(results[['clean_text', 'sentiment_textblob']].rename(columns={'sentiment_textblob': 'category'}), chart_type)
    
# #     st.write('Sentiment distribution using Vader:')
# #     display_chart(results[['clean_text', 'sentiment_vader']].rename(columns={'sentiment_vader': 'category'}), chart_type)
    
# #     # Display word cloud of most common words
# #     st.write('Word cloud of most common words:')
# #     stopwords = set(STOPWORDS)
# #     wordcloud = WordCloud(width = 800, height = 800, 
# #                     background_color ='white', 
# #                     stopwords = stopwords, 
# #                     min_font_size = 10).generate(' '.join(results['clean_text']))
# #     fig4, ax4 = plt.subplots()
# #     ax4.imshow(wordcloud)
# #     ax4.axis("off")
# #     st.pyplot(fig4)
# # if __name__ == '__main__':
# #     main()
# # # ----------------------------------------------------------------------------------------------------------------------
# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import plotly.express as px
# # # from plotly.subplots import make_subplots
# # # import plotly.graph_objects as go
# # # from wordcloud import WordCloud, STOPWORDS
# # # import matplotlib.pyplot as plt

# # # import nltk
# # # from nltk.sentiment.vader import SentimentIntensityAnalyzer

# # # nltk.download('vader_lexicon')

# # # st.set_option('deprecation.showPyplotGlobalUse', False)

# # # # Load the dataset
# # # df = pd.read_csv('Twitter_Data.csv')

# # # # Create a function to analyze sentiment
# # # def analyze_sentiment(clean_text):
# # #     analyzer = SentimentIntensityAnalyzer()
# # #     scores = analyzer.polarity_scores(clean_text)
# # #     if scores['compound'] >= 0.05:
# # #         return 'Positive'
# # #     elif scores['compound'] <= -0.05:
# # #         return 'Negative'
# # #     else:
# # #         return 'Neutral'

# # # # Create a function to search for tweets that contain a specific keyword

# # # def search_tweets(keyword, num_tweets):
# # #     df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')
    
# # #     if keyword:
# # #         results = df[df['clean_text'].str.contains(keyword, na=False)].sample(n=num_tweets)
# # #     else:
# # #         results = df.sample(n=num_tweets)
    
# # #     return results


# # # # Create a function to display a pie chart or a bar chart of the sentiment distribution
# # # def display_chart(data, chart_type):
# # #     if chart_type == 'Pie Chart':
# # #         chart_data = data['category'].value_counts()
# # #         fig1, ax1 = plt.subplots()
# # #         ax1.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
# # #         st.pyplot(fig1)
# # #     elif chart_type == 'Bar Chart':
# # #         chart_data = data['category'].value_counts()
# # #         fig2, ax2 = plt.subplots()
# # #         ax2.bar(chart_data.index, chart_data)
# # #         st.pyplot(fig2)

# # # # Create a streamlit app
# # # def main():
# # #     st.title("Twitter Sentiment Analysis")
# # #     st.write("Enter a keyword to search for tweets:")
# # #     keyword = st.text_input(label="Keyword", value="COVID")
# # #     num_tweets = st.slider("Select the number of tweets you want to see:", min_value=1, max_value=100, value=10)
# # #     chart_type = st.selectbox("Select the type of chart to display:", ["Pie Chart", "Bar Chart"])

# # #     results = search_tweets(keyword, num_tweets)
# # #     st.write(results)

# # #     if not results.empty:
# # #         results['sentiment'] = results['clean_text'].apply(analyze_sentiment)
# # #         st.write(f"Sentiment distribution for keyword: {keyword}")
# # #         display_chart(results, chart_type)

# # # if __name__ == '__main__':
# # #     main()


# # # DATA_URL = (
# # #     "Tweets.csv"
# # # )

# # # st.title("Twitter Sentiment Analysis")
# # # st.sidebar.title("Sentiment Analysis of Tweets")
# # # st.markdown("This application is created to analyse twitter sentiments")
# # # st.sidebar.markdown("This applicationcreated to analyse twitter sentiments")

# # # @st.cache(persist=True)
# # # def load_data():
# # #     data = pd.read_csv(DATA_URL)
# # #     data['tweet_created'] = pd.to_datetime(data['tweet_created'])
# # #     return data

# # # data = load_data()

# # # st.sidebar.subheader("Show random tweet")
# # # random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
# # # st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["clean_text"]].sample(n=1).iat[0, 0])

# # # st.sidebar.markdown("### Number of tweets by sentiment")
# # # select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
# # # sentiment_count = data['airline_sentiment'].value_counts()
# # # sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
# # # if not st.sidebar.checkbox("Hide", True):
# # #     st.markdown("### Number of tweets by sentiment")
# # #     if select == 'Bar plot':
# # #         fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
# # #         st.plotly_chart(fig)
# # #     else:
# # #         fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
# # #         st.plotly_chart(fig)

# # # st.sidebar.subheader("When and where are users tweeting from?")
# # # hour = st.sidebar.slider("Hour to look at", 0, 23)
# # # modified_data = data[data['tweet_created'].dt.hour == hour]
# # # if not st.sidebar.checkbox("Close", True, key='1'):
# # #     st.markdown("### Tweet locations based on time of day")
# # #     st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
# # #     st.map(modified_data)
# # #     if st.sidebar.checkbox("Show raw data", False):
# # #         st.write(modified_data)


# # # st.sidebar.subheader("Total number of tweets for each airline")
# # # each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
# # # airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
# # # airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
# # # if not st.sidebar.checkbox("Close", True, key='2'):
# # #     if each_airline == 'Bar plot':
# # #         st.subheader("Total number of tweets for each airline")
# # #         fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
# # #         st.plotly_chart(fig_1)
# # #     if each_airline == 'Pie chart':
# # #         st.subheader("Total number of tweets for each airline")
# # #         fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
# # #         st.plotly_chart(fig_2)


# # # @st.cache(persist=True)
# # # def plot_sentiment(airline):
# # #     df = data[data['airline']==airline]
# # #     count = df['airline_sentiment'].value_counts()
# # #     count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
# # #     return count


# # # st.sidebar.subheader("Breakdown airline by sentiment")
# # # choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
# # # if len(choice) > 0:
# # #     st.subheader("Breakdown airline by sentiment")
# # #     breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
# # #     fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
# # #     if breakdown_type == 'Bar plot':
# # #         for i in range(1):
# # #             for j in range(len(choice)):
# # #                 fig_3.add_trace(
# # #                     go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
# # #                     row=i+1, col=j+1
# # #                 )
# # #         fig_3.update_layout(height=600, width=800)
# # #         st.plotly_chart(fig_3)
# # #     else:
# # #         fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
# # #         for i in range(1):
# # #             for j in range(len(choice)):
# # #                 fig_3.add_trace(
# # #                     go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
# # #                     i+1, j+1
# # #                 )
# # #         fig_3.update_layout(height=600, width=800)
# # #         st.plotly_chart(fig_3)
# # # st.sidebar.subheader("Breakdown airline by sentiment")
# # # choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key=0)
# # # if len(choice) > 0:
# # #     choice_data = data[data.airline.isin(choice)]
# # #     fig_0 = px.histogram(
# # #                         choice_data, x='airline', y='airline_sentiment',
# # #                          histfunc='count', color='airline_sentiment',
# # #                          facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
# # #                           height=600, width=800)
# # #     st.plotly_chart(fig_0)

# # # st.sidebar.header("Word Cloud")
# # # word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
# # # if not st.sidebar.checkbox("Close", True, key='3'):
# # #     st.subheader('Word cloud for %s sentiment' % (word_sentiment))
# # #     df = data[data['airline_sentiment']==word_sentiment]
# # #     words = ' '.join(df['text'])
# # #     processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
# # #     wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
# # #     plt.imshow(wordcloud)
# # #     plt.xticks([])
# # #     plt.yticks([])
# # #     st.pyplot()

# # # -------------------------------------------------------------------------------------------------------------
# # # import streamlit as st
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # st.set_option('deprecation.showPyplotGlobalUse', False)

# # # # Load the dataset
# # # df = pd.read_csv('Twitter_Data.csv')

# # # # Create a function to search for tweets that contain a specific keyword
# # # def search_tweets(keyword, num_tweets):
# # #     df = pd.read_csv('Twitter_Data.csv', encoding='ISO-8859-1')
    
# # #     if keyword:
# # #         results = df[df['clean_text'].str.contains(keyword, na=False)].head(num_tweets)
# # #     else:
# # #         results = df.head(num_tweets)
    
# # #     return results


# # # # Create a function to display a pie chart or a bar chart of the sentiment distribution
# # # def display_chart(data, chart_type):
# # #     if chart_type == 'Pie Chart':
# # #         chart_data = data['category'].value_counts()
# # #         fig1, ax1 = plt.subplots()
# # #         ax1.pie(chart_data, labels=chart_data.index, autopct='%1.1f%%')
# # #         st.pyplot(fig1)
# # #     elif chart_type == 'Bar Chart':
# # #         chart_data = data['category'].value_counts()
# # #         fig2, ax2 = plt.subplots()
# # #         ax2.bar(chart_data.index, chart_data)
# # #         st.pyplot(fig2)

# # # # Create a streamlit app
# # # def main():
# # #     st.title("Twitter Sentiment Analysis")
# # #     st.write("Enter a keyword to search for tweets:")
# # #     keyword = st.text_input(label="Keyword", value="COVID")
# # #     num_tweets = st.slider("Select the number of tweets you want to see:", min_value=1, max_value=100, value=10)
# # #     chart_type = st.selectbox("Select the type of chart to display:", ["Pie Chart", "Bar Chart"])

# # #     results = search_tweets(keyword, num_tweets)
# # #     st.write(results)

# # #     if not results.empty:
# # #         st.write(f"Sentiment distribution for keyword: {keyword}")
# # #         display_chart(results, chart_type)

# # # if __name__ == '__main__':
# # #     main()
# # # ---------------------------------------------------------------------------------------------------------------------------------

