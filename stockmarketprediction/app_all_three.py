############ Importing All Python Libraries ############

import plotly.express as px
import plotly.io as pio
import tweepy
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objs as go
import statsmodels.api as smapi

from pandas.plotting import lag_plot
from textblob import TextBlob
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from plotly.offline import plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


############ WEB APP USING FLASK ############

from flask import Flask, render_template, request 
app = Flask(__name__) 

@app.route('/') 
def index():
    return render_template('home.html') 


@app.route('/calculate', methods=['POST']) 
def calculate():
    number = str(request.form['number']) 
    result = number


 ################   LINEAR REGRESSION ALGORITHM   ###############

    #Importing csv file
    apple = pd.read_csv(r'D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #Importing csv file
    apple = pd.read_csv('D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #First 5 rows
    apple.head()

    #Prints information about dataframe
    apple.info()

    #Converting the date column into datetime format
    apple['Date'] = pd.to_datetime(apple['Date'])

    #Printing the functions
    print(f'Dataframe contains stock prices between {apple.Date.min()} {apple.Date.max()}') 
    print(f'Total days = {(apple.Date.max()  - apple.Date.min()).days} days')

    #Description of the data in the DataFrame
    apple.describe()

    #Create a boxplot to visually check the outliers
    apple[['Open','High','Low','Close','Adj Close']].plot(kind='box')

    #Plot the graph using plotly libraries
    # Setting the layout for our plot
    layout = go.Layout(
        title=number+'-Stock Price Prediction By Linear Regression Model',
        xaxis=dict(
            title='Date',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ),

            tickformat='%Y-%m-%d'
            
        ),
        yaxis=dict(
            title='Stock Price',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    #Passing the datalist and the layout created to plot variable
    apple_data = [{'x':apple['Date'], 'y':apple['Close']}]
    plot2 = go.Figure(data=apple_data, layout=layout)

    # Building the regression model
    from sklearn.model_selection import train_test_split

    #For preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    #For model evaluation
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import r2_score

    #Split the data into train and test sets
    X = np.array(apple.index).reshape(-1,1)
    Y = apple['Close']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

    # Feature scaling
    scaler = StandardScaler().fit(X_train)

    #Import linear regression
    from sklearn.linear_model import LinearRegression

    #Creating a linear regression model
    lm = LinearRegression()
    lm.fit(X_train, Y_train)

    #Plot actual and predicted values for train dataset
    trace0 = go.Scatter(
        x = apple['Date'] [X_train.T[0]],
        y = Y_train,
        mode = 'markers',
        name = 'Actual'
    )
    trace1 = go.Scatter(
        x = apple ['Date'] [X_train.T[0]],
        y = lm.predict(X_train).T,
        mode = 'lines',
        name = 'Predicted'
    )
    apple_data = [trace0,trace1]
    layout.xaxis.title.text = 'Date'
    plot2 = go.Figure(data=apple_data, layout=layout)
    pio.write_image(plot2, 'static/LR.png')


 ############## LONG SHORT TERM MEMORY ALGORITHM ##############

    #Importing csv file
    apple = pd.read_csv(r'D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #Importing csv file
    apple = pd.read_csv('D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #First 5 rows
    apple.head()

    #Prints information about dataframe
    apple.info()

    #Remove rows that contain null values & #implicit conversion
    apple["Close"]=pd.to_numeric(apple.Close,errors='coerce')
    apple = apple.dropna()
    trainData = apple.iloc[:,4:5].values

    #Prints information about dataframe
    apple.info()

    #Rescale the data
    sc = MinMaxScaler(feature_range=(0,1))
    trainData = sc.fit_transform(trainData)
    trainData.shape

    #Dataflow training
    x_train = []
    y_train = []

    #60 : timestep / 1258 : length of the data
    for i in range (60,1000):
        x_train.append(trainData[i-60:i,0]) 
        y_train.append(trainData[i,0])

    x_train,y_train = np.array(x_train),np.array(y_train)

    #adding the batch size axis
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape

    #Building a ML model that contains four layers of LSTM network followed by dropout layer and using optimizer.
    model = Sequential()

    model.add(LSTM(units=100, return_sequences = True, input_shape =(x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences = False))
    model.add(Dropout(0.2))

    model.add(Dense(units =1))
    model.compile(optimizer='adam',loss="mean_squared_error")

    #Training the data using epochs and batch size
    hist = model.fit(x_train, y_train, epochs = 3, batch_size = 32, verbose=2)

    #Visualize the loss that occurs during the epoch
    plt.plot(hist.history['loss'])
    plt.title('Training model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')

    #Importing csv file
    appledata = pd.read_csv(r'D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #Importing csv file
    appledata = pd.read_csv('D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #Remove rows that contain null values & #implicit conversion
    appledata["Close"]=pd.to_numeric(appledata.Close,errors='coerce')
    appledata = appledata.dropna()
    appledata = appledata.iloc[:,4:5]
    y_test = appledata.iloc[60:,0:].values 

    #input array for the model
    #Converting x_test data into numpy array and printing it's shape
    inputClosing = appledata.iloc[:,0:].values 
    inputClosing_scaled = sc.transform(inputClosing)
    inputClosing_scaled.shape
    x_test = []
    length = len(appledata)
    timestep = 60
    for i in range(timestep,length):  
        x_test.append(inputClosing_scaled[i-timestep:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    x_test.shape

    #Predicting the model and passing the x_test data
    y_pred = model.predict(x_test)
    y_pred

    #Plot the data between actual and predicted prices
    predicted_price = sc.inverse_transform(y_pred)

    plt.clf()
    plt.cla()
    
    #Plot the graph and visualize the actual and predicted stock prices
    plt.plot(y_test, color = 'red', label = 'Actual Stock Price')
    plt.plot(predicted_price, color = 'blue', label = 'Predicted Stock Price')
    plt.title(number+'-Stock Price Prediction By LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.savefig('static/LS.png')
    plt.clf()
    plt.cla()


###################  ARIMA ALGORITHM ####################

    apple1 = pd.read_csv(r'D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #Importing csv file
    apple1 = pd.read_csv('D:\\xampp\\htdocs\\stockmarketprediction\\input\\'+ number +'.csv')

    #First 5 rows
    apple1.head()

    #Let's check if there is some cross-correlation in out data.
    plt.figure()
    lag_plot(apple1['Open'], lag=3)
    plt.title('Apple Stock - Autocorrelation plot with lag = 3')

    #Plotting the stock price evolution over time.
    plt.plot(apple1["Date"], apple1["Close"])
    plt.xticks(np.arange(0,1250, 300), apple1['Date'][0:1250:300])
    plt.title("Stock price over time")
    plt.xlabel("time")
    plt.ylabel("price")

    #Building the predictive ARIMA model
    #training(70%) and testing(30%)
    #default arima parameters p=4,d=1,q=0
    train_data, test_data = apple1[0:int(len(apple1)*0.7)], apple1[int(len(apple1)*0.7):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    print('Testing Mean Squared Error is {}'.format(MSE_error))

    #Plot the graph and visualize the actual and predicted stock prices
    test_set_range = apple1[int(len(apple)*0.7):].index
    plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
    plt.plot(test_set_range, test_data, color='red', label='Actual Price')
    plt.title(number+'-Stock Price Prediction By Arima Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.savefig('static/Arima.png')
    plt.clf()
    plt.cla()
    
    
#################  SENTIMENT ANALYSIS OF TWEETS  ##################

    #Importing the csv file
    twitter = pd.read_csv('D:\\xampp\\htdocs\\stockmarketprediction\\input\\twitter\\'+ number +'.csv')

    #Print the variable
    print(twitter)

    #First 5 rows
    twitter.head()

    #Cleaning up tweets using regular expression.
    def cleanUpTweet(txt):
        txt = re.sub(r'@[A-Za-z0-9_]+','',txt)
        txt = re.sub(r'#','',txt)
        txt = re.sub(r'RT : ','',txt)
        txt = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
        return txt

    #Cleaned up tweets
    twitter['Tweet']=twitter['Tweet'].apply(cleanUpTweet)

    #Using the textblob
    def getTextSubjectivity(txt):
        return TextBlob(txt).sentiment.subjectivity

    #Using the textblob
    def getTextPolarity(txt):
        return TextBlob(txt).sentiment.polarity

    #Getting sub and pol for tweet
    twitter['Subjectivity']=twitter['Tweet'].apply(getTextSubjectivity)
    twitter['Polarity']=twitter['Tweet'].apply(getTextPolarity)

    #First 10 rows
    twitter.head(10)

    #Removing the column
    twitter = twitter.drop(twitter[twitter['Tweet']==''].index)

    #First 10 rows
    twitter.head(10)

    #TextAnalysis
    def getTextAnalysis(a):
        if a<0:
            return "Negative"
        elif a==0:
            return "Neutral"
        else:
            return "Positive"

    #Getting the textanalysis
    twitter["Score"]=twitter['Polarity'].apply(getTextAnalysis)

    #First 10 rows
    twitter.head(10)

    #Calculating % of positive review
    positive=twitter[twitter['Score']=="Positive"]
    print(str(positive.shape[0]/(twitter.shape[0])*100)+"% of positive tweets")
    pos=positive.shape[0]/twitter.shape[0]*100

    #Calculating % of nrgative review
    negative=twitter[twitter['Score']=="Negative"]
    print(str(negative.shape[0]/(twitter.shape[0])*100)+"% of negative tweets")
    neg=negative.shape[0]/twitter.shape[0]*100

    #Calculating % of neutral review
    neutral=twitter[twitter['Score']=="Neutral"]
    print(str(neutral.shape[0]/(twitter.shape[0])*100)+"% of neutral tweets")
    neutrall=neutral.shape[0]/twitter.shape[0]*100

    #Show graph using pie-chart
    explode=(0,0.1,0)
    labels='Positive','Negative','Neutral'
    sizes=[pos,neg,neutrall]
    colors=['yellowgreen','lightcoral','gold']

    #Plotting the pie chart
    plt.pie(sizes,explode=explode,colors=colors,autopct='%1.1f%%',startangle=140)
    plt.legend(labels,loc=(-0.05,0.05),shadow = True)
    plt.axis('equal')
    plt.title(number+'-Sentiment Analysis Of Tweets')
    plt.savefig('static/tweet.png')

    return render_template("calculate.html") 


if __name__ == '__main__':
    app.run()