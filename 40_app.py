
# import libraries 
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly .express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# title 
# app_name="Stock Market Forecasting WebApp"
# st.title(app_name)
# st.subheader("This app is created to forecast the stock market price of the selected company")
app_name = "Stock Market Forecasting WebApp"

# Use HTML to style the title and subheader
st.markdown(
    f'<h1 style="color: #FF5733; text-align: center;">{app_name}</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<h2 style="color: #008080; text-align: center;">'
    'This app is created to forecast the stock market price of the selected company'
    '</h2>',
    unsafe_allow_html=True
)

# add an image from online resource 
st.image("https://images.moneycontrol.com/static-mcnews/2023/07/market_stocks_sensex_Nifty-4-770x433.jpg?impolicy=website&width=770&height=431")

# take input from user of app about the start and end date 

# slidebar 
st.sidebar.header("Select the parameters from below")

start_date =st.sidebar.date_input("Start Date",date(2020,1,1))
end_date=st.sidebar.date_input(" End Date",date(2020,12,31))
# add ticker symbol list
ticker_list=["AAPL","MSFT","GOOG","GOOGL","FB","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"]
ticker=st.sidebar.selectbox("Select the company",ticker_list)

# fetch the data from user inputs using yfinance library
data=yf.download(ticker,start=start_date,end=end_date)
#add date as a column to the dataframe(because data show on the base of index)
data.insert(0,"Date",data.index,True) # date.index show as column wise
# remove index 
data.reset_index(drop=True,inplace=True)
# add date below image 
st.write("Data from",start_date,"to",end_date)
# show data 
st.write(data)

# plotting the data
st.header("Data Vizualization")
st.subheader("Plot of the data")
st.write("**Note:** Select your specific date range on the side bar,or zoom in on the plot and select your specific column")
fig=px.line(data,x="Date",y=data.columns,title="Closing Prize of the Stock",width=1000,height=600)
st.plotly_chart(fig)

# add a selectbox to select the column from data 
column=st.selectbox("Select the column to used for forcasting",data.columns[1:])# why one use beacuse select the one column 
# subseting the data (select the specific column instead of more )
data=data[["Date",column]] # date as a column and print the column data 
st.write("Selected Data ") # just for header 
st.write(data)

# ADF test check stationary
st.header("Is data Stationary?")
st.write(adfuller(data[column])[1]<0.05)# adf.. is library and if data is stationary then used in arima model 

#Let decompose the test
st.header("Decompostion of the Data ")
decomposition=seasonal_decompose(data[column],model="additive",period=12)#import seasona_dec from state models and apply on selected column
st.write(decomposition.plot())

# we can also repersent the data into ploty 
st.write("##Plotting the decompostion in plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend,title="Trend",width=1000,height=400,labels={'x':"Date","y":"Price"}).update_traces(line_color="Blue"))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend,title="Seasonality",width=1000,height=400,labels={'x':"Date","y":"Price"}).update_traces(line_color="green"))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend,title="Residual",width=1000,height=400,labels={'x':"Date","y":"Price"}).update_traces(line_color="Red",line_dash="dot"))

# lets run the model 
# user input for three parameters of the model and seasonal order 
p=st.slider("Select the value of p",0,5,2)
d=st.slider("Select the valuer of d",0,5,1)
q=st.slider("Select the value od q",0,5,2)
seasonal_order=st.number_input("Select the value of seasonal p",0,24,12)
model=sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model=model.fit()
# print the model summary
st.header("Model Summary")
st.write(model.summary())
st.write("---")

# # Predict the future values(forecasting)
st.write("<p style='color:green; fot-size:50px; font-weight:bold;'>Forecasting the data</p>",unsafe_allow_html=True)

forecast_period=st.number_input("Select the number of days to forecast",1,365,10)# how many days select for forcast 
# predict the future values 
predictions =model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions=predictions.predicted_mean # prediction value mean only show 
#st.write(predictions)# use for as heading
# add index to the predictions
predictions.index=pd.date_range(start=end_date,periods=len(predictions),freq="D")
predictions=pd.DataFrame(predictions)
# remove the index value and show in column vise  
predictions.insert(0,"Date",predictions.index,True)
predictions.reset_index(drop=True,inplace=True)# use for remove the index from the dated and predicted results 
st.write("Predictions",predictions)
st.write("Actual Date",data)
st.write("---")

# plot the predicted data by using plotly 
fig=go.Figure()
# add actual data to the plot 
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode="lines",name="Actual",line=dict(color="blue")))
# add predicted data into the plot
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode="lines",name="Predicted",line=dict(color='red')))
# add the title on x axis and y axises 
fig.update_layout(title="Actual vs Predicted",xaxis_title="Date",yaxis_title="Price",width=1200,height=400)
# disploy the plot
st.plotly_chart(fig)
# st.header("Thanks you for using this app share with your friends")
# st.write("About the Author")
# st.header("Arbaz Fina ")
st.markdown('<h1 style="color:blue;">Thanks you for using this app share with your friends</h1>', unsafe_allow_html=True)
st.write("About the Author")
st.markdown('<h2 style="color:green;">Arbaz Fina</h2>', unsafe_allow_html=True)
# about us 
st.write("Connect with me on social media ")
## add social media images links 
linkedin_url="https://img.icons8.com/color/48/000000/linkedin.png"
github_url="https://img.icons8.com/color/48/000000/github.png"
youtube_url="https://img.icons8.com/color/48/000000/youtube-play.png"
twitter_url="https://img.icons8.com/color/48/000000/twitter.png"
facebook_url="https://img.icons8.com/color/48/000000/facebook-new.png"
instagram_url="https://img.icons8.com/color/48/000000/instagram-new.png"
tiktok_url="https://img.icons8.com/color/48/000000/tiktok.png"
# redirect url
linkedin_redirect_url="www.linkedin.com/in/arbaz-arshad-1b8b04216"
github_redirect_url="https://github.com/arbaz4687"
youtub_redirect_url="https://www.youtube.com/channel/@Farming368/featured"
twitter_redirect_url="https://https://twitter.com/arsha7415"
facebook_redirect_url="https://www.facebook.com/arbaz.gujjar.79/"

#add link with images
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
             f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>' 
             f'<a href="{youtub_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>'
             f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>'
             f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>',unsafe_allow_html=True)

