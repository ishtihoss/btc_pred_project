#Bitcoin Price Prediction Project

##Overview
This project aims to predict Bitcoin's closing prices using a combination of historical price data and sentiment analysis of news headlines. By leveraging time series data and deep learning models, the project provides valuable insights into the factors influencing Bitcoin prices, combining financial analytics with natural language processing.

##Features
###1. Data Collection and Preprocessing
Historical Price Data: Loads Bitcoin historical price data from CSV files.
News Headlines: Utilizes the NewsAPI to fetch news headlines related to Bitcoin from reputable sources.
Sentiment Analysis: Employs the transformers library to perform sentiment analysis on news headlines, providing insights into public sentiment and opinion regarding Bitcoin.
Data Integration: Combines price data with sentiment scores to create a comprehensive dataset for modeling.
###2. Exploratory Data Analysis (EDA)
Provides statistical summaries, distribution analysis, and time-series plots of Bitcoin prices.
Utilizes libraries like matplotlib and seaborn for insightful visualizations.
###3. Feature Engineering
Lagged Features: Implements lagged features for time series analysis, allowing the model to recognize patterns and trends in historical data.
Sentiment Encoding: Encodes sentiment labels into one-hot encoding, integrating qualitative data into the model.
Feature Scaling: Scales numerical features using MinMaxScaler and StandardScaler, optimizing model performance.
###4. Model Building
LSTM Networks: Utilizes Long Short-Term Memory (LSTM) networks for sequence modeling, capturing dependencies and patterns over time.
Deep Learning Framework: Employs Keras and TensorFlow for building and training the deep learning model.
Hyperparameter Tuning: Implements a GridSearchCV approach to optimize model parameters for best performance.
###5. Prediction and Evaluation
Forecasting: Trains the model to predict the next three days' closing prices, providing actionable insights for investment decisions.
Early Stopping: Includes early stopping to avoid overfitting, ensuring model robustness.
Training and Validation Visualization: Visualizes training and validation loss, aiding in understanding model convergence and performance.
##Dependencies
pandas
matplotlib
seaborn
NewsAPI
numpy
transformers (for sentiment analysis)
scikit-learn
TensorFlow
Conclusion
This project serves as a introductory framework for analyzing and predicting Bitcoin prices. By integrating financial data with sentiment analysis, it showcases a multi-dimensional approach to investment analytics. The inclusion of sentiment analysis, powered by the transformers library, adds a unique perspective by gauging public sentiment, offering a nuanced view of market dynamics.

##Author
Md Ishtiaque Hossain AKA Ishti