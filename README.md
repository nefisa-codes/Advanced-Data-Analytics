# Advanced-Data-Analytics
Project Summary: Time Series Forecasting and Sentiment Analysis with Neural Networks

Overview:
This portfolio showcases two advanced data analytics projects:

Time Series Forecasting using ARIMA to predict organizational revenue.

Sentiment Analysis using a Bidirectional LSTM neural network to classify customer reviews as positive or negative.

Both projects demonstrate proficiency in data cleaning, exploratory analysis, model selection, and actionable insights—key skills for data-driven decision-making.

1. Time Series Forecasting: Revenue Prediction
Objective:
Forecast daily revenue for the next 30 days to aid financial planning and resource allocation.

Key Steps:

Data Preparation: Cleaned and formatted time series data (731 days) and addressed non-stationarity via differencing.

Modeling: Identified the best-fit ARIMA model (ARIMA(1,1,0)(5,1,0)[12]) using auto-ARIMA, accounting for seasonality and trends.

Evaluation: Achieved a Mean Absolute Error (MAE) of 0.387, indicating high predictive accuracy.

Outcome: Generated a 30-day forecast with confidence intervals, revealing a gradual revenue uptrend.

Business Impact:
Enables proactive budgeting and aligns operational strategies with seasonal revenue patterns.

2. Sentiment Analysis: Review Classification
Objective:
Build a neural network to classify product reviews as positive or negative for customer feedback analysis.

Key Steps:

Data Preparation:

Cleaned text data (2,748 reviews) by removing noise, tokenizing, and padding sequences.

Split data into 80% training and 20% testing sets.

Modeling:

Designed a Bidirectional LSTM network with embedding layers, dropout (for overfitting), and sigmoid activation.

Trained using binary cross-entropy loss and Adam optimizer.

Evaluation:

Achieved 79% accuracy on test data, with validation metrics showing strong generalization.

Early stopping prevented overfitting (training halted at epoch 8).

Business Impact:
Provides scalable insights into customer sentiment, guiding product improvements and marketing strategies.

Technical Highlights
Tools Used: Python, Pandas, Statsmodels (ARIMA), TensorFlow/Keras (LSTM), NLTK.

Key Skills:

Time series decomposition, stationarity testing, and hyperparameter tuning.

NLP techniques (tokenization, embeddings) and neural network architecture design.

Deliverables:

Interactive Jupyter Notebooks with visualizations (ACF/PACF plots, loss/accuracy curves).

Deployed models saved for reproducibility (my_model.keras).
