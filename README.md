# Kaggle_CareerCon_2019

My top 2% solution to Kaggle's CareerCon 2019 competition

Interesting things about this project include the out-of-the-box solution used to overcome the noise in the data. The sequences provided were too noisy to classify accurately so a novel method of chaining together sequences was used. This method was based on the sequences' orientation data.

Each sequence was first labelled using a random forest classifier. Then the sequences were chained together. Each sequence then 'voted' for the classification of the entire chain which then overruled each individual sequence's original classification.

**Keywords:** Python (Pandas, Matplotlib) / Logistic Regression / SVM / Random Forest / Gradboost / XGBoost / Calibration / Model Pipeline
