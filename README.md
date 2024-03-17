# Udacity Capstone Starbucks
This repo hosts all files for the Capstone Project of the Udacity "Data Scientist" nanodegree program

# High-level overview
This is the final project, a.k.a. Capstone Project which is part of the Udacity "Data Scientist" nanodegree program.

# Description of Input Data
The dataset was based on Starbucks rewards mobile app was simulated. It is supposed to imitate customer behaviour w.r.t. purchasing and viewing/completing offers. For all customers there are records of different events, e.g. receiving offers, opening offers and making purchases. For incentivization, customers can receive three types of offers: BOGO (buy one get one free), Discount and informational (no reward).

All data are provided by Udacity, Inc. as part of the "Data Scientist" nanodegree program

# Strategy for solving the problem
The goal of this project is to leverage Starbucks customer and offer data, in order answer the following two questions:

Q1. What is the probability of viewing an offer based on user statistics and offer characteristics?

Q2. What is the probability of completing an offer w/o viewing based on user and offer characteristics?

For both questions, a machine learning model is trained and afterwards tested against unknown data.

# Discussion of the expected solution

The workflow is as follows:

Pre-Calculations:

1. Data import
2. Data processing
3. Data modelling
4. Hyper-parameter tuning
5. Depositing pre-calculated models (pickle files)

The web app

The user of the web app is required to select between Q1 and Q2. Then, he or she can calculate probabilities for either of those questions. The results allow the user to indicate, which customer group should be adressed by offers and which not.

# Metrics with justification

For evaluation, the balanced accuracy score was used. The balanced accuracy score is an important metric, especially in scenarios where class imbalance exists within the dataset. Class imbalance refers to situations where the number of observations in each class is not evenly distributed. In such cases, accuracy alone might not be an adequate measure of model performance.

# EDA

# Data & Code

The datasets are json files, which were zipped due to filesize restrictions in Github. 

The two models to answer the questions below, are pre-calculated and deposited as pickle files. Those models can be executed by running "models/process_data.py"
The models are used in an interactive web app (streamlit), which can be executed with the folowwing command line expression "streamlit run app/run.py"

# Installations

The project was written in Python (IDE: PyCharm). The following public packackes were used:

- streamlit
- pandas
- pickle
- json
- zipfile
- datetime
- scikit-learn
- multipage (adapted from https://github.com/prakharrathi25/data-storyteller/blob/main/multipage.py)

# Data Preprocessing

# Modeling



# Hyperparameter Tuning



# Results



# Comparison Table



# Conclusion



# Improvements



# Acknowledgement

I would like thank Starbucks for providing the dataset to the Udacity "Data Scientist" nanodegree program.
