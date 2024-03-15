# Udacity_Capstone_Starbucks
This repo hosts all files for the Capstone Project of the Udacity Data Scientist Nanodegree

# Data & Code

The datasets are json files, which were zipped due to filesize restrictions in Github. 

The two models to answer the questions below, are pre-calculated and deposited as pickle files. Those models can be executed by running "models/process_data.py"
The models are used in an interactive web app (streamlit), which can be executed with the folowwing command line expression "streamlit run app/run.py"

# Project Overview

This is the final project, a.k.a. Capstone Project which is part of the Udacity "Data Scientist" nanodegree program. For this, a dataset was based on Starbucks rewards mobile app was simulated. It is supposed to imitate customer behaviour w.r.t. purchasing and viewing/completing offers. For all customers there are records of different events, e.g. receiving offers, opening offers and making purchases. For incentivization, customers can receive three types of offers: BOGO (buy one get one free), Discount and informational (no reward).

# Problem Statement

The goal of this project is to leverage Starbucks customer and offer data, in order answer the following two questions:

1. What is the probability of viewing an offer based on user statistics and offer characteristics?
2. What is the probability of completing an offer w/o viewing based on user and offer characteristics?

# Installations

The project was written in Python (IDE: PyCharm). The following public packackes were used:

- streamlit
- pandas
- pickle
- json
- zipfile
- datetime
- scikit-learn

# Data Source

All data are provided by Udacity, Inc. as part of the "Data Scientist" nanodegree program

# Acknowledgements

I would like thank Starbucks for providing the dataset to the Udacity "Data Scientist" nanodegree program.
