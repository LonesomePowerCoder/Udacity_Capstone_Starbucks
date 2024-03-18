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
The models are used in an interactive web app (streamlit), which can be executed with the following command line expression "streamlit run app/run.py"

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

![image](https://github.com/LonesomePowerCoder/Udacity_Capstone_Starbucks/assets/31338782/0be5b540-7620-4f02-bce1-7b83178107b3)

# Modeling

For modelling, the random forest classifier was used as it performs quite well in classification tasks. Random Forest is an ensemble learning method used for classification and regression tasks. It builds multiple decision trees using bootstrapped samples and random feature selection. The final prediction is determined by aggregating the predictions of individual trees (mode for classification, average for regression).

# Hyperparameter Tuning

To finetune the model, a hyperparameter tuning based on grid search and cross-validation (using GridSerchCV) was conducted:

![image](https://github.com/LonesomePowerCoder/Udacity_Capstone_Starbucks/assets/31338782/615adea7-8a59-42c0-ab88-d8c727ede794)


# Results



# Comparison Table



# Conclusion

In conclusion, the model offers a good robust and efficient solution for determining the suitability of offers for distribution. By leveraging advanced machine learning techniques, it accurately assesses the relevance and potential impact of each advertisement, optimizing targeting efforts while minimizing the risk of sending irrelevant or unwanted content. With its ability to enhance advertising campaign efficiency and effectiveness, the model is suitable to support the decision-making processes in marketing, driving better engagement and ultimately maximizing return on investment.

# Limitations and suggestions for improvement

1. Data Collection Mechanism:

A larger data set would definitely improve the quality of the resulting ML-model.

2. Web app UI Refinement:

Conducting usability testing and incorporating user feedback can improve web app intuitiveness and accessibility.

3. Feature Engineering Refinement:

Iteratively refining feature selection and engineering techniques can enhance model interpretability and robustness, potentially through advanced feature selection algorithms or domain-specific feature engineering strategies.

4. Model Explainability:

Integrating model interpretability techniques such as SHAP values or LIME can provide insights into model predictions and increase stakeholder trust and understanding.

Implementing these improvements will enhance project performance and usability, ensuring long-term relevance and sustainability.

# Acknowledgement

I would like thank Starbucks for providing the dataset to the Udacity "Data Scientist" nanodegree program.
