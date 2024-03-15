import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import json
import pickle

def app():

    st.title("Detecting customers that should not get offer")
    st.subheader("This app will calculate the probability that a customer will complete an offer without viewing it."
                 "Please select your parameters in the sidebar to the left")

    # load pre-calculated model from repository
    file = open('models/completer_model.pkl', 'rb')
    cv = pickle.load(file)

    # building streamlit sidebar items
    with st.sidebar:
        buttonClick = st.button('Calculate')
        st.header('Premises')
        age = st.number_input('Age of customer', min_value=1, max_value=99, step=1, value=50)
        views = st.number_input('Viewed offers up to now', min_value=0, max_value=99, step=1, value=5)
        member_days = st.number_input('Days of membership', min_value=1, max_value=9999, step=25, value=200)
        income = st.number_input('income', min_value=1, max_value=999999, step=1000, value=50000)
        gender = st.radio('gender', options=['F', 'M', 'O'])
        duration = st.number_input('Duration of offer', min_value=1, max_value=20, step=1, value=7)
        ch_email = st.number_input('Offer via Email', min_value=0, max_value=1, step=1, value=1)
        ch_mobile = st.number_input('Offer via Mobile', min_value=0, max_value=1, step=1, value=1)
        ch_social = st.number_input('Offer via Social Media', min_value=0, max_value=1, step=1, value=1)
        ch_web = st.number_input('Offer via Web', min_value=0, max_value=1, step=1, value=1)
        offer_type = st.radio('Offer type', options=['bogo', 'discount'])

    # Convert user input from radio buttons to binary variables
    if buttonClick:
        if gender == 'F':
            F = 1
            M = 0
            O = 0
        elif gender == 'M':
            F = 0
            M = 1
            O = 0
        else:
            F = 0
            M = 0
            O = 1

        if offer_type == 'bogo':
            bogo = 1
            discount = 0
        else:
            bogo = 0
            discount = 1

        # set up the DataFrame with a new observation
        X_test = pd.DataFrame(
            columns=['cumsum_views', 'age', 'income', 'member_days', 'F', 'M', 'O', 'duration', 'ch_email', 'ch_mobile',
                     'ch_social', 'ch_web', 'bogo', 'discount'])

        X_test.loc[0] = [views, age, income, member_days, F, M, O, duration, ch_email, ch_mobile, ch_social, ch_web, bogo, discount]

        # Predict probability of viewing an offer
        y_pred_test = cv.predict_proba(X_test)

        # Display the metric
        st.metric(f'Probability of completing an offer w/o viewing is', value=f'{round(y_pred_test[0,0]*100,2)} %')