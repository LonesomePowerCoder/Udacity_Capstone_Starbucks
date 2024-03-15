import streamlit as st
from multipage import MultiPage
from func import offer_viewer, offer_completer

app = MultiPage()

# Title of the main page
st.set_page_config(layout="wide",
                    page_icon="chart_with_upwards_trend",
                   page_title ="Udacity Capstone Project",
                   )

# func to be displayed
app.add_page("Offer viewer", offer_viewer.app)
app.add_page("Offer completer", offer_completer.app)

# The main app
app.run()