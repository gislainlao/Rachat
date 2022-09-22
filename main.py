import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import shap
import plotly.express as px
from zipfile import ZipFile
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from datetime import datetime
plt.style.use('fivethirtyeight')


#Loading data……
#df= load_data()

#targets = pred.value_counts()
#######################################
    # SIDEBAR
 #######################################

 #Title display
html_temp = """
<div style="background-color: #D92F21; padding:10px; border-radius:10px">
<h1 style="color: white; text-align:center">TABLEAU DE BORD DE PRÉVISION DES RACHATS</h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">Cette application web permet de visualiser le portefeuil de SUNU Vie et de faire des préisions...</p>
"""
st.markdown(html_temp, unsafe_allow_html=True)


#cols_when_model_builds = clf.get_booster().feature_names
#xtest = xtest[cols_when_model_builds]
#Customer ID selection
st.sidebar.header("**Info Générale**")

c=['ASSURANCE INDIVIDUELLE','ASSURANCE COLLECTIVE']
Garantie = st.sidebar.selectbox('Type Assurance',c)
if Garantie=='ASSURANCE INDIVIDUELLE':
    uploaded_file=st.sidebar.file_uploader("Choose a XLSX file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
