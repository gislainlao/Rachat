import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import plotly.express as px
from zipfile import ZipFile
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from datetime import datetime
plt.style.use('fivethirtyeight')

#@st.cache
#def load_data():

    #df= pd.read_excel("C:/Users/22892/Desktop/FINAL/Final.xlsx")
    
    #return df
def data_load(df):
   
    df=df.assign(Class_Social=0)
    for i in range(len(df)):
        if df.Prime_Totale[i] <= 10000 :
            df.Class_Social[i]= 0
        else:
            if df.Prime_Totale[i] > 10000 and df.Prime_Totale[i] <= 50000 :
                df.Class_Social[i]= 1
            else:
                if df.Prime_Totale[i] > 50000 and df.Prime_Totale[i] <= 100000 :
                    df.Class_Social[i]= 2
                else:
                    df.Class_Social[i]= 3
    
    date_format = "%Y-%m-%d"
    Ag=round(((pd.to_datetime(df.Date_effet, date_format)-pd.to_datetime(df.Date_naissance, date_format)).dt.days)/365,0)
    Anc=round(((pd.to_datetime(df.Date_cloture, date_format)-pd.to_datetime(df.Date_effet, date_format)).dt.days)/365,0)
    gr=round(((pd.to_datetime(df.Date_cloture, date_format)-pd.to_datetime(df.Date_naissance, date_format)).dt.days)/365,0)
    dr=round(((pd.to_datetime(df.Date_echeance, date_format)-pd.to_datetime(df.Date_effet, date_format)).dt.days)/365,0)


    df=df.assign(Genre=0)
    for i in range(len(df.Police)):
        if df.Titre[i] == "MME" or  df.Titre[i]=="MLE":
            df.Genre[i]='F'
        else:
            df.Genre[i]='G'

    for i in range(len(df)):
        if df.IMpayé_Rachat[i] < 0:
            df.IMpayé_Rachat[i]=0
    
    df['Age_souscription'] = pd.DataFrame(Ag)
    df['Ancienete_contract'] = pd.DataFrame(Anc)
    df['Ag_Rachat'] = pd.DataFrame(gr)
    df['Dure_contrat'] = pd.DataFrame(dr)
    data = df
    Police = df['Police']
    df=df.drop(columns = ['Non_Assure','Police','Date_cloture','Date_naissance','Date_effet','Titre','Prime_Totale','Valeur_rachat','Date_echeance'])
    cat_data=[]
    num_data=[]
    for i,c in enumerate(df.dtypes):
        if c==object:
            cat_data.append(df.iloc[:,i])
        else:
            num_data.append(df.iloc[:,i])
    cat_data=pd.DataFrame(cat_data).transpose()
    num_data=pd.DataFrame(num_data).transpose()
    cat_data=cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
    cat_data.isnull().sum().any()
    num_data.fillna(method='bfill',inplace=True)
    num_data.isnull().sum().any()
    le=LabelEncoder()
    for i in cat_data:
        cat_data[i]=le.fit_transform(cat_data[i].astype(str))
    sample = pd.concat([Police,cat_data,num_data],axis=1)
    return data,sample

def load_model():
    '''loading the trained model'''
    pickle_in = open('XGBClassifier.pkl', 'rb') 
    clf = pickle.load(pickle_in)
    return clf

@st.cache(suppress_st_warning=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn
    
@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                     round(data["Age_souscription"].mean(), 0),
                     round(data["Ancienete_contract"].mean(), 0),
                     round(data["Ag_Rachat"].mean(), 0)]

    nb_individu = lst_infos[0]
    age_sous_moy= lst_infos[1]
    anciennete_moy= lst_infos[2]
    age_rachat_moy = lst_infos[3]

    return nb_individu, age_sous_moy, anciennete_moy,age_rachat_moy

def identite_client(data, Police):
    data_client = data[data.Police == Police]
    return data_client

def val_client(data, Police):
    Valeur_client = data[data.Police == Police].probabilite
    return Valeur_client

def valeur_rachat_prod(data, Produit):
    Valeur = data[data.Produit == Produit].Valeur_Rachat_Proba.sum()
    return Valeur

def valeur_rachat_total(data, Produit):
    listealea = []
    Pm=[]
    Val=[]
    for i in Produit:
        listealea.append(data[data.Produit == i].Valeur_Rachat_Proba.sum())
        Pm.append(data[data.Produit == i].Valeur_Rachat_Proba.sum())
    return listealea

def inf_prod(data, Produit,data2):
    Pm=[]
    Val=[]
    Nb=[]
    Nb_pred=[]
    for i in Produit:
        Pm.append(data[data.Produit == i].PM_au_Rachat.sum())
        Val.append(data[data.Produit == i].Valeur_rachat.sum())
        Nb.append(len(data[data.Produit == i]))
        Nb_pred.append(len(data2[data2.Produit == i][data2.predictions==1]))
    return Pm,Val,Nb,Nb_pred
    #Valeur_Rachat_test=resultat_final[resultat_final.Rachat == 1].Valeur_Rachat.sum()
#@st.cache(hash_funcs={XGBClassifier: id})
def load_prediction(sample, clf):     
    X=sample.iloc[:, 1:]
    #y_pred_proba_8=clf.predict_proba(X_test)[:,1]
    score = clf.predict_proba(X)[:,1]
    pred = clf.predict(X)
    return score,pred

@st.cache
def load_age_population(data):
    data_age = round((data["Age_souscription"]), 0)
    return data_age

@st.cache
def knn_training(sample):
    X=sample.iloc[:, 1:]
    knn = KMeans(n_clusters=2).fit(X)
    return knn 

@st.cache
def load_kmeans(sample, id, mdl):
    index = data[data.Police == id].index.values
    index = index[0]
    X=sample.iloc[:, 1:]
    data_client = pd.DataFrame(X.loc[X.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

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

        data,sample=data_load(df)
        id_client = data.Police.values
        clf = load_model()
        proba,pred = load_prediction(sample, clf)

        resultat_final = pd.concat([data.Police,data.Produit,data.Valeur_rachat], axis = 1).dropna()
        resultat_final['predictions'] = pred
        resultat_final['probabilite'] = proba
        resultat_final['Valeur_Rachat_Proba'] = data.Valeur_rachat*proba

        target = resultat_final.predictions.value_counts()

#Loading selectbox
        chk_id = st.sidebar.selectbox("Client ID", id_client)

            #Loading general info
        nb_individu , age_sous_moy, anciennete_moy, age_rachat_moy = load_infos_gen(data)


    ### Display of information in the sidebar ###
    #Number of loans in the sample
        st.sidebar.markdown("<u>Nombre d'individus dans la Base :</u>", unsafe_allow_html=True)
        st.sidebar.text(nb_individu)

            #Average income
        st.sidebar.markdown("<u>Age moyen a la souscription :</u>", unsafe_allow_html=True)
        st.sidebar.text(age_sous_moy)

            #Average income
        st.sidebar.markdown("<u>Ancienneté moyen du contrat:</u>", unsafe_allow_html=True)
        st.sidebar.text(anciennete_moy)

            #Average income
        st.sidebar.markdown("<u>Age moyen au Rachat :</u>", unsafe_allow_html=True)
        st.sidebar.text(age_rachat_moy)

        #st.sidebar.markdown("<u>Age moyen au Rachat :</u>", unsafe_allow_html=True)
        #st.sidebar.write(resultat_final)
            
            #PieChart
            #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,5))
        plt.pie(target, explode=[0, 0.1], labels=['No Rachat', 'Rachat'], autopct='%1.1f%%', startangle=90)
        st.sidebar.pyplot(fig)
#######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    #Display Customer ID from Sidebar


        st.write("ID de la Police :", chk_id)


            #Customer information display : Customer Gender, Age, Family status, Children, …
        st.header("**INFORMATION DU CLIENT**")

        if st.checkbox("Afficher les informations du client ?"):

            infos_client = identite_client(data, chk_id)
            st.write("**Gender : **", infos_client["Genre"].values[0])
            st.write("**Age : ** ", infos_client["Ag_Rachat"].values[0]) 
            st.write("**Nom de l'Assuré : **", infos_client["Non_Assure"].values[0])
            st.write("**Produit : **", infos_client["Produit"].values[0])
            st.write("**Prime Payée : **", infos_client["Prime_Totale"].values[0])
            st.write("**Valeur de Rachat : **", infos_client["Valeur_rachat"].values[0])

        st.header("**Analyse de données du client**")

        st.markdown("<u>Probabilité de Rachat: </u>", unsafe_allow_html=True)
        st.write(val_client(resultat_final, chk_id))
        st.markdown("<u>Données du Client :</u>", unsafe_allow_html=True)
        st.write(identite_client(sample, chk_id))

        if st.checkbox("Afficher les prévisions par Police?"):
            
            st.write(resultat_final)
            csv = convert_df(resultat_final)
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #D42E20;
            color:#ffffff;padding:10px; border-radius:10px;
            font-size: 20px; font-weight: bold; text-align:center
            }
            div.stButton > button:hover {
            background-color: #00ff00;
            color:#ff0000;padding:10px; border-radius:10px
            font-size: 20px; font-weight: bold; text-align:center
            }
            </style>""", unsafe_allow_html=True)

            col1, col2, col3  = st.columns(3)

            with col1:
                pass
            with col3:
                pass
            with col2:
                st.download_button(
            label="Download data",
            data=csv,
            file_name='Rapport_police.csv',
            mime='text/csv',
        )  
        else:
                st.markdown("<i>…</i>", unsafe_allow_html=True)

        if st.checkbox("Importance des Variables pour la Police:" ,chk_id):
                shap.initjs()
                X=sample
                X = X[X.Police == chk_id]
                X=X.iloc[:, 1:]
                number = st.slider("Choisisez le nombre de variables…", 0, 20, 5)
                fig, ax = plt.subplots(figsize=(10, 10))
                explainer = shap.TreeExplainer(load_model())
                shap_values = explainer.shap_values(X)
                shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
                st.pyplot(fig)
        else:
                st.markdown("<i>…</i>", unsafe_allow_html=True)

        if st.checkbox("Statistique descriptive"):
            fig = plt.figure(figsize=(20,15))
            plt.subplot(221)
            data_age = load_age_population(data)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
            #ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
            ax.set(title='Age a la souscription de Assuuré', xlabel='Age(Année)', ylabel='')
            st.pyplot(fig)
            plt.subplot(222)
            x = data["Ancienete_contract"]
            y = resultat_final['probabilite']
            donne=pd.concat([x,y], axis = 1)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=donne, x="Ancienete_contract", y="probabilite")
            ax.set(title='Probabilité de Rachat X Anciennetée du contrat', xlabel='Anciennetée du contract', ylabel='Probabilitéde Rachat')
            st.pyplot(fig)
        else:
                st.markdown("<i>…</i>", unsafe_allow_html=True)

        chk_voisins = st.checkbox("Affichages des polices simillaires ?")

        if chk_voisins:
                knn = load_knn(sample)
                st.markdown("<u>Liste des 10 fichiers les plus proches de ce Client :</u>", unsafe_allow_html=True)
                st.dataframe(load_kmeans(sample, chk_id, knn))
                st.markdown("<i>Cible 1 = Client avec défaut</i>", unsafe_allow_html=True)
        else:
                st.markdown("<i>…</i>", unsafe_allow_html=True)
        
        if st.checkbox("**VALEUR DE RACHAT PRÉDITE PAR PRODUIT**"):

            st.header("**Montant prédit pour les rachat par produit**")

            Pm,Val,Nb,Nb_pred = inf_prod(data, data.Produit.unique(),resultat_final)

            chk_prod = st.selectbox("Type de produit", data.Produit.unique())
            st.write("**La valeure de rachat prédite pour le Produit est : **{:.0f}".format(valeur_rachat_prod(resultat_final, chk_prod)))

            st.markdown("<u>Donnée par produit :</u>", unsafe_allow_html=True)

            prd = pd.DataFrame(data.Produit.unique())
            Resultat2 = pd.concat([prd], axis = 1)
            Resultat2.columns = ['Produit']
            Resultat2['Provision_Mathématique'] = pd.DataFrame(Pm)
            Resultat2['Valeur_Rachat'] = pd.DataFrame(Val)
            Resultat2['Montant_Predit']= pd.DataFrame((valeur_rachat_total(resultat_final, data.Produit.unique())))
            Resultat2['Nombre_Polices'] = pd.DataFrame(Nb)
            Resultat2['Nombrre_predit'] =pd.DataFrame(Nb_pred)

            #Resultat2 = Resultat2[['Produit','Provision_Mathématique','Valeur_Rachat','Nombrre_predit','Montant_Predit']]
            Montant1 = resultat_final[resultat_final.predictions == 1].Valeur_rachat.sum()
            Montant2 = resultat_final.Valeur_Rachat_Proba.sum()
            st.write(Montant1)
            st.write(Montant2)
            st.write(Resultat2)

            csv = convert_df(Resultat2)
            m = st.markdown("""
            <style>
            div.stButton > button:first-child {
            background-color: #D42E20;
            color:#ffffff;padding:10px; border-radius:10px;
            font-size: 20px; font-weight: bold; text-align:center
            }
            div.stButton > button:hover {
            background-color: #00ff00;
            color:#ff0000;padding:10px; border-radius:10px
            font-size: 20px; font-weight: bold; text-align:center
            }
            </style>""", unsafe_allow_html=True)

            col1, col2, col3  = st.columns(3)

            with col1:
                pass
            with col3:
                pass
            with col2:
                chk = st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Rappor_produit.csv',
            mime='text/csv',
        )   
            if chk:
                st.success('Rapport créé avec succes!',icon="✅")
        #st.write(valeur_rachat_prod(resultat_final, chk_prod))
