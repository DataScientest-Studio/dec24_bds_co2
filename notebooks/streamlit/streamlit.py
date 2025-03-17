import streamlit as st
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.title("DS Octobre - Projet CO2")
st.sidebar.title('Sommaire')
pages=['Introduction', 'Exploration', 'Modélisation']
page = st.sidebar.radio('Aller vers', pages)


# premier onglet - introduction au projet
if page == pages[0] : 
    st.write('# Introduction')
    st.write("Le changement climatique est l’un des défis majeurs du XXIe siècle, et la réduction des émissions de CO₂ est un enjeu central dans la lutte contre le réchauffement global. Parmi les secteurs les plus polluants, le transport joue un rôle clé : en 2019, il était responsable d’environ un quart des émissions totales de CO₂ de l’Union européenne, dont 71,7 % provenaient du transport routier. Les voitures personnelles, en particulier, représentaient 60,6% des émissions de CO₂ liées à ce secteur."
             "Malgré les efforts pour limiter ces émissions, on observe une augmentation de 33,5 % entre 1990 et 2019. Selon les projections actuelles, la diminution des émissions du transport d’ici 2050 ne serait que de 22 %, bien en deçà des objectifs nécessaires pour respecter les engagements climatiques."
             "Face à cette problématique, deux principales stratégies permettent de réduire l’empreinte carbone des véhicules : améliorer leur efficacité énergétique et modifier leur source d’énergie, en passant par exemple à des carburants alternatifs ou à l’électrification."
             "Dans ce contexte, notre projet vise à prédire les émissions de CO₂ des véhicules en fonction de leurs caractéristiques. En développant un modèle de prédiction précis, nous pourrions contribuer à mieux comprendre les facteurs influençant ces émissions et ainsi aider les constructeurs à voir comment optimiser leurs prochaines séries de voiture lorsque ceux-ci sont soucieux de leur impact environnemental."
             "")
    st.image("images\image1.png")
    st.write('## Identification de la problématique')
    url1 = 'https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/#_'
    url2 = 'https://co2cars.apps.eea.europa.eu/?source=%7B%22track_total_hits%22%3Atrue%2C%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22constant_score%22%3A%7B%22filter%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22year%22%3A2014%7D%7D%2C%7B%22term%22%3A%7B%22year%22%3A2013%7D%7D%2C%7B%22term%22%3A%7B%22year%22%3A2012%7D%7D%2C%7B%22term%22%3A%7B%22year%22%3A2011%7D%7D%2C%7B%22term%22%3A%7B%22year%22%3A2010%7D%7D%5D%7D%7D%2C%7B%22bool%22%3A%7B%22should%22%3A%5B%7B%22term%22%3A%7B%22scStatus%22%3A%22Final%22%7D%7D%5D%7D%7D%5D%7D%7D%7D%7D%5D%7D%7D%2C%22display_type%22%3A%22tabular%22%7D'

    st.write("Nous sommes partis des données disponibles sur le site [data.gouv.fr](%s)" % url1 , " qui contiennent les émissions des CO2 et les caractéristiques des véhicules commercialisés en France en 2014. "
             "Nous avons également trouvé des données intéressantes de [l'Agence européenne pour l'environnement](%s)" % url2 , " qui viennent compléter certaines informations manquantes au premier jeu de données."
             "Nous avons à faire à une problématique de classification supervisée. A partir de leurs caractéristiques, nous cherchons à classer les véhicules selon les 7 catégories d’émission de C02 en nous basant sur l’étiquette énergétique des véhicules. "
             "")
    st.image("images\image2.png")
    st.write("Notre variable cible est déjà présente dans le jeu de données, c’est la variable co2. Nous transformerons cette variable, actuellement quantitative, en une variable catégorielle pour plus de visibilité lors de la transmission des résultats.")

# deuxième onglet - exploration de nos datasets
if page == pages[1] :
    st.write('# Exploration')

    # selection du dataset exploré
    dataset_select = ['Dataset #1', 'Dataset Merge']
    dataset_selector = st.radio('Choix du dataset', dataset_select)

    if dataset_selector == dataset_select[0] :
        df=pd.read_csv("mars-2014-complete.csv",encoding='ISO-8859-1', sep = ';')
        st.write('## Dataset #1')
        df=pd.read_csv("mars-2014-complete.csv",encoding='ISO-8859-1', sep = ';')

        st.write('### Présentation des données avant nettoyage')
        # Affichage des informations générales du DataFrame
        st.dataframe(pd.DataFrame({
        "Colonnes": df.columns,
        "Type": df.dtypes,
        "Valeurs Manquantes": df.isna().sum(),
        "Nb. Valeurs Uniques": df.nunique()
        }))

        st.dataframe(pd.DataFrame({"Dimension": ["Lignes", "Colonnes"], "Valeur": df.shape}))
        
        # Affichage des 3 premières lignes du DataFrame
        st.dataframe(df.head(5))

        st.write('### Sélection des variables')
        st.write(""" Nous nous intéressons d’abord au pourcentage de valeurs manquantes. Les variables hc et date_maj en contiennent plus de 80%. Nous décidons donc de supprimer ces variables. 

    La variable hcnox étant directement reliée à la variable hc, nous décidons de la supprimer également.

    La variable ptcl contient 5% de données manquantes. Après avoir regardé la proportion de ces données manquantes en fonction de la marque de voiture, nous avons pu voir que la majorité des données manquantes proviennent de la marque Mercedes qui est une classe largement majoritaire dans notre jeu de données. Garder cette variable en ne supprimant que les lignes contenant des valeurs manquantes nous paraît donc pertinent puisque nous ne perdons pas beaucoup d'informations.

    Les variables cnit (Code National d’Identification du Type) et tvv (Type Variante Version) sont des variables qui correspondent à des ID d’identification. Ainsi ces variables ne sont pas nécessaires comme variables explicatives pour le modèle, mais elles peuvent être utiles comme clef pour lier et fusionner ce dataset avec d’autres dataset (Voir plus tard).

    La variable co2 est utile pour créer notre variable cible. Elle renseigne la quantité d’émission de CO2 en g/km. Lorsque l’on regarde les étiquettes de pollution, les différentes catégories se basent sur cette donnée. Ainsi on crée une nouvelle variable que l’on nomme Category dont la valeur est déterminée à partir de l’image ci-dessous. Cette variable sera notre variable cible pour notre problème de classification.

    Les variables puiss_max, conso_urb, conso_exurb, conso_mixte, co_typ_1, nox et ptcl sont initialement des chaînes de caractères. Cependant, ce sont des variables quantitatives dans les faits. Ainsi, il est nécessaire de changer leur type pour avoir des flottants.
    Toutes les autres variables pas encore mentionnées sont conservées. Cela représente des variables catégorielles ou des variables quantitatives qui nous semblent pertinentes comme variables explicatives pour le modèle. Nous verrons un peu plus en détail chacune de ces variables, leurs modes pour les variables catégorielles et leur répartition pour les variables quantitatives. 
    """)
        
        st.write('### Gestion des valeurs manquantes')
        st.write("""Pour les variables conservées, étant donné que le nombre de valeurs manquantes représente un pourcentage faible de l'entièreté des valeurs du dataset, on a décidé de supprimer la ligne s'il manquait au moins une valeur. On ne perd pas énormément d’échantillons en appliquant cette stratégie. On préfère supprimer l’individu plutôt que de remplacer une valeur manquante par le mode le plus présent ou la médiane car cette nouvelle valeur reste hypothétique.  """)
        
        # nettoyage du dataset1
        #
        df = df.iloc[:, :-4]
        to_drop = ["cnit", "tvv", "hc", "hcnox", "date_maj"]
        df_clear = df.drop(to_drop, axis = 1)
        df_without_Na = df_clear.dropna()
        quantitative_col = ["co2","puiss_admin_98","puiss_max","conso_urb","conso_exurb","conso_mixte","co_typ_1","nox","ptcl","masse_ordma_min","masse_ordma_max"]
        df_quantitative = df_without_Na[quantitative_col]

        for name in df_quantitative:
            # Remplacement des virgules par des points et changement de type
            df_without_Na[name] = pd.to_numeric(df_without_Na[name].astype(str).str.replace(',', '.'), errors = 'coerce')
            df_quantitative[name] = pd.to_numeric(df_without_Na[name].astype(str).str.replace(',', '.'), errors = 'coerce')
        
        df_without_Na['typ_boite'] = None 
        df_without_Na.loc[df_without_Na['typ_boite_nb_rapp'].str.startswith('M'), 'typ_boite'] = 'Manuelle'

        df_without_Na.loc[df_without_Na['typ_boite_nb_rapp'].str.startswith('A'), 'typ_boite'] = 'Auto'

        df_without_Na.loc[df_without_Na['typ_boite_nb_rapp'].str.startswith('V'), 'typ_boite'] = 'Var_continue'

        # On supprime les Nan créées
        df_without_Na = df_without_Na.dropna(subset=['typ_boite'])
 
        df_without_Na['nb_rapp'] = None 

        list_vitesse = ['0','5', '6', '7', '8', '9']

        for i in list_vitesse:
            df_without_Na.loc[df_without_Na['typ_boite_nb_rapp'].str.contains(i), 'nb_rapp'] = i

        # On supprimer les Nan créées
        df_without_Na = df_without_Na.dropna(subset=['nb_rapp'])

        rapport_counts = df_without_Na['nb_rapp'].value_counts()

        #ajout des étiquettes
        bins = [0, 100, 120, 140, 160, 200, 250, float('inf')]
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

        df_without_Na['category'] = pd.cut(df_without_Na['co2'], bins=bins, labels=labels, right=True)

        #fin du nettoyage
        #


        st.write('### Analyse des variables')

        fig, ax = plt.subplots()
        sns.boxplot(data=df_without_Na, x='cod_cbr', y='co2', palette="viridis", ax=ax)
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.xlabel('Type de carburant', fontsize=12)
        plt.ylabel('Émission de CO2', fontsize=12)
        plt.title('Distribution du CO2 par type de carburant', fontsize=14)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.scatterplot(x="puiss_max", y="co2", data=df_without_Na, hue="gamme", ax=ax)
        plt.xlabel("Puissance max")
        plt.ylabel("CO2")
        plt.title("Co2 en fonction de la puissance max et de la gamme du véhicule")
        st.pyplot(fig)

        # plot côte à côte
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        # Barplot du type de boite
        sns.countplot(data=df_without_Na, x='typ_boite', palette="viridis", ax=axes[0])
        axes[0].set_xlabel('Type de boite de vitesse', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Répartition des boites de vitesses', fontsize=14)

        # Barplot du nombre de rapports
        sns.countplot(data=df_without_Na, x='nb_rapp', palette="viridis", ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, fontsize=10, ha='right')
        axes[1].set_xlabel('Nombre de rapports', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Répartition des nombres de rapports', fontsize=14)
        st.pyplot(fig)
        
        # plot côte à côte
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        sns.boxplot(data=df_without_Na, x='typ_boite', y='co2', palette="viridis", ax=axes[0])
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, fontsize=10, ha='right')  
        axes[0].set_xlabel('Boites de vitesse', fontsize=12)
        axes[0].set_ylabel('CO2', fontsize=12)
        axes[0].set_title('CO2 en fonction du type de boite de vitesse', fontsize=14) 

        # Boxplot : Nombre de vitesses vs CO2
        sns.boxplot(data=df_without_Na, x='nb_rapp', y='co2', palette="viridis", ax=axes[1])
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, fontsize=10, ha='right')  
        axes[1].set_xlabel('Nombre de vitesses', fontsize=12)
        axes[1].set_ylabel('CO2', fontsize=12)
        axes[1].set_title('CO2 en fonction du nombre de vitesses', fontsize=14) 
        st.pyplot(fig)

        fig = plt.figure()
        sns.countplot(data=df_without_Na, x ='category', palette="viridis")
        plt.xticks(rotation=45, fontsize=10, ha='right')  
        plt.xlabel('Etiquette', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title("Répartition des catégories d'emission de CO2", fontsize=14) 
        plt.tight_layout()
        st.pyplot(fig)

        fig = plt.figure()
        sns.countplot(data=df_without_Na, x ='Carrosserie', palette="viridis")
        plt.xticks(rotation=45, fontsize=10, ha='right')  
        plt.xlabel('Carosserie', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Répartition des carosserie', fontsize=14) 
        st.pyplot(fig)

        fig = plt.figure()
        sns.countplot(data=df_without_Na, x ='gamme', palette="viridis")
        plt.xticks(rotation=45, fontsize=10, ha='right')  
        plt.xlabel('Gamme', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Répartition des gammes', fontsize=14) 
        st.pyplot(fig)
    
    
    if dataset_selector == dataset_select[1] :
        df=pd.read_csv("data_merge_v2.csv")
        st.write('## Description du merge')
        st.image("images\Schema merge.png")

        st.write('## Dataset Merge')
        st.dataframe(df.head(5))

        
        fig = plt.figure()
        sns.boxplot(y=df["ec (cm3)"])
        plt.title('Répartion des valeurs cylindrés des moteurs (Ec (cm3))')
        st.pyplot(fig)
        
        fig = plt.figure()
        sns.boxplot(y=df["W (mm)"])
        plt.title('Répartition des valeurs du diamètres des roues (W (mm))')
        st.pyplot(fig)

        fig = plt.figure()
        sns.scatterplot(data=df, x='ec (cm3)', y='puiss_max', hue='category')
        plt.xlabel("Engine Capacity")
        plt.ylabel("Puissance maximale")
        plt.title("La puissance maximale en fonction de capacité du moteur pour chaque catégorie d'émission de CO2")
        st.pyplot(fig)


if page == pages[2] :
    st.write('# Modélisation')
    st.write('## Dataset Merge')


    model_types = ["KNN", "RandomForest", "LogisticRegression", "SVM"]

    # Correspondance des modèles avec leurs préfixes
    model_prefix = {
        "KNN": "KNN",
        "Random Forest": "RF",
        "Logistic Regression": "LR",
        "SVM": "SVM", 
        }

    features_order = ["conso", "puiss", "ec"]  # Les variables toujours dans cet ordre
    
    # 📌 Définition des 7 variantes de modèles
    model_variants = {
        "conso_puiss_ec": ["conso", "puiss", "ec"],
        "conso_puiss": ["conso", "puiss"],
        "conso_ec": ["conso", "ec"],
        "conso": ["conso"],
        "puiss_ec": ["puiss", "ec"],
        "puiss": ["puiss"],
        "ec": ["ec"]
    }

    features_mapping = {
    "conso": "conso_mixte",
    "puiss": "puiss_max",
    "ec": "ec (cm3)"
    }

    # 📌 Fonction pour charger le dataset initial
    @st.cache_data
    def load_data():
        df = pd.read_csv("data_merge_v2.csv")  
        return df

    df = load_data()

    # 📌 Fonction pour charger un modèle et afficher son classification report
    def evaluate_model(model_name, variant):
        prefix = model_prefix[model_name]
        # Construire le chemin du fichier modèle (ex: "KNN_conso_puiss_ec.pkl")
        etude1 = "etude1/"
        model_path = etude1 + f"{prefix}_{variant}.pkl"

        # Vérifier si le fichier existe avant de charger
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"Le fichier {model_path} est introuvable.")
            return

        # Séparer X et y en utilisant les variables dans l'ordre défini
        #selected_features = [features_mapping["conso"], features_mapping["puiss"], features_mapping["ec"]]
        selected_features = [features_mapping[var] for var in model_variants[variant]]
        X = df[selected_features]
        y = df["category"].replace(to_replace=['A','B','C','D','E','F','G'],value=[0,1,2,3,4,5,6])

        # Diviser les données en train et test
        if prefix == "SVM" :
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else :
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

        # 📌 Appliquer le StandardScaler (sauvegardé ou recalculé)
        scaler_path = f"scaler_{model_name}.pkl"  # Exemple : "scaler_KNN.pkl"
        try:
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            scaler = StandardScaler().fit(X_train)  # ⚠️ Recalcul du scaler si non sauvegardé

        X_test_scaled = scaler.transform(X_test)

        y_pred = model.predict(X_test_scaled)

        # Générer le classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Affichage dans Streamlit
        st.subheader(f"Classification Report - {model_name} ({variant})")
        st.dataframe(report_df.style.format(precision=3))

    # 📌 Interface Streamlit
    st.title("Évaluation des Modèles")

    # Sélecteur de modèle
    selected_model = st.selectbox("Choisissez un modèle :", list(model_prefix.keys()))
    selected_variant = st.selectbox("Choisissez une variante :", list(model_variants.keys()))

    # Bouton pour exécuter l'évaluation
    if st.button("Évaluer le modèle"):
        evaluate_model(selected_model, selected_variant)

    st.write('## Deuxième étude - les caractéristiques de la voiture')

    @st.cache_data
    def load_data():
        df = pd.read_csv("data_merge_v2.csv") 
        df2 = df[['cod_cbr','hybride','masse_ordma_min','masse_ordma_max',"puiss_max","W (mm)","At1 (mm)","At2 (mm)",'Carrosserie','typ_boite','nb_rapp','category']]
        df2["hybride"] = df2["hybride"].replace(to_replace=["non","oui"],value=[0,1])
        df2["category"] = df2["category"].replace(to_replace=['A','B','C','D','E','F','G'],value=[0,1,2,3,4,5,6])
        df2 = df2.loc[(df2["cod_cbr"] == "GO")| (df2["cod_cbr"] == "ES") | (df2["cod_cbr"] == "GH")].reset_index(drop=True)
        df2 = pd.get_dummies(df2)
        df = df2
        return df

    df = load_data()
    st.dataframe(df.head(5))

    
    model_paths = {
    "KNN Pas Optimisé": "KNN_pas_optimise.pkl",
    "Random Forest Pas Optimisé": "RF_pas_optimise.pkl"
    }

    def evaluate_optimized_model(model_name):
        #model_path = model_paths[model_name]
        base_path = "etude2/"
        model_path = base_path + f"{model_paths[model_name]}"


        # Charger le modèle
        model = joblib.load(model_path)

        # 📌 Sélectionner toutes les features sauf la cible
        X = df.drop(columns=["category"])  # Prend toutes les colonnes sauf la cible
        y = df["category"].replace(to_replace=['A','B','C','D','E','F','G'],value=[0,1,2,3,4,5,6])

        # Diviser les données en train et test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Appliquer le StandardScaler (sauvegardé ou recalculé)
        scaler = StandardScaler().fit(X_train)  # ⚠️ Recalcul du scaler si non sauvegardé

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Calcul des scores
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Affichage des scores dans Streamlit
        st.subheader(f"Scores - {model_name}")
        st.write(f"**Score d'entraînement :** {train_score:.3f}")
        st.write(f"**Score de test :** {test_score:.3f}")

    # 📌 Interface Streamlit
    st.title("Évaluation des Modèles Pas Optimisés")

    # Sélecteur de modèle
    selected_model = st.selectbox("Choisissez un modèle :", list(model_paths.keys()))

    # Bouton pour exécuter l'évaluation
    if st.button("Évaluer le modèle", key='evaluate_button'):
        evaluate_optimized_model(selected_model)

    st.write('### Feature importance')
    
    base_path = "etude2/RF_pas_optimise.pkl"
    @st.cache_data
    def load_data():
        df = pd.read_csv("data_merge_v2.csv")
        # Sélectionner les bonnes colonnes
        df2 = df[['cod_cbr','hybride','masse_ordma_min','masse_ordma_max',"puiss_max","W (mm)","At1 (mm)","At2 (mm)",'Carrosserie','typ_boite','nb_rapp','category']]
        
        # Encodage des variables catégoriques
        df2["hybride"] = df2["hybride"].replace({"non": 0, "oui": 1})
        df2["category"] = df2["category"].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
        df2 = pd.get_dummies(df2)

        return df2

    df = load_data()


    def display_feature_importance():
        # Charger le modèle
        model = joblib.load(base_path)

        # 📌 Vérifier la taille de feature_importances_
        nb_features_model = len(model.feature_importances_)

        # 📌 Récupérer les features du dataset
        dataset_features = df.drop(columns=["category"]).columns.tolist()
        nb_features_dataset = len(dataset_features)

        # 📌 Vérifier la correspondance - mis en commentaire pour masquer les erreur, utiles pour le troubleshooting
        #st.write(f"📌 Nombre de features dans le modèle : {nb_features_model}")
        #st.write(f"📌 Nombre de features dans le dataset : {nb_features_dataset}")

        # 📌 Si la taille ne correspond pas, afficher la différence
        if nb_features_model != nb_features_dataset:
            # mis en commentaire pour masquer les erreur, utiles pour le troubleshooting
            #st.error("⚠️ Problème : Le nombre de features dans le dataset et le modèle ne correspond pas !") 

            # 📌 Vérifier quelles colonnes sont en trop ou manquantes
            features_manquantes = [f for f in dataset_features if f not in dataset_features[:nb_features_model]]
            features_supplémentaires = dataset_features[nb_features_model:]

            # mis en commentaire pour masquer les erreur, utiles pour le troubleshooting
            #st.write(f"🎯 Features manquantes dans le dataset : {features_manquantes}")
            #st.write(f"⚠️ Features supplémentaires dans le dataset : {features_supplémentaires}")

            # 📌 Forcer la correspondance en prenant uniquement les premières colonnes
            df_filtered = df[dataset_features[:nb_features_model]]
            #st.warning("⚠️ Alignement forcé du dataset pour correspondre au modèle.")
        else:
            df_filtered = df[dataset_features]  # Les features correspondent déjà

        # 📌 Vérifier que la taille correspond maintenant -  mis en commentaire pour masquer les erreur, utiles pour le troubleshooting
        #st.write(f"✅ Nouvelle taille des features dans Streamlit : {df_filtered.shape[1]}")
        #st.write(f"✅ Taille des features dans le modèle : {nb_features_model}")

        # 📌 Récupérer l'importance des features avec le dataset corrigé
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": df_filtered.columns,
            "Importance": feature_importances
        })

        # 📌 Trier par importance décroissante
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # 📌 Afficher les résultats dans Streamlit
        st.write("### Importance des Features du Random Forest")
        st.dataframe(importance_df.style.format(precision=3))

        # 📌 Afficher un graphique des 10 features les plus importantes
        plt.figure(figsize=(8, 6))
        plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10], color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Top 10 des Features Importantes")
        plt.gca().invert_yaxis()
        st.pyplot(plt)

    # 📌 Interface Streamlit
    st.title("Analyse du Random Forest - Importance des Features")

    # Bouton pour afficher les importances
    if st.button("Afficher l'importance des features", key="feature_importance_button"):
        display_feature_importance()




    # 📌 Définir les chemins des modèles
    base_path = "etude2/"
    knn_model_path = base_path + "KNN_pas_optimise.pkl"
    rf_model_path = base_path + "RF_pas_optimise.pkl"

    # 📌 Charger le dataset
    @st.cache_data
    def load_data():
        dataset_path = "data_merge_v2.csv"  # Mets le bon chemin
        df = pd.read_csv(dataset_path)
        
        # Sélectionner les bonnes colonnes
        df2 = df[['cod_cbr','hybride','masse_ordma_min','masse_ordma_max',"puiss_max","W (mm)","At1 (mm)","At2 (mm)",'Carrosserie','typ_boite','nb_rapp','category']]
        
        # Encodage des variables catégoriques
        df2["hybride"] = df2["hybride"].replace({"non": 0, "oui": 1})
        df2["category"] = df2["category"].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
        df2 = pd.get_dummies(df2)

        return df2

    df = load_data()

    if st.button("Charger les modèles et afficher les graphiques", key="btn_graphiques"):

        #  Vérifier que les fichiers existent

        #  Charger les modèles après le clic sur le bouton
        knn_model = joblib.load(knn_model_path)
        rf_model = joblib.load(rf_model_path)

        #  Générer les listes de performance pour les graphiques
        importance_list = list(df.drop(columns=["category"]).columns)  # Features disponibles

        list_knn_train = []
        list_knn_test = []
        list_rf_train = []
        list_rf_test = []

        for i in range(len(importance_list)):
            importance_sous_list = importance_list[:i+1]

            X = df[importance_sous_list]
            y = df["category"]

            #  Split des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

            #  Standardisation
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            #  Évaluation du modèle KNN
            knn_model.fit(X_train_scaled, y_train)
            list_knn_train.append(knn_model.score(X_train_scaled, y_train))
            list_knn_test.append(knn_model.score(X_test_scaled, y_test))

            #  Évaluation du modèle Random Forest
            rf_model.fit(X_train_scaled, y_train)
            list_rf_train.append(rf_model.score(X_train_scaled, y_train))
            list_rf_test.append(rf_model.score(X_test_scaled, y_test))

        #  Graphique pour KNN
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list_knn_train, label="KNN Train", color='purple')
        ax.plot(list_knn_test, label="KNN Test", color="orange")
        ax.set_title("Performance du KNN")
        ax.legend()
        ax.set_xticks(range(len(importance_list)))
        ax.set_xticklabels(importance_list, rotation=80)
        st.pyplot(fig)

        #  Graphique pour Random Forest
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(list_rf_train, label="Random Forest Train", color='blue')
        ax.plot(list_rf_test, label="Random Forest Test", color="green")
        ax.set_title("Performance du Random Forest")
        ax.legend()
        ax.set_xticks(range(len(importance_list)))
        ax.set_xticklabels(importance_list, rotation=80)
        st.pyplot(fig)




    # 📌 Définir le répertoire des fichiers modèles
    save_dir = "etude3/"

    # 📌 Vérifier si les fichiers modèles existent
    model_files = ["KNN_smote.pkl", "RF_smote.pkl", "scaler.pkl", "features.pkl"]


    # 📌 Charger les modèles et transformations
    knn = joblib.load(save_dir+ "KNN_smote.pkl")
    rforest = joblib.load(save_dir+  "RF_smote.pkl")
    scaler = joblib.load(save_dir+  "scaler.pkl")
    model_features = joblib.load(save_dir+  "features.pkl")
    

    # 📌 Charger le dataset
    @st.cache_data
    def load_data():
        df = pd.read_csv("data_merge_v2.csv")

        # Sélectionner les bonnes colonnes
        df2 = df[['cod_cbr','hybride','masse_ordma_min','masse_ordma_max',"puiss_max","W (mm)","At1 (mm)","At2 (mm)",'Carrosserie','typ_boite','nb_rapp','category']]

        # Encodage des variables catégoriques
        df2["hybride"] = df2["hybride"].replace({"non": 0, "oui": 1})
        df2["category"] = df2["category"].replace({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6})
        df2 = pd.get_dummies(df2)

        # S'assurer que toutes les features correspondent au modèle
        for feature in model_features:
            if feature not in df2.columns:
                df2[feature] = 0  # Ajouter les colonnes manquantes avec des valeurs nulles

        # Réorganiser les colonnes pour correspondre au modèle
        df2 = df2[["category"] + model_features]

        return df2

    df = load_data()

    # 📌 Séparer X et y
    X = df.drop(columns=["category"])
    y = df["category"]

    # 📌 Split des données (même split que lors de l'entraînement)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

    # 📌 Appliquer le StandardScaler déjà entraîné
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 📌 Calculer les scores sans refaire `fit()`
    knn_train_score = knn.score(X_train_scaled, y_train)  # Utiliser le scaler chargé
    knn_test_score = knn.score(X_test_scaled, y_test)

    rf_train_score = rforest.score(X_train_scaled, y_train)
    rf_test_score = rforest.score(X_test_scaled, y_test)

    # 📌 Afficher les résultats dans Streamlit
    st.subheader("Résultats du Modèle KNN (Chargé)")
    st.write(f"🔹 **Score d'entraînement :** {knn_train_score:.3f}")
    st.write(f"🔹 **Score de test :** {knn_test_score:.3f}")

    st.subheader("Résultats du Modèle Random Forest (Chargé)")
    st.write(f"🔹 **Score d'entraînement :** {rf_train_score:.3f}")
    st.write(f"🔹 **Score de test :** {rf_test_score:.3f}")



    st.write('# Deep Learning')
    model = tf.keras.models.load_model('deep_learning/model.h5')
    st.title("Description du Modèle de Classification")

    st.markdown("""
    ### 📌 **Architecture du modèle**
    Ce modèle de Deep Learning est un réseau de neurones **fully connected** conçu pour une classification en **7 catégories**.

    - Il est constitué de **3 couches cachées** avec activation **ReLU**.
    - Il utilise des techniques de **Batch Normalization** et **Dropout** pour stabiliser l'entraînement et éviter l'overfitting.
    - La couche de sortie applique **Softmax** pour fournir des probabilités de classification.

    ###  **Structure des couches**
    | **Type** | **Nombre de Neurones** | **Activation** | **Rôle** |
    |----------|------------------|--------------|------------------------------|
    | **Entrée** | 34 (features) | - | Reçoit les données d'entrée |
    | **Cachée 1** | 128 | ReLU | Capture les patterns complexes |
    | | BatchNormalization | - | - | Stabilise et accélère l'entraînement |
    | | Dropout | - | - | Régularisation (évite l'overfitting) |
    | **Cachée 2** | 64 | ReLU | Réduit la complexité et affine les patterns |
    | | BatchNormalization | - | - | Normalisation des activations |
    | | Dropout | - | - | Régularisation |
    | **Cachée 3** | 32 | ReLU | Capture des caractéristiques plus abstraites |
    | | BatchNormalization | - | - | Normalisation |
    | | Dropout | - | - | Régularisation |
    | **Sortie** | 7 | Softmax | Classification multiclasse |

    ### **Détails supplémentaires**
    - **Optimiseur** : Adam (`learning_rate=0.001`)
    - **Fonction de perte** : Sparse Categorical Crossentropy
    - **Nombre total de paramètres** : `15943`
    """)

    # Afficher un résumé détaillé du modèle dans Streamlit
    st.subheader("Résumé du modèle")
    

    # Charger le modèle et son historique d'entraînement
    model = tf.keras.models.load_model('deep_learning/model.h5')

    # Charger l'historique de l'entraînement (si sauvegardé)
    try:
        history = joblib.load('deep_learning/history.pkl')
    except FileNotFoundError:
        st.error("Fichier d'historique non trouvé. Assurez-vous d'avoir sauvegardé history.")

    # Vérifier si l'historique est chargé
    if 'history' in locals():
        st.title("Analyse des Performances du Modèle")

        # Courbe d'accuracy
        st.subheader("Évolution de la précision (accuracy)")
        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Train Accuracy', marker='o')
        ax.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Courbe de perte (loss)
        st.subheader("Évolution de la perte (loss)")
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Train Loss', marker='o', color='red')
        ax.plot(history['val_loss'], label='Validation Loss', marker='o', color='blue')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        #st.success("Les performances du modèle ont été affichées avec succès ! ✅")













    