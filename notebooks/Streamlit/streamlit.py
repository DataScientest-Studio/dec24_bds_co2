import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    dataset_select = ['Dataset #1', 'Dataset #2']
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

        # Barplot du type de boite
        fig, ax = plt.subplots()
        sns.countplot(data=df_without_Na, x ='typ_boite', palette="viridis")
        plt.xlabel('Type de boite de vitesse', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Répartition des boites de vitesses', fontsize=14) 

        st.pyplot(fig)

        # Barplot du nombre de rapports
        fig, ax = plt.subplots()
        sns.countplot(data=df_without_Na, x ='nb_rapp', palette="viridis")
        plt.xticks(rotation=45, fontsize=10, ha='right')  
        plt.xlabel('Nombre de rapports', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Répartition des nombres de rapports', fontsize=14) 

        st.pyplot(fig)

# troisième onglet - Modélisation
if page == pages[2] :
    st.title('Modelisation')

    # Sélection du dataset dans la barre latérale
    dataset_select = ['Dataset 1', 'Dataset 2']
    dataset_selector = st.sidebar.radio('Choix du dataset', dataset_select)

    # Charger le dataset choisi
    if dataset_selector == dataset_select[0]:
        df = pd.read_csv("Donnees_propres.csv")

        # Pretraitement des données
        # Selection des variables
        to_drop = ['lib_mrq', 'lib_mod_doss', 'lib_mod', 'dscom', 'champ_v9', 'co2', 'puiss_admin_98', 'co_typ_1',
           'conso_urb', 'conso_exurb', 'conso_mixte']
        df_clear = df.drop(to_drop, axis = 1)
        # Reencodage de Hybride
        df_clear['hybride'] = df_clear['hybride'].replace({'oui':1, 'non': 0})

        # Reencodage categoy
        Etiquette = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        df_clear['category'] = df_clear['category'].replace(Etiquette)

        # Réencodage des autres variables catégorielles à l'aide d'un get_dummies
        df_encoded = pd.get_dummies(df_clear, dtype='int')
        st.markdown("""
        Après avoir nettoyé les données, nous avons réencodé les variables catégorielles   
        """)
        st.write("Affichage des données rééncodées")
        st.dataframe(df_encoded.head(5))

        # Séparation des données en variables explicatives et cible
        X = df_encoded.drop(columns ='category', axis = 1)
        y= df_encoded['category']

        # Split des données en train et test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Standardisation des données
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) 

        # Premiers modèles
        st.write('## 1. Premiers modèles')

        st.write('### Premiers modèles avec différents paramètres')

        # Bibliotheques
        import joblib
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import plotly.express as px


        models_dict = {
            "Random Forest (300 arbres)": "RF_300",
            "Random Forest (500 arbres)": "RF_500",
            "Random Forest (Balanced)": "RF_weight",
            "Random Forest (GridsearchCV)": "RF_gridsearch",
            "XGBoost": "XGB"
        }

        selected_model_name = st.selectbox("Choisissez un modèle :", list(models_dict.keys()))

        # Charger le modèle sélectionné
        model_path = models_dict[selected_model_name]
        model = joblib.load(model_path)

        # Faire les prédictions
        y_pred = model.predict(X_test_scaled)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

        # Affichage des résultats
        st.write(f"#### Résultats pour : {selected_model_name}")
        st.write(f"**Accuracy :** {accuracy:.4f}")
        st.progress(int(accuracy * 100))  # Barre de progression pour l'accuracy

        # Affichage des scores F1 par classe
        st.write("#### Scores F1 par classe :")
        for classe, valeurs in report.items():
            if isinstance(valeurs, dict):
                f1_score = valeurs.get("f1-score", None)
            if f1_score is not None:
                st.write(f"- **Classe {classe}** : {f1_score:.4f}")

         # Affichage de la matrice de confusion avec une meilleure esthétique
        st.write("#### Matrice de confusion normalisée :")
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f"Classe {i}" for i in range(len(conf_matrix))],
                              columns=[f"Classe {i}" for i in range(len(conf_matrix))])

        # Arrondir les valeurs de la matrice de confusion
        conf_matrix_df = conf_matrix_df.round(2)  # Arrondir à 2 décimales

        fig = px.imshow(conf_matrix_df, text_auto=True, aspect="auto", title="Matrice de confusion",
                labels={'x': 'Prédictions', 'y': 'Vraies valeurs'})
        fig.update_xaxes(side="top")
        st.plotly_chart(fig)


        st.write('### Premiers modèles avec différents données')
        st.markdown("""
        Les données les plus importantes qui sortent du premier RandomForest le plus performant sont (nous ne prendrons que les 5 plus importantes) :   
        """)
        st.image("importances_features.png")

        ("""
        Les données caractéristiques uniquement sont :
        - carburant
        - hybride
        - masse_ordma_min
        - masse_ordma_max
        - puiss_max
        - Carrosserie   
        - typ_boite
        - nb_rapp
        - gamme


        Choisissez un modèle et voyez ses performances en termes d'accuracy, scores F1 et matrice de confusion.
        """)

        # Récupération des données
        df=pd.read_csv("Donnees_propres.csv")
        X_imp = df[['nox', 'masse_ordma_min', 'masse_ordma_max', 'nb_rapp', 'puiss_max']]

        # Réencodage
        X_encoded_imp = pd.get_dummies(X_imp, dtype='int')

        # Séparation des données en train et test
        X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_encoded_imp, y, test_size=0.2, random_state=12)

        # Standardisation
        scaler = StandardScaler()
        X_train_importances = scaler.fit_transform(X_train_imp)
        X_test_importances = scaler.transform(X_test_imp) 

        # Récupération des données
        df=pd.read_csv("Donnees_propres.csv")
        df_caract = df[['carburant','hybride','masse_ordma_min','masse_ordma_max',
                "puiss_max",'Carrosserie','typ_boite','nb_rapp', 'gamme']]

        # Réencodage
        df_caract['hybride'] = df_caract['hybride'].replace({'oui':1, 'non': 0})
        df_encoded_cara = pd.get_dummies(df_caract, dtype='int')

        # Séparation des données en train et test
        X_train_cara, X_test_cara, y_train_cara, y_test_cara = train_test_split(df_encoded_cara, y, test_size=0.2, random_state=12)

        # Standardisation
        scaler = StandardScaler()
        X_train_scaled_cara = scaler.fit_transform(X_train_cara)
        X_test_scaled_cara = scaler.transform(X_test_cara) 


        # Dictionnaire des modèles sauvegardés
        models_dict = {
            "RandomForest (Données originales)": "RF_300",
            "XGBoost (Données originales)": "XGB",
            "RandomForest (SMOTE)": "RF_SMOTE",
            "XGBoost (Données SMOTE)": "XGB_smote",
            "RandomForest (Undersampling)": "RF_undersampling",
            "XGBoost (Undersampling)": "XGB_undersampling",
            "RandomForest (Variables plus importantes)": "RF_300_important",
            "RandomForest (Caractéristiques uniquement)": "RF_caracteristiques"         
             
        }

        # Sélectionner le modèle via Streamlit
        selected_model_name = st.selectbox("Choisissez un modèle :", list(models_dict.keys()))

        # Charger le modèle sélectionné
        model_path = models_dict[selected_model_name]
        model = joblib.load(model_path)

        # Adapter les données en fonction du modèle choisi
        if "Variables plus importantes" in selected_model_name:
            X_train_selected = X_train_importances
            X_test_selected = X_test_importances
            y_train_selected = y_train_imp
            y_test_selected = y_test_imp
        elif "Caractéristiques uniquement" in selected_model_name:
            X_train_selected = X_train_scaled_cara
            X_test_selected = X_test_scaled_cara
            y_train_selected = y_train_cara
            y_test_selected = y_test_cara
        else:  # SMOTE, Undersampling et données originales
            X_train_selected = X_train_scaled  # Utilisation des X d'origine
            X_test_selected = X_test_scaled
            y_train_selected = y_train  # Cible correspondante
            y_test_selected = y_test

        # Prédictions
        y_pred = model.predict(X_test_selected)

        # Calcul des métriques
        accuracy = accuracy_score(y_test_selected, y_pred)
        report = classification_report(y_test_selected, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test_selected, y_pred, normalize='true')

        # Affichage des résultats
        st.write(f"#### Résultats pour : {selected_model_name}")
        st.write(f"**Accuracy :** {accuracy:.4f}")
        st.progress(int(accuracy * 100))  # Barre de progression pour l'accuracy

        # Affichage des scores F1 par classe
        st.write("#### Scores F1 par classe :")
        for classe, valeurs in report.items():
            if isinstance(valeurs, dict):
                f1_score = valeurs.get("f1-score", None)
            if f1_score is not None:
                st.write(f"- **Classe {classe}** : {f1_score:.4f}")

         # Affichage de la matrice de confusion avec une meilleure esthétique
        st.write("#### Matrice de confusion normalisée :")
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f"Classe {i}" for i in range(len(conf_matrix))],
                              columns=[f"Classe {i}" for i in range(len(conf_matrix))])

        # Arrondir les valeurs de la matrice de confusion
        conf_matrix_df = conf_matrix_df.round(2)  # Arrondir à 2 décimales

        fig = px.imshow(conf_matrix_df, text_auto=True, aspect="auto", title="Matrice de confusion",
                labels={'x': 'Prédictions', 'y': 'Vraies valeurs'})
        fig.update_xaxes(side="top")
        st.plotly_chart(fig)


        # Modèles de Deep Learning
        st.write('## 2. Deep Learning')
        import tensorflow as tf
        from tensorflow.keras.models import load_model


        # Dictionnaire des modèles enregistrés
        models_dict = {
        "MLP (Multi-Layer Perceptron)": "MLP.h5",
        "Modèle Séquentiel (1 couche cachée)": "deep1.h5",
        "Modèle Séquentiel (4 couches cachées)": "Deep2.h5",
        "Modèle Séquentiel (Autre fonction d'activation)": "Deep3.h5",
        "Modèle Séquentiel (Autres paramètres)": "Deep4.h5"
        }

        # Sélection du modèle via Streamlit
        selected_model_name = st.selectbox("Choisissez un modèle :", list(models_dict.keys()))

        # Charger le modèle sélectionné
        model_path = models_dict[selected_model_name]
        model = load_model(model_path)

        # Faire les prédictions
        test_pred = model.predict(X_test_scaled)
        test_pred_class = np.argmax(test_pred, axis=1)

        # Calculer les métriques
        report = classification_report(y_test, test_pred_class, output_dict=True)
        accuracy = report['accuracy']
        conf_matrix = confusion_matrix(y_test, test_pred_class, normalize='true').round(2)

        # Affichage des résultats
        st.write(f"### Résultats pour : {selected_model_name}")
        st.write(f"**Accuracy :** {accuracy:.4f}")
        st.progress(int(accuracy * 100))  # Barre de progression pour l'accuracy

        # Affichage des scores F1 par classe
        st.write("#### Scores F1 par classe :")
        for classe, valeurs in report.items():
            if isinstance(valeurs, dict):
                f1_score = valeurs.get("f1-score", None)
            if f1_score is not None:
                st.write(f"- **Classe {classe}** : {f1_score:.4f}")

        # Affichage de la matrice de confusion
        st.write("#### Matrice de confusion normalisée :")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", ax=ax, annot_kws={"size": 12})
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies valeurs")
        st.pyplot(fig)

    if 'history' in locals():  # Vérifie si l'objet 'history' existe
        st.write(f"#### Evolution de l'accuracy en fonction des epochs :")
        plt.plot(model.history['accuracy'], label='Train Accuracy')
        plt.plot(model.history['val_accuracy'], label='Test Accuracy')
        plt.title(f"Evolution de l'accuracy pour : {selected_model_name}")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        st.pyplot(plt)

if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")    