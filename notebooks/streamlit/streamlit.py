import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("DS Octobre - Projet CO2")
st.sidebar.title('Sommaire')
pages=['Introduction', 'Exploration', 'Data Visualisation', 'Modélisation']
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



st.write("Introduction")
if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")    