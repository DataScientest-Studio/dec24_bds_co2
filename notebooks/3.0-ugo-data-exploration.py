'''
Ce fichier python est principalement un brouillon qui permet d'explorer le second dataset.
Ces lignes permettent de comprendre les lignes et colonnes de ce second dataset (avant la fusion)
'''

###Bibliothèques
import numpy as np
import pandas as pd
import seaborn as sns

###Get base de données
df = pd.read_csv("data/data_co2_before_2014_all_country.csv")
columns = df.columns
nbr_rows = df.shape[0]
nbr_cols = df.shape[1]
print("Nombre de lignes : ",nbr_rows,", Nombre de colonnes :",nbr_cols)

###Affichage informations de la base de données.
print(df.shape)
print(df.dtypes)
print(columns)
print(df.head())
print(df.info())


print("\n\n\n")
print("Pourcentage valeur nulle : ")
print(df.isnull().sum()/nbr_rows*100)

###Colonnes suprimées jugées non utiles
columns_to_remove = ["ID","VFN","Cr","Mt","Ewltp (g/km)",
                     "z (Wh/km)","IT","Ernedc (g/km)","Erwltp (g/km)","De",
                     "Vf","Status","year","Date of registration","Fuel consumption ",
                     "ech","RLFI","Electric range (km)"]
print("Before : ",df.shape)
df = df.drop(columns_to_remove,axis=1)
print("After : ",df.shape)

 ###Affichage du value_count pour chaque variable  
for name in list(df.columns):
    print("\n")
    print("VARIABLE : ",name)
    print(df[name].value_counts())

###Etude des variables quantitatives
quantitative_col = ["m (kg)","Enedc (g/km)","W (mm)","At1 (mm)","At2 (mm)","ec (cm3)","ep (KW)"]
df_quantitative = df[quantitative_col]
print(df_quantitative.describe(()))


print("Fin de code")