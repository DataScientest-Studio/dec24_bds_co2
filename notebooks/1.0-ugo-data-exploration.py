'''
Readme :
Ce fichier python est principalement un brouillon utilisé pour explorer les données du premier dataset.
Les différentes lignes de ce code m'ont permis de mieux comprendre ce que comportait ce dataset.
'''

###Bibliothèques
import numpy as np
import pandas as pd
import seaborn as sns



###Get data (on récupère les données)
df = pd.read_csv("data/mars-2014-complete.csv",sep = ';',encoding = "latin-1",on_bad_lines='skip')
columns = df.columns
nbr_rows = df.shape[0]
nbr_cols = df.shape[1]
print("Nombre de lignes : ",nbr_rows,", Nombre de colonnes :",nbr_cols)
#df = pd.read_csv("data/mars-2014-complete.csv")



### Affichage informations des données
print(df.shape)
print(df.dtypes)
print(columns)
print(df.head())
print(df.info())

print("\n\n\n")
print(df[["cod_cbr","puiss_admin_98"]].head())

print("\n\n\n")
print("Pourcentage valeur nulle : ")
print(df.isnull().sum()/nbr_rows*100)



###Affichage du value_count pour chaque variable
for name in list(columns[:26]):
    print("\n")
    print("VARIABLE : ",name)
    print(df[name].value_counts())



###Etude des variables quantitatives
quantitative_col = ["puiss_admin_98","puiss_max","conso_urb","conso_exurb","conso_mixte","co_typ_1","hc","nox","hcnox","ptcl","masse_ordma_min","masse_ordma_max"]
quantitative_string_col = ["puiss_max","conso_urb","conso_exurb","conso_mixte","co_typ_1","hc","nox","hcnox","ptcl"]
df_quantitative = df[quantitative_col]

for name in quantitative_string_col:
    df_quantitative[name] = df[name].str.replace(",",".").astype("float64")
    
for name in list(quantitative_col):
    print("\n")
    print("VARIABLE : ",name)
    print(df[name].value_counts())

print(df_quantitative.describe(()))


print("Fin de code")