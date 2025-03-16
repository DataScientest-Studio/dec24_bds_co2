'''
README
Ce dataset est la première partie du merge entre le dataset initial et le second dataset pour obtenir le dataset mergé.
La deuxième partie se fait sur un jupyter notebook.
Initialement je voulais utiliser uniquement un seul fichier (un fichier jupyter de préférence) pour faire le merge car c'est plus visible.
Mais les données du second dataset étaient trop volumineux pout les télécharger sur jupyter.
Alors le merge commence sur ce fichier python car j'arrive à télécharger le dataset.
Puis une fois que je suis arrivé à télécharger les données, et à faire une première partie du merge,
la suite du code se fait sur 4.1-ugo-merge.ipynb
'''

###Bibliothèques
import numpy as np
import pandas as pd

df2 = pd.read_csv("data/data_co2_before_2014_all_country.csv")
df = pd.read_csv("data/mars-2014-complete.csv",sep = ';',encoding = "latin-1",on_bad_lines='skip')

#Drop duplicates dans df
df = df.drop_duplicates()

#Drop useless columns
columns_to_remove = ["ID","VFN","Cr","Mt","Ewltp (g/km)",
                     "z (Wh/km)","IT","Ernedc (g/km)","Erwltp (g/km)","De",
                     "Vf","Status","year","Date of registration","Fuel consumption ",
                     "ech","RLFI","Electric range (km)",'r']
df2 = df2.drop(columns_to_remove,axis = 1)
columns_to_remove = ["cnit","hc","hcnox","date_maj",
                     "Unnamed: 26","Unnamed: 27","Unnamed: 28","Unnamed: 29",]
df = df.drop(columns_to_remove,axis = 1)

#Drop duplicates dans df après avoir supprimer les colonnes inutiles
print(df.shape)
df = df.drop_duplicates()
print(df.shape)

#Remplacer les valeurs de Mk dans df2. Cela permet de s'assurer que les deux datasets matchent
to_replace = ["ALFA ROMEO","AUTOMOBILES CITROEN","FIAT ","Peugeot","Citroen","Dacia","DACIA AUTOMOBILE SA","Renault","Volvo","VOLVO/CARRUS"]
value = ["ALFA-ROMEO","CITROEN","FIAT","PEUGEOT","CITROEN","DACIA","DACIA","RENAULT","VOLVO","VOLVO"]
df2 = df2.replace(to_replace=to_replace, value=value)

#Split en 3 df
df2_T = df2.drop(["Va","Ve"],axis = 1).rename({'T':'tvv','Mk':'lib_mrq'},axis =1)
df2_Va = df2.drop(["T","Ve"],axis = 1).rename({'Va':'tvv','Mk':'lib_mrq'},axis =1)
df2_Ve = df2.drop(["T","Va"],axis = 1).rename({'Ve':'tvv','Mk':'lib_mrq'},axis =1)

df_merge_T = df.merge(right=df2_T,on=["tvv",'lib_mrq']).reset_index(drop=True)
df_merge_Va = df.merge(right=df2_Va,on=["tvv",'lib_mrq']).reset_index(drop=True)
df_merge_Ve = df.merge(right=df2_Ve,on=["tvv",'lib_mrq']).reset_index(drop=True)

df_merge_T = df_merge_T.drop(["tvv"], axis=1)
df_merge_Va = df_merge_Va.drop(["tvv"], axis=1)
df_merge_Ve = df_merge_Ve.drop(["tvv"], axis=1)

df_merge = pd.concat([df_merge_T,df_merge_Va,df_merge_Ve],axis=0).reset_index(drop=True)




#Suppresion des duplicates
df_merge = df_merge.drop_duplicates().reset_index(drop=True)

print(df_merge.shape)
df_merge.to_csv("data/data_merge.csv",index=False)
    

print('Fin de code')