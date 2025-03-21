'''
README
Ce fichier python est principalement un brouillon pour tester de merger le premier et le second dataset.
J'essaye de trouver les clefs qui permettent de relier les deux datasets.
'''

###Bibliothèques
import numpy as np
import pandas as pd

df = pd.read_csv("data/data_co2_before_2014_all_country.csv")
df2 = pd.read_csv("data/mars-2014-complete.csv",sep = ';',encoding = "latin-1",on_bad_lines='skip')

print(df.shape)
print(df.columns)
print(df.head())

###Colonnes suprimées jugées non utiles
columns_to_remove = ["ID","VFN","Cr","Mt","Ewltp (g/km)",
                     "z (Wh/km)","IT","Ernedc (g/km)","Erwltp (g/km)",'r',"De",
                     "Vf","Status","year","Date of registration","Fuel consumption ",
                     "ech","RLFI","Electric range (km)"]
df = df.drop(columns_to_remove,axis=1)
df.drop_duplicates()
print(df.shape)

###On change des valeurs pour que les clefs matchent entre les 2 datasets
to_replace = ["ALFA ROMEO","AUTOMOBILES CITROEN","FIAT ","Peugeot","Citroen","Dacia","DACIA AUTOMOBILE SA","Renault","Volvo","VOLVO/CARRUS"]
value = ["ALFA-ROMEO","CITROEN","FIAT","PEUGEOT","CITROEN","DACIA","DACIA","RENAULT","VOLVO","VOLVO"]
df = df.replace(to_replace=to_replace, value=value)

###Des tests pour comprendre comment les clefs matchent

# #print(df["Country"].value_counts())
# PNCFB4 = df.loc[df["Ve"] == "PNCFB4"]
# PNCFB42 = df2.loc[df2["tvv"] == "PNCFB4"]
# columns = list(PNCFB4.columns) #['Country', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct', 'm (kg)', 'Enedc (g/km)', 'W (mm)', 'At1 (mm)', 'At2 (mm)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)']

# PNCFB4_quant = PNCFB4.drop(['Country', 'Mp', 'Mh', 'Man', 'MMS', 'Tan', 'T', 'Va', 'Ve', 'Mk', 'Cn', 'Ct',   'Ft', 'Fm'],axis=1)
# PNCFB4_quant_2 = PNCFB4_quant.drop(['m (kg)','Enedc (g/km)','ep (KW)'],axis=1)

# PNCFB4_Peugeot = df.loc[(df["Ve"] == "PNCFB4")|(df["Mk"] == "PEUGEOT")]

version = df["Ve"]
cnit = df2["cnit"]
tvv = df2["tvv"]
T = df["T"]
Va = df["Va"]

list_cnit = list(cnit)

keep_tvv = tvv.loc[tvv.isin(version)]
keep_version = version.loc[version.isin(tvv)]

df_keep_tv = df2.loc[df2["tvv"].isin(version)]
df_keep_tv2 = df2.loc[df2["tvv"].isin(T)]
df_keep_tv3 = df2.loc[df2["tvv"].isin(Va)]
df_keep_version = df.loc[df["Ve"].isin(tvv)]
keep_va = df[["T","Va","Ve"]].loc[df["Va"].isin(tvv)]
keep_T = df[["T","Va","Ve"]].loc[df["T"].isin(tvv)]


print('keep_tvv.shape : ',keep_tvv.shape)
print('keep_version.shape : ',keep_version.shape)

keep_tvv.value_counts()
keep_version.value_counts()
df_keep_tv["lib_mrq"].value_counts()
df_keep_version["Mh"].value_counts()

keep_marque = list(df_keep_tv["lib_mrq"].value_counts().index)
keep_marque = keep_marque[:-3] + [keep_marque[-2]]

#IMPORTANT : plutôt regarder Mk que Mh
to_replace = ["AUTOMOBILES CITROEN","AUTOMOBILES PEUGEOT","FIAT GROUP AUTOMOBILES SPA"]
value = ["CITROEN","PEUGEOT","FIAT"]

df_keep_version = df_keep_version.replace(to_replace=to_replace,value=value)

df_keep_tv = df_keep_tv.loc[df_keep_tv["lib_mrq"].isin(keep_marque)]
df_keep_version = df_keep_version.loc[df_keep_version["Mh"].isin(keep_marque)]

PNCFB4 = df.loc[df["Ve"] == "PNCFB4"]
PNCFB42 = df2.loc[df2["tvv"] == "PNCFB4"]

PMCFB4 = df.loc[df["Ve"] == "PMCFB4"]
PMCFB42 = df2.loc[df2["tvv"] == "PMCFB4"]

les5R0G0H = df.loc[df["Ve"] == "5R0G0H"]
les5R0G0H2 = df2.loc[df2["tvv"] == "5R0G0H"]

#keep_M = version.loc[version.str[0] == 'M'].reset_index(drop=True)

# for i in range(100):
#     print(i,' : ',keep_M[i])

for name in list(df.columns):
    print("\n\n")
    print('Nom de la variable : ',name)
    print(PNCFB4[name].value_counts())

# for name in list(df2.columns):
#     print("\n\n")
#     print('Nom de la variable : ',name)
#     print(PMCFB42[name].value_counts())

print("Fin du code")