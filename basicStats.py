# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#importaer les tables
train=pd.read_csv("C:/Users/redni/Desktop/Nouveau dossier/Machine learning/bank-train.csv",sep=";")
test=pd.read_csv("C:/Users/redni/Desktop/Nouveau dossier/Machine learning/bank-test.csv",sep=";")

train.describe()
test.describe()

#variables quantitatives
num=[i for i in dict(train.dtypes) if dict(train.dtypes)[i] in ['float64', 'int64']]
#variables qualitatives
quali=[i for i in dict(train.dtypes) if dict(train.dtypes)[i] not in ['float64', 'int64']]

series = []
for i in quali:
    series.append(train[i].value_counts()*100/len(train))
for i in range(len(series)):
    series[i].plot(subplots=True, kind='barh',rot=0)
    plt.show()


#balance moyen par type d'habitats
balance_habitat=train.groupby('housing')['balance'].mean()

#pourcentage de la population par niveaux d'étude et profession
pop_etu_prof=train.groupby([train['job'],train['education']])
count= pop_etu_prof.size().unstack().fillna(0)*100/len(train)
count.plot(kind='barh', stacked=False)#agrandir le graphique ou mettre stacked=True

#age moyen des tyoes de couple
mean_age_couple=train.groupby('marital')['age'].mean()

#2.3 analyse univariée

#question 1:
cible= {"yes": 1,"no": 0}
train["y"] =train["y"].map(cible)
train['y']=train['y'].apply(pd.to_numeric, errors='coerce')

    #corrélation avec les explicatives quantitatives
#vu qu'il y a certaines variables numériques qvace lesquelles une corrélation n'a pas de sens, 
#je les supprime
lis=['Unnamed: 0','id','mono']
newlist = []
[newlist.append(x) for x in num if x not in lis]
coef_num=[]
pvalue_num=[]
test_num=[]
#np.delete(num,['Unnamed: 0','beginning','end','id','mono','day'])
def univar_stat():
    import scipy.stats as stats
    for j in range(len(num)):
        t="Coefficient de pearson"
        cor, pval =stats.pearsonr(train['y'], train[num[j]])
        coef_num.append(cor)
        pvalue_num.append(pval)
        test_num.append(t)           
    return pd.DataFrame(list(zip(np.transpose(num),np.transpose(coef_num),np.transpose(pvalue_num))),columns=["Variables","valeur corrélation","P-value test"])
univar_stat()

