import tkinter as tk
from bs4 import BeautifulSoup
from urllib.request import urlopen
from PIL import Image, ImageTk
import webbrowser
import os
from owlready2 import *
from owlready2 import get_ontology, sync_reasoner_pellet
import requests
from datetime import datetime
from collections import Counter
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdflib import Namespace, Graph, RDF, Literal, XSD
from sklearn.ensemble import RandomForestRegressor
from rdflib.plugins.sparql import prepareQuery
from decimal import Decimal
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
from imblearn.over_sampling import SMOTE
from urllib.parse import quote
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import numpy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score



Dim=200
Dim2=205
Dim3=1576
Dim4=1965
Dim5=2000
Dim6=2036
Dim7=166
Dim8=162
Dim9=170
Dim10=168
Dim11=159
Dim12=159
Dim13=196
Dim14=203
Dim15=200
DimLogica=3169

Voto = [i*2 for i in range(1, Dim+1)]
Soldi_spesi = [i*2 for i in range(1, Dim7+1)]
Filmetto7 = [i*2 for i in range(1, Dim7+1)]
Soldi_spesi2 = [i*2 for i in range(1, Dim8+1)]
Filmetto8 = [i*2 for i in range(1, Dim8+1)]
Soldi_spesi3 = [i*2 for i in range(1, Dim9+1)]
Filmetto9 = [i*2 for i in range(1, Dim9+1)]
Soldi_spesi4 = [i*2 for i in range(1, Dim10+1)]
Filmetto10 = [i*2 for i in range(1, Dim10+1)]
Soldi_spesi5 = [i*2 for i in range(1, Dim11+1)]
Filmetto11 = [i*2 for i in range(1, Dim11+1)]
Soldi_spesi6 = [i*2 for i in range(1, Dim12+1)]
Filmetto12 = [i*2 for i in range(1, Dim12+1)]
Filmetto = [i*2 for i in range(1, Dim+1)]
Filmetto2 = [i*2 for i in range(1, Dim2+1)]
Voto2 = [i*2 for i in range(1, Dim2+1)]
Filmetto3 = [i*2 for i in range(1, Dim3+1)]
Voto3 = [i*2 for i in range(1, Dim3+1)]
Filmetto4 = [i*2 for i in range(1, Dim4+1)]
Voto4 = [i*2 for i in range(1, Dim4+1)]
Filmetto5 = [i*2 for i in range(1, Dim5+1)]
Voto5 = [i*2 for i in range(1, Dim5+1)]
Filmetto6 = [i*2 for i in range(1, Dim6+1)]
Voto6 = [i*2 for i in range(1, Dim6+1)]

Soldi_Ottenuti = [i*2 for i in range(1, Dim13+1)]
Filmetto13 = [i*2 for i in range(1, Dim13+1)]
Soldi_Ottenuti2 = [i*2 for i in range(1, Dim14+1)]
Filmetto14 = [i*2 for i in range(1, Dim14+1)]
Soldi_Ottenuti3 = [i*2 for i in range(1, Dim3+1)]
Filmetto15 = [i*2 for i in range(1, Dim3+1)]
Soldi_Ottenuti4 = [i*2 for i in range(1, Dim4+1)]
Filmetto16 = [i*2 for i in range(1, Dim4+1)]
Soldi_Ottenuti5 = [i*2 for i in range(1, Dim5+1)]
Filmetto17 = [i*2 for i in range(1, Dim5+1)]
Soldi_Ottenuti6 = [i*2 for i in range(1, Dim6+1)]
Filmetto18 = [i*2 for i in range(1, Dim6+1)]

Eta = []  # Lista per i valori di media_rating
Filmetto19 = []  # Lista per i valori di mese
Eta2 = []  # Lista per i valori di media_rating
Eta3 = []  # Lista per i valori di media_rating
Eta4 = []  # Lista per i valori di media_rating
Eta5 = []  # Lista per i valori di media_rating
Eta6 = []  # Lista per i valori di media_rating

Gener = []
Gener2 = []
Gener3 = []
Gener4 = []
Gener5 = []
Gener6 = []
ListaLogisticaScore = [i*2 for i in range(1, DimLogica+1)]
ListaLogisticaGuadagno = [i*2 for i in range(1, DimLogica+1)]

Nazion = []
Nazion2 = []
Nazion3 = []
Nazion4 = []
Nazion5 = []
Nazion6 = []

df = pd.read_csv('venv/movies.csv')

# Sostituisci 'nome_colonna' con il nome della colonna che vuoi stampare
nome = 'name'
# Verifica se la colonna esiste nel DataFrame
if nome in df.columns:
    # Stampa la colonna desiderata
    print(df[nome])
print(df.loc[5,nome])

onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto = get_ontology("http://www.w3.org/2002/07/Film2000")
onto.save(file="Film2000.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2000.owl"
into = get_ontology(file_path).load()
i=0
formato_data = "%B %d %Y"
while(i<=7667):
    try:

       rilaa = str(df.loc[i,"released"])
       rilaa = rilaa.replace(",", "")
       parti = rilaa.split(" (")
       nuova_stringa = parti[0]
       datta = datetime.strptime(nuova_stringa, formato_data)
       df.loc[i,"released"] = datta
       i=i+1
    except:
        df.loc[i, "released"] = datetime.strptime("May 12 2021", formato_data)
        i=i+1


df.sort_values(by='released', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
print(df[nome])
print(df.loc[1,nome])
print(df.loc[245])
i=0
while(i<7668):
  if ("2000" in str(df.loc[i,"released"])):
   print(df.loc[i,"name"])
  if ("2001" in str(df.loc[i,"released"])):
    i=7777
  i = i + 1
i=0
with onto:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [datetime]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0
formato_data = "%B %d %Y"
while(i<=7667):
  if ("2000" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilaa=str(df.loc[i,rilasc])
    rilaa=rilaa.replace(",","")
    parti = rilaa.split(" (")
    nuova_stringa = parti[0]
    #datta=datetime.strptime(nuova_stringa, formato_data)
    rilascio = rilaa
    try:
        score = float(df.loc[i,scor])
    except:
        score=0
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    durata = int(df.loc[i,durat])
    with onto:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto.save()
  else:
      if("2001" in str(df.loc[i,rilasc])):
          i=7667
          print("fiuert")
  i=i+1



#individui_con_date = [(individuo, individuo.Rilascio) for individuo in onto.Film.instances()]

# Ordina gli individui in base alle date
#individui_ordinati = sorted(individui_con_date, key=lambda x: x[1])

# Aggiorna l'ontologia con l'ordine desiderato
#for i, (individuo, _) in enumerate(individui_ordinati):
    #individuo.priority = i
#print(individui_ordinati)
# Sincronizza il reasoner (pellet) per riflettere le modifiche
#sync_reasoner_pellet()

# Salva l'ontologia nel file .owl
#onto.save()



i=0
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto2 = get_ontology("http://www.w3.org/2002/07/Film2001")
onto2.save(file="Film2001.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2001.owl"
into = get_ontology(file_path).load()


with onto2:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [str]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0
while(i<=7667):
  if ("2001" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilascio = df.loc[i,rilasc]
    score = float(df.loc[i,scor])
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    durata = int(df.loc[i,durat])
    with onto2:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto2.save()
  else:
      if("2002" in str(df.loc[i,rilasc])):
          i=7667
  i=i+1

i=0

#onto_path.append("file://C:/Users/Vanni/PycharmProjects/pythonProject")

#onto80 = get_ontology("file://C:/Users/Vanni/PycharmProjects/pythonProject/FilmAnni80.owl")
#onto80.load()


#cax
#http://www.w3.org/2002/07/FilmAnni80





#file_path = "FilmAnni80.owl"
#onto80 = get_ontology(file_path).load()





#onto80 = get_ontology("http://www.w3.org/2002/07/FilmAnni80.owl").load()
#with onto80:
    #sync_reasoner()
#with onto80:
    #sync_reasoner_pellet(debug=0)
#explanation = onto80.inconsistent_classes()

# Stampare la spiegazione
#print("Spiegazione dell'inconsistenza:")
#for explanation_line in explanation:
    #print(explanation_line)


# Esegui query SPARQL sull'ontologia
# Ad esempio, trova tutti i film di un determinato genere

from rdflib.plugins.sparql import prepareQuery

query = '''
    PREFIX ns: <http://www.w3.org/2002/07/FilmAnni80#>
    SELECT ?film ?genere
    WHERE {
        ?film a onto:Film .
        ?film onto:Genere ?genere .
        FILTER regex(str(?genere), "Action")
    }
'''
g2 = Graph()
g2.parse("FilmAnni90.owl")
g = Graph()
g.parse("FilmAnni80.owl")
g3 = Graph()
g3.parse("FilmAnni2000.owl")
g4 = Graph()
g4.parse("FilmAnni2010.owl")




"""
query2 = prepareQuery('''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni80#>
    SELECT ?film ?genere ?score ?filmdisuccesso_score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Genere ?genere .
        ?film ns:Score ?score .
        ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
        FILTER regex(str(?genere), "Action")
    }
    GROUP BY ?film
''',

initNs={'ns': ns, 'rdf': RDF})
"""





queryA = '''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni80#>
    INSERT {
    ?s ns:FilmDiSuccesso_Score ?result .
     }
     WHERE {
    ?s ns:Score ?score .
    BIND(IF(?score > 7, true, false) AS ?result)
    }
    '''







#ns=Namespace('http://www.w3.org/2002/07/FilmAnni2010')
"""
query6 = prepareQuery('''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2010#>
    SELECT ?film ?genere ?score ?filmdisuccesso_score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Genere ?genere .
        ?film ns:Score ?score .
        ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
        FILTER regex(str(?genere), "Action")
    }
    GROUP BY ?film
''',

initNs={'ns': ns, 'rdf': RDF})
"""


queryF = '''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2010#>
    INSERT {
    ?film ns:FilmDiSuccesso_Guadagno ?result .
}
WHERE {
    ?film ns:Budget ?budget .
    ?film ns:Guadagno ?guadagno .
    FILTER(?budget > 0 && ?guadagno > 0)
    BIND(IF(?guadagno >= (?budget + 0.5 * ?budget), true, false) AS ?result)
}
    '''





"""
query5 = prepareQuery('''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2000#>
    SELECT ?film ?genere ?score ?filmdisuccesso_score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Genere ?genere .
        ?film ns:Score ?score .
        ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
        FILTER regex(str(?genere), "Action")
    }
    GROUP BY ?film
''',

initNs={'ns': ns, 'rdf': RDF})
"""


queryD = '''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2010#>
    INSERT {
    ?s ns:FilmDiSuccesso_Score ?result .
     }
     WHERE {
    ?s ns:Score ?score .
    BIND(IF(?score > 7, true, false) AS ?result)
    }
    '''




queryG = '''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni90#>
    INSERT {
    ?film ns:FilmDiSuccesso_Guadagno ?result .
}
WHERE {
    ?film ns:Budget ?budget .
    ?film ns:Guadagno ?guadagno .
    FILTER(?budget > 0 && ?guadagno > 0)
    BIND(IF(?guadagno >= (?budget + 0.5 * ?budget), true, false) AS ?result)
}
    '''

ns = Namespace('http://www.w3.org/2002/07/FilmAnni2010')

query12 =  prepareQuery('''
PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2010#>

SELECT ?genere (COUNT(?film) AS ?numFilm) ((COUNT(?filmConSuccessoScore) / COUNT(?filmConScore) * 100) AS ?percentualeSuccesso)
WHERE {
    ?film ns:Genere ?genere .
     FILTER EXISTS {
        ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
    }
    
    OPTIONAL {
        ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
        FILTER (?filmdisuccesso_score = true)
        BIND(1 AS ?filmConSuccessoScore)
    }
    BIND(1 AS ?filmConScore)
}
GROUP BY ?genere
''',

initNs={'ns': ns, 'rdf': RDF})


results = g4.query(query12)
for row in results:
    print("Genere:", row.genere)
    print("Numero di film:", row.numFilm)
    print("Percentuale di successo:", row.percentualeSuccesso)
    print()

for row in results:
    print(row)

ns = Namespace('http://www.w3.org/2002/07/FilmAnni2010')
query20 = prepareQuery('''PREFIX ns: <http://www.w3.org/2002/07/FilmAnni2010#>

SELECT ?film ?movie ?genere ?rating ?budget ?guadagno ?filmdisuccesso_guadagno ?score ?filmdisuccesso_score ?percentualeSuccesso
WHERE {
    ?film ns:Movie ?movie .
    ?film ns:Genere ?genere .
    ?film ns:Rating ?rating .
    ?film ns:Budget ?budget .
    ?film ns:Guadagno ?guadagno .
    ?film ns:FilmDiSuccesso_Guadagno ?filmdisuccesso_guadagno .
    ?film ns:Score ?score .
    ?film ns:FilmDiSuccesso_Score ?filmdisuccesso_score .
}
ORDER BY ASC(?score)
LIMIT 1''',

initNs={'ns': ns, 'rdf': RDF})

results = g4.query(query20)
for row in results:
    print("Film: ",row.movie)
    print("Genere: ", row.genere)
    print("Rating: ", row.rating)
    print("Budget: ", row.budget)
    print("Guadagno: ", row.guadagno)
    print("Film di Successo Guadagno: ", row.filmdisuccesso_guadagno)
    print("Score: ", row.score)
    print("Film di Successo Score: ", row.filmdisuccesso_score)
'''
g4.update(queryG)
g4.serialize("FilmAnni2010.owl", format="xml")
'''
# Stampa i risultati




'''
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto3 = get_ontology("http://www.w3.org/2002/07/FilmAnni80")
onto3.save(file="FilmAnni80.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/FilmAnni80.owl"
into = get_ontology(file_path).load()


with onto3:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [str]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0
df.dropna(inplace=True)
while(i<=7667):
 try:
  if ("198" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilascio = df.loc[i,rilasc]
    score = float(df.loc[i,scor])
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    durata = int(df.loc[i,durat])
    with onto3:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto3.save()
  else:
    if("2003" in str(df.loc[i,rilasc])):
        i=7667
 except:
     i=i
 i=i+1
'''
i=0
'''
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto4 = get_ontology("http://www.w3.org/2002/07/FilmAnni90")
onto4.save(file="FilmAnni90.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/FilmAnni90.owl"
into = get_ontology(file_path).load()


with onto4:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [str]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
budg='budget'
guad='gross'
compa='company'
durat='runtime'
'''
i=0
'''
while(i<=7667):
 try:
  if ("199" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilascio = df.loc[i,rilasc]
    score = float(df.loc[i,scor])
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    durata = int(df.loc[i,durat])
    with onto4:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto4.save()
  else:
      if("2020" in str(df.loc[i,rilasc])):
          i=7667
 except:
  i = i
 i=i+1
'''

i=0
'''
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto5 = get_ontology("http://www.w3.org/2002/07/FilmAnni2000")
onto5.save(file="FilmAnni2000.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/FilmAnni2000.owl"
into = get_ontology(file_path).load()


with onto5:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [str]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
'''
print("provina")
i=0
ff=df.dropna()
print(type(df.iloc[i]["released"]))
df_filtered = df[(df['year'] >= 2000) & (df['year'] <= 2020)]
df_2000 = ff[ff['year'] == 2000]
df_2001 = df[(df['year'] == 2001)]
df_2002 = df[(df['year'] == 2002)]
df_2003 = df[(df['year'] == 2003)]
df_2004 = df[(df['year'] == 2004)]
df_2005 = df[(df['year'] == 2005)]
df_2006 = df[(df['year'] == 2006)]
df_2007 = df[(df['year'] == 2007)]
df_2008 = df[(df['year'] == 2008)]
df_2009 = df[(df['year'] == 2009)]
df_2010 = df[(df['year'] == 2010)]
df_2001=df_2001.dropna()
df_2002=df_2002.dropna()
df_2003=df_2003.dropna()
df_2004=df_2004.dropna()
df_2005=df_2005.dropna()
df_2006=df_2006.dropna()
df_2007=df_2007.dropna()
df_2008=df_2008.dropna()
df_2009=df_2009.dropna()
df_2010=df_2010.dropna()


media2000=0
media2001=0
media2002=0
media2003=0
media2004=0
media2005=0
media2006=0
media2007=0
media2008=0
media2009=0
media2010=0


while(i<len(df_2000)):
    media2000=media2000+df_2000.iloc[i]["score"]
    i=i+1
i=0
media2000=media2000/len(df_2000)
print(media2000)

while (i < len(df_2001)):
    media2001 = media2001 + df_2001.iloc[i]["score"]
    i = i + 1
i = 0
media2001 = media2001 / len(df_2001)
print(media2001)

while (i < len(df_2002)):
    media2002 = media2002 + df_2002.iloc[i]["score"]
    i = i + 1
i = 0
media2002 = media2002 / len(df_2002)
print(media2002)

while (i < len(df_2003)):
    media2003 = media2003 + df_2003.iloc[i]["score"]
    i = i + 1
i = 0
media2003 = media2003 / len(df_2003)
print(media2003)

while (i < len(df_2004)):
    media2004 = media2004 + df_2004.iloc[i]["score"]
    i = i + 1
i = 0
media2004 = media2004 / len(df_2004)
print(media2004)

while (i < len(df_2005)):
    media2005 = media2005 + df_2005.iloc[i]["score"]
    i = i + 1
i = 0
media2005 = media2005 / len(df_2005)
print(media2005)

while (i < len(df_2005)):
    media2005 = media2005 + df_2005.iloc[i]["score"]
    i = i + 1
i = 0
media2005 = media2005 / len(df_2005)
print(media2005)

while (i < len(df_2006)):
    media2006 = media2006 + df_2006.iloc[i]["score"]
    i = i + 1
i = 0
media2006 = media2006 / len(df_2006)
print(media2006)

while (i < len(df_2007)):
    media2007 = media2007 + df_2007.iloc[i]["score"]
    i = i + 1
i = 0
media2007 = media2007 / len(df_2007)
print(media2007)

while (i < len(df_2008)):
    media2008 = media2008 + df_2008.iloc[i]["score"]
    i = i + 1
i = 0
media2008 = media2008 / len(df_2008)
print(media2008)


while (i < len(df_2009)):
    media2009 = media2009 + df_2009.iloc[i]["score"]
    i = i + 1
i = 0
media2009 = media2009 / len(df_2009)
print(media2009)


while (i < len(df_2010)):
    media2010 = media2010 + df_2010.iloc[i]["score"]
    i = i + 1
i = 0
media2010 = media2010 / len(df_2010)
print(media2010)

dfscore = df_filtered.dropna()
dfbudget = df_filtered.dropna()
dfgross=df_filtered.dropna()# Elimina i valori mancanti dalla colonna "score"
print(len(dfscore))
while (i < 3169):
    if (dfscore.iloc[i]["score"] >= 7.0):
        ListaLogisticaScore[i] = 1
    else:
        ListaLogisticaScore[i] = 0
    i = i + 1

i=0
while (i < 3169):
    if (dfgross.iloc[i]["gross"] >= dfbudget.iloc[i]["budget"] + dfbudget.iloc[i]["budget"]/2):
        ListaLogisticaGuadagno[i] = 1
    else:
        ListaLogisticaGuadagno[i] = 0
    i = i + 1
print(ListaLogisticaScore)
print(ListaLogisticaGuadagno)
print(dfscore.iloc[0]["name"], dfscore.iloc[7]["score"])
print(dfscore.iloc[6]["name"], dfscore.iloc[6]["score"])
print(dfscore.iloc[2]["name"], dfscore.iloc[2]["budget"])
print(dfscore.iloc[5]["name"], dfscore.iloc[5]["gross"])

print(len(dfscore))
print("sv")
print(df[(df['year'] >= 2000) & (df['year'] <= 2010)].dropna()['score'])
print(df[(df['year'] >= 2000) & (df['year'] <= 2010)].dropna()['gross'])
print(df[(df['year'] >= 2000) & (df['year'] <= 2010)].dropna()['budget'])
'''
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0

while(i<=7667):
 try:
  if ("200" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilascio = df.loc[i,rilasc]
    score = float(df.loc[i,scor])
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    try:
      durata = int(df.loc[i,durat])
    except:
        durata=0
    with onto5:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto5.save()
  else:
      if("2023" in str(df.loc[i,rilasc])):
          i=7667
 except:
     i=i
 i=i+1
'''
i=0
'''
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto6 = get_ontology("http://www.w3.org/2002/07/FilmAnni2010")
onto6.save(file="FilmAnni2010.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/FilmAnni2010.owl"
into = get_ontology(file_path).load()


with onto6:
    class Film(Thing):
        pass

    class Movie(Property):
        domain = [Film]
        range = [str]

    class Rating(Property):
        domain = [Film]
        range = [str]
        multiple = True

    class Anno(Property):
        domain = [Film]
        range = [int]

    class Genere(Property):
        domain = [Film]
        range = [str]

    class Rilascio(Property):
        domain = [Film]
        range = [str]

    class Score(Property):
        domain = [Film]
        range = [float]

    class Nazione(Property):
        domain = [Film]
        range = [str]

    class Budget(Property):
        domain = [Film]
        range = [int]

    class Guadagno(Property):
        domain = [Film]
        range = [int]

    class Compagnia(Property):
        domain = [Film]
        range = [str]

    class Durata(Property):
        domain = [Film]
        range = [int]
ratin='rating'
gener='genre'
ann='year'
rilasc='released'
scor='score'
naz='country'
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0
while(i<=7667):
 try:
  if ("201" in str(df.loc[i,rilasc])):
    movie=df.loc[i,nome]
    genere=df.loc[i,gener]
    rating = df.loc[i,ratin]
    anno = int(df.loc[i,ann])
    rilascio = df.loc[i,rilasc]
    score = float(df.loc[i,scor])
    nazione = df.loc[i,naz]
    try:
       budget=int(df.loc[i,budg])
    except:
       budget=0
    try:
        guadagno = int(df.loc[i,guad])
    except:
        guadagno=0
    compagnia = df.loc[i,compa]
    try:
      durata =int(df.loc[i,durat])
    except:
      durata =0

    with onto6:
        film = Film()
        film.Movie.append(movie)
        film.Movie = [movie]
        film.Genere = [genere]
        film.Rating = [rating]
        film.Anno = [anno]
        film.Rilascio = [rilascio]
        film.Score = [score]
        film.Nazione = [nazione]
        film.Budget = [budget]
        film.Guadagno = [guadagno]
        film.Compagnia = [compagnia]
        film.Durata = [durata]
        film.Movie.append(movie)
        film.Genere.append(genere)
        film.Rating.append(rating)
        film.Anno.append(anno)
        film.Rilascio.append(rilascio)
        film.Score.append(score)
        film.Nazione.append(nazione)
        film.Budget.append(budget)
        film.Guadagno.append(guadagno)
        film.Compagnia.append(compagnia)
        film.Durata.append(durata)
    onto6.save()
  else:
      if("2020" in str(df.loc[i,rilasc])):
          i=7667
 except:
  i=i
 i=i+1
'''
'''
  #for individuo in onto.individuals():
      #for prop in individuo.get_properties():
          #for value in prop[individuo]:

#onto_path = "Film2000.owl"  # Sostituisci con il percorso reale del tuo file .owl
#onto = get_ontology("file://" + onto_path).load()

# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2000")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2000#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto[i]=float(media_score)
        Filmetto[i]=datetime.strptime(mese, "%Y-%m-%d %H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/200
print(media)
print(Voto)
print(Filmetto)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto[0:], Voto, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2000)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()






i=0
ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma

somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto2[i]=float(media_score)
        Filmetto2[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/205
print(media)
print(Voto2)
print(Filmetto2)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()
print(Voto2)
print(Filmetto2)
i=0
film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto2[0:], Voto2, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2001)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()






i=0
ns = Namespace("http://www.w3.org/2002/07/Film2002")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma

somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2002#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto3[i]=float(media_score)
        Filmetto3[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/193
print(media)
print(Voto3)
print(Filmetto3)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()
print(Voto3)
print(Filmetto3)
i=0
film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto3[0:], Voto3, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2002)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()






i=0
ns = Namespace("http://www.w3.org/2002/07/Film2003")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma

somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2003#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto4[i]=float(media_score)
        Filmetto4[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/207
print(media)
print(Voto4)
print(Filmetto4)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()
print(Voto4)
print(Filmetto4)
i=0
film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto4[0:], Voto4, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2003)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()






i=0
ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma

somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto5[i]=float(media_score)
        Filmetto5[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/191
print(media)
print(Voto5)
print(Filmetto5)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()
print(Voto5)
print(Filmetto5)
i=0
film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto5[0:], Voto5, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2004)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()






i=0
ns = Namespace("http://www.w3.org/2002/07/Film2005")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma

somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_score():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2005#>
    SELECT ?film (AVG(?score) as ?media_score) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        film = risultato['film']
        mese = risultato['rilascio']
        media_score = risultato['media_score']
        somma=somma + media_score
        print(f"Film: {film} - Media Score: {media_score}")
        Voto6[i]=float(media_score)
        Filmetto6[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
        i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_score()
media=float(somma)/200
print(media)
print(Voto6)
print(Filmetto6)



def ottieni_scores():
    query = """
    SELECT ?film ?score
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Score ?score .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, score in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()
print(Voto6)
print(Filmetto6)
i=0
film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto6[0:], Voto6, marker='o')
plt.title("Aumento/Diminuzione dello Score")
plt.xlabel("Film (2005)")
plt.ylabel("Differenza Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()




# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2000")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2000#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi[i]=float(media_budget)
           Filmetto7[i]=datetime.strptime(mese, "%Y-%m-%d %H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/166
print(media)
print(Soldi_spesi)
print(Filmetto7)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto7[0:], Soldi_spesi, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2000)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi2[i]=float(media_budget)
           Filmetto8[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/162
print(media)
print(Soldi_spesi2)
print(Filmetto8)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto8[0:], Soldi_spesi2, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2001)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2002")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2002#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi3[i]=float(media_budget)
           Filmetto9[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/170
print(media)
print(Soldi_spesi3)
print(Filmetto9)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto9[0:], Soldi_spesi3, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2002)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2003")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2003#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi4[i]=float(media_budget)
           Filmetto10[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/168
print(media)
print(Soldi_spesi4)
print(Filmetto10)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto10[0:], Soldi_spesi4, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2003)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi5[i]=float(media_budget)
           Filmetto11[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/159
print(media)
print(Soldi_spesi5)
print(Filmetto11)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto11[0:], Soldi_spesi5, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2004)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





# Definisci il namespace
ns = Namespace("http://www.w3.org/2002/07/Film2005")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_budget():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2005#>
    SELECT ?film (AVG(?budget) as ?media_budget) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_budget'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_budget = risultato['media_budget']
           somma=somma + media_score
           print(f"Film: {film} - Media Budget: {media_budget}")
           Soldi_spesi6[i]=float(media_budget)
           Filmetto12[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_budget()
media=float(somma)/159
print(media)
print(Soldi_spesi6)
print(Filmetto12)



def ottieni_scores():
    query = """
    SELECT ?film ?budget
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Budget ?budget .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto12[0:], Soldi_spesi6, marker='o')
plt.title("Aumento/Diminuzione del Budget")
plt.xlabel("Film (2005)")
plt.ylabel("Differenza Budget")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()




ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_guadagno():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
    SELECT ?film (AVG(?guadagno) as ?media_guadagno) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_guadagno'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_guadagno = risultato['media_guadagno']
           somma=somma + media_score
           print(f"Film: {film} - Media Guadagno: {media_guadagno}")
           Soldi_Ottenuti2[i]=float(media_guadagno)
           Filmetto14[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_guadagno()
media=float(somma)/203
print(media)
print(Soldi_Ottenuti2)
print(Filmetto14)



def ottieni_scores():
    query = """
    SELECT ?film ?guadagno
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto14[0:], Soldi_Ottenuti2, marker='o')
plt.title("Aumento/Diminuzione del Guadagno")
plt.xlabel("Film (2001)")
plt.ylabel("Differenza Guadagno")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





ns = Namespace("http://www.w3.org/2002/07/Film2002")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_guadagno():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2002#>
    SELECT ?film (AVG(?guadagno) as ?media_guadagno) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_guadagno'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_guadagno = risultato['media_guadagno']
           somma=somma + media_score
           print(f"Film: {film} - Media Guadagno: {media_guadagno}")
           Soldi_Ottenuti3[i]=float(media_guadagno)
           Filmetto15[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_guadagno()
media=float(somma)/193
print(media)
print(Soldi_Ottenuti3)
print(Filmetto15)



def ottieni_scores():
    query = """
    SELECT ?film ?guadagno
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto15[0:], Soldi_Ottenuti3, marker='o')
plt.title("Aumento/Diminuzione del Guadagno")
plt.xlabel("Film (2002)")
plt.ylabel("Differenza Guadagno")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





ns = Namespace("http://www.w3.org/2002/07/Film2003")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_guadagno():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2003#>
    SELECT ?film (AVG(?guadagno) as ?media_guadagno) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_guadagno'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_guadagno = risultato['media_guadagno']
           somma=somma + media_score
           print(f"Film: {film} - Media Guadagno: {media_guadagno}")
           Soldi_Ottenuti4[i]=float(media_guadagno)
           Filmetto16[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_guadagno()
media=float(somma)/207
print(media)
print(Soldi_Ottenuti4)
print(Filmetto16)



def ottieni_scores():
    query = """
    SELECT ?film ?guadagno
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto16[0:], Soldi_Ottenuti4, marker='o')
plt.title("Aumento/Diminuzione del Guadagno")
plt.xlabel("Film (2003)")
plt.ylabel("Differenza Guadagno")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_guadagno():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
    SELECT ?film (AVG(?guadagno) as ?media_guadagno) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_guadagno'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_guadagno = risultato['media_guadagno']
           somma=somma + media_score
           print(f"Film: {film} - Media Guadagno: {media_guadagno}")
           Soldi_Ottenuti5[i]=float(media_guadagno)
           Filmetto17[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_guadagno()
media=float(somma)/191
print(media)
print(Soldi_Ottenuti5)
print(Filmetto17)



def ottieni_scores():
    query = """
    SELECT ?film ?guadagno
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto17[0:], Soldi_Ottenuti5, marker='o')
plt.title("Aumento/Diminuzione del Guadagno")
plt.xlabel("Film (2004)")
plt.ylabel("Differenza Guadagno")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





ns = Namespace("http://www.w3.org/2002/07/Film2005#")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

somm=0
#global somma
somma=Literal(somm)
# Funzione per calcolare la media degli score
def calcola_media_guadagno():
    i=0
    global somma
    global media_score
    query = """
    PREFIX ns: <http://www.w3.org/2002/07/Film2005#>
    SELECT ?film (AVG(?guadagno) as ?media_guadagno) ?rilascio
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
        ?film ns:Rilascio ?rilascio .
    }
    GROUP BY ?film
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})


    for risultato in risultati:
        if(float(risultato['media_guadagno'])>10):

           film = risultato['film']
           mese = risultato['rilascio']
           media_guadagno = risultato['media_guadagno']
           somma=somma + media_score
           print(f"Film: {film} - Media Guadagno: {media_guadagno}")
           Soldi_Ottenuti6[i]=float(media_guadagno)
           Filmetto18[i] = datetime.strptime(mese, "%Y-%m-%dT%H:%M:%S")
           i=i+1
    #return media_score



# Esegui la funzione per calcolare la media degli score
calcola_media_guadagno()
media=float(somma)/200
print(media)
print(Soldi_Ottenuti6)
print(Filmetto18)



def ottieni_scores():
    query = """
    SELECT ?film ?guadagno
    WHERE {
        ?film rdf:type ns:Film .
        ?film ns:Guadagno ?guadagno .
    }
    """
    risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})
    scores = {film: float(score) for film, budge in risultati}
    return scores

# Esegui la funzione per ottenere gli score
scores = ottieni_scores()


film_sorted = sorted(scores.keys())
score_sorted = [scores[film] for film in film_sorted]

# Calcola le differenze tra score consecutivi
differenze_score = [score_sorted[i+1] - score_sorted[i] for i in range(len(score_sorted)-1)]

# Rappresenta il grafico a linee delle differenze
plt.plot(Filmetto18[0:], Soldi_Ottenuti6, marker='o')
plt.title("Aumento/Diminuzione del Guadagno")
plt.xlabel("Film (2005)")
plt.ylabel("Differenza Guadagno")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()





ns = Namespace("http://www.w3.org/2002/07/Film2000")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2000#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta.append("Not Rated")
    else:
        Eta.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2000)")







ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta2.append("Not Rated")
    else:
        Eta2.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta2)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2001)")


ns = Namespace("http://www.w3.org/2002/07/Film2002")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2002#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta3.append("Not Rated")
    else:
        Eta3.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta3)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2002)")


ns = Namespace("http://www.w3.org/2002/07/Film2003")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2003#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta4.append("Not Rated")
    else:
        Eta4.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta4)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2003)")


ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta5.append("Not Rated")
    else:
        Eta5.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta5)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2004)")


ns = Namespace("http://www.w3.org/2002/07/Film2005")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

#somm=0
#global somma
#somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2005#>
SELECT ?rating
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Rating ?rating .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['rating']
    #Eta.append(str(rating))
    if(str(rating)=="None" or str(rating)=="Unrated"):
        Eta6.append("Not Rated")
    else:
        Eta6.append(str(rating))



# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Eta6)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Ratings (Film anni 2005)")





ns = Namespace("http://www.w3.org/2002/07/Film2000")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# somm=0
# global somma
# somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2000#>
SELECT ?genere
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Genere ?genere .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['genere']
    # Eta.append(str(rating))
    if (str(rating) == "None" or str(rating) == "Unrated"):
        Gener.append("Not Rated")
    else:
        Gener.append(str(rating))

# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Gener)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta


ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# somm=0
# global somma
# somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
SELECT ?genere
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Genere ?genere .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['genere']
    # Eta.append(str(rating))
    if (str(rating) == "None" or str(rating) == "Unrated"):
        Gener2.append("Not Rated")
    else:
        Gener2.append(str(rating))

# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Gener2)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta


ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# somm=0
# global somma
# somma=Literal(somm)
# Funzione per calcolare la media degli score
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
SELECT ?genere
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Genere ?genere .
}
"""
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai i rating e aggiungili alla lista
for risultato in risultati:
    rating = risultato['genere']
    # Eta.append(str(rating))
    if (str(rating) == "None" or str(rating) == "Unrated"):
        Gener5.append("Not Rated")
    else:
        Gener5.append(str(rating))

# Calcola la frequenza di ciascun rating
conteggio_rating = Counter(Gener5)

# Prepara i dati per il grafico a torta
labels = conteggio_rating.keys()
sizes = conteggio_rating.values()

# Crea il grafico a torta




'''
print("encon")
label_encoder = LabelEncoder()
df_filtered=df_filtered.dropna()
df_filtered['genre-encoded'] = label_encoder.fit_transform(df_filtered['genre'])
print(df_filtered['genre-encoded'])
print(df_filtered['genre'])
df_filtered['rating-encoded'] = label_encoder.fit_transform(df_filtered['rating'])
print(df_filtered['rating-encoded'])
df_filtered.reset_index(drop=True)
print("--.--")
print(numpy.unique(numpy.array(df_filtered['genre-encoded'])))
print(numpy.unique(numpy.array(df_filtered['genre'])))
print(df_filtered.iloc[2]['genre-encoded'])
print(df_filtered.iloc[2]['genre'])
print(df_filtered.iloc[3]['genre-encoded'])
print(df_filtered.iloc[3]['genre'])
print(df_filtered.iloc[4]['genre-encoded'])
print(df_filtered.iloc[4]['genre'])
print(",.,.,")
print(df_filtered.iloc[22]['genre-encoded'])
print(df_filtered.iloc[22]['genre'])
print(",.,.,")
print(df_filtered.iloc[23]['genre-encoded'])
print(df_filtered.iloc[23]['genre'])
print(",.,.,")
print(df_filtered.iloc[12]['genre-encoded'])
print(df_filtered.iloc[12]['genre'])
print(",.,.,")
print("--9--")
print(numpy.unique(numpy.array(df_filtered['rating-encoded'])))
print(numpy.unique(numpy.array(df_filtered['rating'])))
print(df_filtered.iloc[2]['rating-encoded'])
print(df_filtered.iloc[2]['rating'])
print(df_filtered.iloc[3]['rating-encoded'])
print(df_filtered.iloc[3]['rating'])
print(df_filtered.iloc[4]['rating-encoded'])
print(df_filtered.iloc[4]['rating'])
print(",.,.,")
print(df_filtered.iloc[22]['rating-encoded'])
print(df_filtered.iloc[22]['rating'])
print(",.,.,")
print(df_filtered.iloc[23]['rating-encoded'])
print(df_filtered.iloc[23]['rating'])
print(",.,.,")
print(df_filtered.iloc[12]['rating-encoded'])
print(df_filtered.iloc[12]['rating'])
print(",.,.,")





# Seleziona le colonne rilevanti dal DataFrame
features = ['rating-encoded','genre-encoded', 'year', 'score', 'budget', 'gross']
features2 = ['rating-encoded','genre-encoded', 'year', 'budget', 'gross']
# Esempio di colonne selezionate

# Codifica le variabili categoriche e crea il DataFrame dei dati
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore  # Sostituire con ListaLogisticaGuadagno se necessario

# Dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza e addestra il modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Valuta le prestazioni del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred,zero_division=1))



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Rappresenta la curva di apprendimento
title = "Learning Curves (Logistic Regression Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()




print("-----")
print(y_test)
print(len(y_test))
print("-----")
print(y_pred)
print(len(y_pred))





classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Regressione Logistica (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()




print("--------")




# Seleziona le colonne rilevanti dal DataFrame
features = ['rating-encoded','genre-encoded', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
 # Esempio di colonne selezionate

# Codifica le variabili categoriche e crea il DataFrame dei dati
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno  # Sostituire con ListaLogisticaGuadagno se necessario

# Dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Inizializza e addestra il modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Valuta le prestazioni del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred,zero_division=1))



title = "Learning Curves (Logistic Regression Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Regressione Logistica (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()






X_encoded = pd.get_dummies(df.dropna()[['genre', 'rating', 'budget']])

# Suddivisione del dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, df.dropna()[['gross']], test_size=0.2, random_state=42)

# Creazione e addestramento del modello di regressione lineare
model = LinearRegression()
model.fit(X_train, y_train)

# Predizione sui dati di test
y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



X_encoded = pd.get_dummies(df.dropna()[['genre', 'rating', 'budget']])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, df.dropna()[['score']], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


X = df[(df['year'] == 1980)].dropna().drop('gross', axis=1)
y = df[(df['year'] == 1980)].dropna()['gross']


# Suddivisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione delle colonne numeriche e categoriche
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Costruzione del modello di rete neurale feedforward nel pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=4000,random_state=42,tol=1e-3,solver='adam',))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



X = df[(df['year'] == 1980)].dropna().drop('score', axis=1)
y = df[(df['year'] == 1980)].dropna()['score']


# Suddivisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione delle colonne numeriche e categoriche
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Costruzione del preprocessore per gestire le colonne numeriche e categoriche
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Costruzione del modello di rete neurale feedforward nel pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=4000,random_state=42,tol=1e-3,solver='adam',))
])

# Addestramento del modello
model.fit(X_train, y_train)

# Valutazione delle prestazioni del modello
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

X = df[(df['year'] == 1980)].dropna().drop('gross',axis=1)
y = df[(df['year'] == 1980)].dropna()['gross'].values
X_encoded = pd.get_dummies(X)
X = pd.get_dummies(X)
#X = X.transpose()
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_encoded)
#Y_encoded = pd.get_dummies(y)
model.fit(X_train, y_train)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisci il modello di regressione
model = LinearRegression()

# Addestra il modello
model.fit(X_train, y_train)

# Effettua predizioni sui dati di test
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
X_test=X_test['score']
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print(X_test)
print("--")
print(y_test)
print("---")
print(y_pred)

print("encon")
label_encoder = LabelEncoder()
df_filtered=df_filtered.dropna()
df_filtered['genre-encoded'] = label_encoder.fit_transform(df_filtered['genre'])
print(df_filtered['genre-encoded'])
df_filtered['rating-encoded'] = label_encoder.fit_transform(df_filtered['rating'])
print(df_filtered['rating-encoded'])
df_filtered.reset_index(drop=True)


features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','rating-encoded','genre-encoded','year','gross','budget']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore  # Assumendo che ListaLogisticaGuadagno sia la variabile target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))



title = "Learning Curves (Reti Neurali Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()



classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Reti Neurali (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()







features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno  # Assumendo che ListaLogisticaGuadagno sia la variabile target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))



title = "Learning Curves (Reti Neurali Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Reti Neurali (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()





features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','rating-encoded','genre-encoded','year','gross','budget']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (Random Forest Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Random Forest (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()





features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','rating-encoded','genre-encoded','year','gross','budget']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (AdaBoost Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap AdaBoost (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()






features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (Random Forest Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Random Forest (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()



features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (AdaBoost Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap AdaBoost (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()




features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','rating-encoded','genre-encoded','year','gross','budget']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore  # Assumendo che ListaLogisticaGuadagno sia la variabile target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (Albero Decisionale Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Albero Decisionale (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()




features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','rating-encoded','genre-encoded','year','gross','budget']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaScore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (AdaBoost Score)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap AdaBoost (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()





features = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno  # Assumendo che ListaLogisticaGuadagno sia la variabile target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (Albero Decisionale Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Albero Decisionale (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()




eatures = ['rating', 'genre', 'year', 'score', 'budget', 'gross']
features2 = ['year','score','rating-encoded','genre-encoded','year']
X = pd.get_dummies(df_filtered[features2].dropna())
y = ListaLogisticaGuadagno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


title = "Learning Curves (AdaBoost Guadagno)"
cv = 5  # Numero di fold nella cross-validation
plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=-1)

plt.show()


classification_report_str = classification_report(y_test, y_pred, zero_division=1)
# Analizza la stringa di testo del classification report e ottieni le metriche
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:  # Ignora le prime due righe e le ultime cinque righe
    row_data = line.split()
    if row_data:  # Ignora le righe vuote
        report_data.append(row_data)

# Estrai le etichette delle classi e i valori delle metriche
class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap AdaBoost (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()





# Prepara i dati (X) e i target (y) per il clustering
#X = df_filtered[['rating', 'genre', 'released', 'score', 'budget', 'gross']].dropna()
#X = pd.get_dummies(X)
#y = ListaLogisticaScore  # Target per il clustering

# Esegui il clustering con l'algoritmo k-means
#kmeans = KMeans(n_clusters=2, random_state=42)
#cluster_labels = kmeans.fit_predict(X)

#silhouette_avg = silhouette_score(X, cluster_labels)
#print(silhouette_avg)


#plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', alpha=0.5)
#plt.xlabel('Altri Dati')
#plt.ylabel('Dati Binari Score')
#plt.title('Clustering con k-means')
#plt.show()


print(" ------ ")



# Prepara i dati (X) e i target (y) per il clustering
#X = df_filtered[['rating', 'genre', 'released', 'score', 'budget', 'gross']].dropna()
#X = pd.get_dummies(X)
#y = ListaLogisticaGuadagno  # Target per il clustering

# Esegui il clustering con l'algoritmo k-means
#kmeans = KMeans(n_clusters=2, random_state=42)
#cluster_labels = kmeans.fit_predict(X)

#silhouette_avg = silhouette_score(X, cluster_labels)
#print(silhouette_avg)


#plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', alpha=0.5)
#plt.xlabel('Altri Dati')
#plt.ylabel('Dati Binari Guadagno')
#plt.title('Clustering con k-means')
#plt.show()



#mean_scores_by_year = df_filtered.groupby('released')['score'].mean()

# Visualizza le medie delle valutazioni per anno
#print("Media delle valutazioni per anno:")
#print(mean_scores_by_year)

# Plot delle medie delle valutazioni per anno
#plt.figure(figsize=(10, 6))
#mean_scores_by_year.plot(marker='o')
#plt.title('Media delle valutazioni per anno')
#plt.xlabel('Anno')
#plt.ylabel('Media delle valutazioni')
#plt.grid(True)
#plt.show()







year_means = df_filtered.groupby('year')['score'].mean()

# Aggiungi le nuove caratteristiche (media degli anni) al DataFrame X
X['mean2000'] = year_means.get(2000, 0) # Se non c'è media per quell'anno, assegna 0
X['mean2001'] = year_means.get(2001, 0)
X['mean2002'] = year_means.get(2002, 0)
X['mean2003'] = year_means.get(2003, 0)
X['mean2004'] = year_means.get(2004, 0)
X['mean2005'] = year_means.get(2005, 0)
# Continua con gli altri anni fino a 2005

# Esegui la divisione dei dati
X_train, X_test, y_train, y_test = train_test_split(X, ListaLogisticaScore, test_size=0.2, random_state=42)

# Addestra il modello
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Valuta l'accuratezza
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Analizza il classification report
classification_report_str = classification_report(y_test, y_pred, zero_division=1)
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:
    row_data = line.split()
    if row_data:
        report_data.append(row_data)

class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

# Aggiungi le nuove etichette di classe (media degli anni) e i valori delle metriche
class_labels += ['mean2000', 'mean2001', 'mean2002','mean2003','mean2004', 'mean2005']
num_new_columns = len(class_labels) - metrics_values.shape[1]
if num_new_columns > 0:
    metrics_values = np.hstack([metrics_values, np.zeros((metrics_values.shape[0], num_new_columns))])
metrics_values = np.vstack([metrics_values, np.zeros(len(class_labels))])

# Crea una heatmap utilizzando le metriche come dati
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Albero Decisionale (Score)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()







#test

features2 = ['year', 'score', 'rating-encoded', 'genre-encoded', 'gross', 'budget']
df_filtered = df_filtered.dropna(subset=features2)  # Rimuovi i valori NaN
X = df_filtered[features2]
y = ListaLogisticaGuadagno

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')  # Cambia n_bins secondo necessità
X_discretized = discretizer.fit_transform(X)

X_discretized_df = pd.DataFrame(X_discretized, columns=features2)

X_discretized_df['target'] = y

X_train, X_test, y_train, y_test = train_test_split(X_discretized_df, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

model = BayesianNetwork([
    ('year', 'score'),
    ('rating-encoded', 'score'),
    ('genre-encoded', 'score'),
    ('score', 'gross'),
    ('budget', 'gross'),
    ('gross', 'target')
])

model.fit(X_train_balanced, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)

y_pred = []
for index, row in X_test.iterrows():
    evidence = row.to_dict()
    del evidence['target']
    q = infer.map_query(variables=['target'], evidence=evidence)
    y_pred.append(q['target'])

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=1))

plt.figure(figsize=(10, 8))
G = nx.DiGraph()
G.add_edges_from(model.edges())
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=14, font_weight='bold')
plt.title('Bayesian Network Graph')
plt.show()

classification_report_str = classification_report(y_test, y_pred, zero_division=1)
report_data = []
lines = classification_report_str.split('\n')
for line in lines[2:-5]:
    row_data = line.split()
    if row_data:
        report_data.append(row_data)

class_labels = [row_data[0] for row_data in report_data]
metrics_values = np.array([row_data[1:] for row_data in report_data], dtype=np.float32)

print(class_labels)
print(metrics_values)

plt.figure(figsize=(10, 6))
sns.heatmap(metrics_values, annot=True, cmap='coolwarm', xticklabels=['precision', 'recall', 'f1-score'], yticklabels=class_labels)
plt.title('Classification Report Heatmap Bayesian Network (Guadagno)')
plt.xlabel('Metrics')
plt.ylabel('Class Labels')
plt.show()








# Supponiamo che df_filtered sia il DataFrame con i dati già discreti
features2 = ['year', 'score', 'rating-encoded', 'genre-encoded', 'gross', 'budget']

# Rimuovi i valori NaN
df_filtered = df_filtered.dropna(subset=features2)

# Seleziona le caratteristiche e la variabile target
X = df_filtered[features2].copy()  # Usa .copy() per evitare SettingWithCopyWarning
y = ListaLogisticaGuadagno  # Variabile target

# Converti 'y' in una Series di pandas se è una lista
if isinstance(y, list):
    y = pd.Series(y)

# Converti tutte le variabili in categoriali
for col in X.columns:
    X[col] = X[col].astype('category')

# Aggiungi la variabile target al DataFrame X per il training
X['target'] = y.astype('category')

# Dividi il dataset in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiamo la struttura della rete bayesiana
model = BayesianNetwork([
    ('year', 'score'),
    ('rating-encoded', 'score'),
    ('genre-encoded', 'score'),
    ('score', 'gross'),
    ('budget', 'gross'),
    ('gross', 'target')
])

# Aggiungiamo i dati al modello
model.fit(X_train, estimator=MaximumLikelihoodEstimator)

# Inizializziamo l'inferenza
infer = VariableElimination(model)

# Prediciamo il valore di 'target' per il set di test
y_pred = []
for index, row in X_test.iterrows():
    evidence = row.drop('target').to_dict()  # Rimuovi 'target' dall'evidenza
    q = infer.map_query(variables=['target'], evidence=evidence)
    y_pred.append(q['target'])

# Calcoliamo l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=1))