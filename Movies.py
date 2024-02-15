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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



Dim=200
Dim2=205
Dim3=193
Dim4=207
Dim5=191
Dim6=200
Dim7=166
Dim8=162
Dim9=170
Dim10=168
Dim11=159
Dim12=159
Dim13=196
Dim14=203
Dim15=200

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

Nazion = []
Nazion2 = []
Nazion3 = []
Nazion4 = []
Nazion5 = []
Nazion6 = []

df = pd.read_csv('movies.csv')

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
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto3 = get_ontology("http://www.w3.org/2002/07/Film2002")
onto3.save(file="Film2002.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2002.owl"
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
while(i<=7667):
  if ("2002" in str(df.loc[i,rilasc])):
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
  i=i+1

i=0
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto4 = get_ontology("http://www.w3.org/2002/07/Film2003")
onto4.save(file="Film2003.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2003.owl"
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
i=0
while(i<=7667):
  if ("2003" in str(df.loc[i,rilasc])):
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
      if("2004" in str(df.loc[i,rilasc])):
          i=7667
  i=i+1

i=0
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto5 = get_ontology("http://www.w3.org/2002/07/Film2004")
onto5.save(file="Film2004.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2004.owl"
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
budg='budget'
guad='gross'
compa='company'
durat='runtime'
i=0
while(i<=7667):
  if ("2004" in str(df.loc[i,rilasc])):
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
      if("2005" in str(df.loc[i,rilasc])):
          i=7667
  i=i+1


i=0
onto_path.append("C:/Users/Vanni/PycharmProjects/pythonProject")
onto6 = get_ontology("http://www.w3.org/2002/07/Film2005")
onto6.save(file="Film2005.owl", format="rdfxml")

file_path = "C:/Users/Vanni/PycharmProjects/pythonProject/Film2005.owl"
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
  if ("2005" in str(df.loc[i,rilasc])):
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
      if("2006" in str(df.loc[i,rilasc])):
          i=7667
  i=i+1

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
plt.show()





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
plt.show()





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
plt.show()





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
plt.show()





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
plt.show()





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
plt.show()



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
plt.show()




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
plt.show()




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
plt.show()




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
plt.show()




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
plt.show()




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
plt.show()



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
plt.show()




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
plt.show()




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
plt.show()




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
plt.show()




ns = Namespace("http://www.w3.org/2002/07/Film2005")

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
plt.show()




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
plt.show()






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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()









ns = Namespace("http://www.w3.org/2002/07/Film2000")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2000#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2000)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()




ns = Namespace("http://www.w3.org/2002/07/Film2001")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2001#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion2.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion2)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2001)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()




ns = Namespace("http://www.w3.org/2002/07/Film2002")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2002#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion3.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion3)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2002)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()




ns = Namespace("http://www.w3.org/2002/07/Film2003")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2003#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion4.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion4)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2003)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()




ns = Namespace("http://www.w3.org/2002/07/Film2004")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2004#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion5.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion5)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2004)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()





ns = Namespace("http://www.w3.org/2002/07/Film2005")

# Crea il grafo RDF utilizzando rdflib
grafo = default_world.as_rdflib_graph()

# Lista per memorizzare le nazioni
Nazion = []

# Query SPARQL per ottenere le nazioni dei film
query = """
PREFIX ns: <http://www.w3.org/2002/07/Film2005#>
SELECT ?nazione
WHERE {
    ?film rdf:type ns:Film .
    ?film ns:Nazione ?nazione .
}
"""

# Esegui la query SPARQL
risultati = grafo.query(query, initNs={'ns': ns, 'rdf': RDF})

# Estrai le nazioni e aggiungile alla lista
for risultato in risultati:
    nazion = risultato['nazione']
    Nazion.append(str(nazion))

# Calcola la frequenza di ciascuna nazione
conteggio_nazioni = Counter(Nazion)

# Prepara i dati per il grafico a istogramma
labels = list(conteggio_nazioni.keys())
sizes = list(conteggio_nazioni.values())

# Crea il grafico a istogramma
plt.bar(labels, sizes)

# Aggiungi etichette agli assi
plt.xlabel('Nazione')
plt.ylabel('Frequenza')
plt.title('Distribuzione delle Nazioni (Film anni 2005)')
plt.xticks(rotation=45, ha='right', fontsize=6)

# Mostra il grafico
plt.show()

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
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Generi (Film anni 2000)")
plt.show()

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
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Generi (Film anni 2001)")
plt.show()


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
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Distribuzione dei Generi (Film anni 2004)")
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


# Plot dei dati di test e delle predizioni
plt.scatter(X_test, y_test, color='blue', label='Dati di test')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predizioni')

plt.title('Regressione lineare')
plt.xlabel('Variabile indipendente')
plt.ylabel('Variabile dipendente')
plt.legend()
plt.show()





demographic_data = pd.read_csv('movies.csv')
df = pd.merge(df, demographic_data, on='country', how='left')

# Suddividi il dataset in training set e test set
X = df[(df['year'] == 1990)].dropna().drop('name','gross','year', axis=1)
y = df[(df['year'] == 1990)].dropna()['gross']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione delle features numeriche
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modello di regressione avanzato
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Valutazione del modello
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)