# -*- coding: utf-8 -*-

# Import für Beispieldatensätze
from sklearn import datasets
# Erzeugen des Iris-Datensatzes
iris = datasets.load_iris()

# Import und Erzeugung eines DecisionTree Klassierers
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

# # Alternativ kann ein NaiveBayes Klassierer verwendet werden
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# # Weitere Alternative: SupportVector Maschine
# from sklearn.svm import SVC
# clf = SVC()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y)

# Die Variable clf für den Klassierer ist nun definiert und kann gebraucht werden;
# der folgende Code funktioniert vor alle Typen von Klassierern gleich.

# Der Klassierer wird nun mit dem Iris-Datensatz ohne das letzte Element trainiert.
clf.fit(Xtrain,ytrain)
# Der Klassierer soll nun das letzte Element des Datensatzes klassieren
print(clf.predict(Xtest))
# Zum Vergleich der effektive Typ des letzten Elementes des Datensatzes
print(ytest)

# Mit Hilfe der CrossValidation soll der Klassierer getestet werden.
# http://scikit-learn.org/stable/modules/cross_validation.html
# Import des Package für die CrossValidation
from sklearn import cross_validation

# Aufspalten des Datensatzes in Trainings- und Testdaten (die Elemente werden gemischt)
# test_size ist entweder ein float zwischen 0.0 und 1.0 (Anteil) oder ein ganze Zahl
data_train, data_test, target_train, target_test = \
    cross_validation.train_test_split(iris.data, iris.target, test_size=0.2)

# Trainieren des Klassifizierers mit den Trainingsdaten
clf.fit(data_train, target_train)
# Klassierungen der testdaten
print clf.predict(data_test)
# Vergleich der vorhergesageten Klassen mit den effektiven Klassen
print(target_test)
# Bestimmung der Genauigkeit ders Klassierers auf den Testdaten
print(clf.score(data_test, target_test))


gs = GridSearchCV(clf,{max_depth:[1,3,5,10],criterion:['gini','cross-entropy']}
gs.fit(
