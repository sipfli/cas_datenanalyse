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

# Die Variable clf für den Klassierer ist nun definiert und kann gebraucht werden;
# der folgende Code funktioniert vor alle Typen von Klassierern gleich.

# Der Klassierer wird nun mit dem Iris-Datensatz ohne das letzte Element trainiert.
clf.fit(iris.data[:-1], iris.target[:-1]) 
# Der Klassierer soll nun das letzte Element des Datensatzes klassieren
print(clf.predict(iris.data[-1]))
# Zum Vergleich der effektive Typ des letzten Elementes des Datensatzes
print(iris.target[-1])

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

# Verwendung von Iteratoren zur CrossValidation

# KFold: Datensatz wird in n Teile aufgeteilt,
# jeweils n-1 viele Teile als Trainingsdaten und
# ein teil als Testdaten.
# Die Funktion teilt nicht den Datensatz auf, sondern erzeugt
# einen Iterator mit entsprechende Indexmengen.
for trainIndices, testIndices in cross_validation.KFold(len(iris.data), n_folds=3) :
    clf.fit(iris.data[trainIndices], iris.target[trainIndices])
    print(clf.score(iris.data[testIndices], iris.target[testIndices]))
# Das Ergebnis ist unbrauchbar, da der Iris-Datensatz nach Klasse sortiert ist
# und KFold nicht mischt, wenn man es nicht verlangt. Besser:
for trainIndices, testIndices in cross_validation.KFold(len(iris.data), n_folds=3, shuffle=True) :
    clf.fit(iris.data[trainIndices], iris.target[trainIndices])
    print(clf.score(iris.data[testIndices], iris.target[testIndices]))
# Statt for-Schleife selber zu schreiben, kann man die Funktion verwenden
print(cross_validation.cross_val_score(clf, iris.data, iris.target, \
                cv=cross_validation.KFold(len(iris.data), n_folds=3, shuffle=True)))
# Eine Alternative zu KFold ist LeaveOneOut 
print(cross_validation.cross_val_score(clf, iris.data, iris.target, \
                cv=cross_validation.LeaveOneOut(len(iris.data))))






