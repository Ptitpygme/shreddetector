# shreddetector

##Project / Projet

Shreddetector is an intership project. Its objective is to identify shredders, from strips of papers.
The project helps to extract information about strips and shredder. With this information it's possible to identify different references of shredders.
The project needs to be perfected to identify two shredders of the same reference.

Sheddector est un projet de stage. son objectif est d'identifier des broyeurs à papier à partir des bandes de papier.
Le projet aide à extraire des données sur les bandes et le broyeur. Avec ces données il est possible d'identifier différents broyeurs de modèle différents.
Le projet doit être perfectionné pour identifier deux broyeurs de même référence.


##Usage

```python

import master

master.run('Strips','F1', 0) # An exemple of call 

listFileGuessMe=['C','D','E','F','O']

master.guessMeRandom('/udd/cvolantv/Pictures/ScanDetector/',listFileGuessMe,170) #Show informations about a unknow strip before showing the solution.

```


##Prerequisite
There is actually a set of strips available to experiment the project.

If you want to make your own set, you will need to scan or take a photo of yours strips in a green background.

Actually the only format the project can take is A3.

##Possible upgrades

-Accept more format
-Accept cross-cut shredder
-Create a data set and automate the choice of shredder in the guess game.

##To try

-See if the results of the project on cross-cut shredder

