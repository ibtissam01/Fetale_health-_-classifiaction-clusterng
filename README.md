# Projet : Analyse de données en apprentissage automatique pour résoudre des problèmes de classification et de clustering dans le domaine de la santé 
Bien sûr ! Voici le fichier README correspondant :

## Ce projet a été réalisé par :
- Elghazi Soufiane
- Amine Maasri
- Ibtissam Labyady

Dans ce projet, nous avons analysé des données dans le domaine de la santé, en particulier la classification de la santé fœtale, en utilisant des techniques d'apprentissage automatique. Nous avons exploré différentes techniques de classification et de clustering pour résoudre des problèmes spécifiques dans ce domaine, les avons évaluées et avons présenté les résultats à interpréter.

## Dataset

Le lien vers le jeu de données : [Fetal Health Classification | Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification?select=fetal_health.csv)

### Contexte

La réduction de la mortalité infantile est un objectif clé du développement durable des Nations Unies et un indicateur important du progrès humain. L'ONU prévoit que d'ici 2030, tous les pays chercheront à mettre fin aux décès évitables de nouveau-nés et d'enfants de moins de 5 ans, en réduisant la mortalité des moins de 5 ans à au moins 25 pour 1 000 naissances vivantes.

Les cardiotocogrammes (CTG) sont une méthode simple et abordable pour évaluer la santé fœtale, permettant aux professionnels de la santé de prendre des mesures pour prévenir la mortalité infantile et maternelle. Les CTG fonctionnent en envoyant des impulsions ultrasonores et en lisant les réponses, fournissant des informations sur la fréquence cardiaque fœtale (FHR), les mouvements fœtaux, les contractions utérines, etc.

### Données

Ce jeu de données contient 2126 enregistrements d'examens de cardiotocogrammes, avec des caractéristiques extraites et classées en 3 classes par trois experts obstétriciens :

- Normal
- Suspect
- Pathologique

## Fonctionnalités du code

Ce code est une application Streamlit qui effectue les tâches suivantes sur les données de santé fœtale :

1. EDA (Analyse exploratoire des données) : Le code charge un ensemble de données à partir d'un fichier CSV et affiche les données dans un DataFrame à l'aide de la bibliothèque Pandas. Il affiche également une heatmap de corrélation pour visualiser les relations entre les variables.

2. Classification : Le code charge des modèles de classification (KNN, arbre de décision, régression logistique, forêt aléatoire, SVM et Naive Bayes) à partir de fichiers pickle préalablement enregistrés. Il permet à l'utilisateur de sélectionner un modèle, de définir les paramètres du modèle et de prédire la classe de sortie en utilisant les données de test. Le code affiche ensuite une matrice de confusion, les prédictions et les valeurs réelles, ainsi que la précision du modèle.

3. Clustering : Le code charge des modèles de clustering (K-means, clustering hiérarchique et DBSCAN) à partir de fichiers pickle préalablement enregistrés. Il permet à l'utilisateur de choisir un modèle, de définir les paramètres du modèle et de générer les clusters à partir des données d'entrée. Les résultats sont affichés sous forme de graphiques de dispersion.

En résumé, ce code combine des fonctionnalités d'EDA, de classification et de clustering pour analyser un ensemble de données de santé fœtale, fournir des visualisations et des prédictions à l'utilisateur.

