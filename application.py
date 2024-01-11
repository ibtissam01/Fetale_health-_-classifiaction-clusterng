
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.model_selection import train_test_split
import seaborn as sns
#----------------------------------------------------------------------------------------------------
# Ajouter une image
#from PIL import Image
#image = Image.open('esi_icone.png')
#st.image(image, caption='ESI', use_column_width=True,width=50)
#----------------------------------------------------------------------------------------------------

# Ajouter des icônes d'auteurs avec leur nom
''' st.write('## Auteurs')
st.write('Créer Par :')
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        image_author1 = Image.open('icons8_user.ico')
        st.image(image_author1, width=50)
        st.write('Elghazi Soufiane')
    with col2:
        image_author2 = Image.open('icons8_user.ico')
        st.image(image_author2, width=50)
        st.write('Amine Maasri')
    with col3:
        image_author3 = Image.open('icons8_user.ico')
        st.image(image_author3, width=50)
        st.write('Ibtissam Labyady')
'''
#----------------------------------------------------------------------------------------------------
def plot_confusion_matrix(y_test,y_pred, cmap):
    classes=['class 1', 'class 2','class 3']
    # Créer la matrice de confusion
    conf_mat = confusion_matrix(y_test, y_pred)

    # Afficher la matrice de confusion avec les valeurs formatées
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(conf_mat, cmap=cmap)

    # Ajouter les valeurs dans la matrice de confusion
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], 'd'),
                    ha="center", va="center", color="white")

    # Ajouter les noms des classes et du graphe
    ax.set_title("Matrice de confusion")
    ax.set_xticks(range(conf_mat.shape[1]))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(conf_mat.shape[0]))
    ax.set_yticklabels(classes)
    ax.set_ylabel('Vraie classe')
    ax.set_xlabel('Classe prédite')
    plt.colorbar(im)

    # Afficher la figure
    plt.show()
    st.pyplot(fig)
#----------------------------------------------------------------------------------------------------
st.title("EDA")
# Charge les données
data = pd.read_csv("fetal_health.csv")
X = data.loc[:, ['histogram_median', 'histogram_mean', 'percentage_of_time_with_abnormal_long_term_variability','histogram_variance', 'histogram_mode', 'baseline value', 'accelerations',                 'abnormal_short_term_variability', 'prolongued_decelerations', 'mean_value_of_short_term_variability']]
Y = data['fetal_health']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Création d'une zone de défilement
with st.container():
    # Affichage du DataFrame
    st.dataframe(data, width=700, height=400)
#----------------------------------------------------------------------------------------------------
'''# Créez votre plot avec Matplotlib
fig, ax = plt.subplots(figsize=(20, 15))
data.hist(ax=ax)
ax.set_title('Histogramme des colonnes')

# Affichez votre plot avec Streamlit
st.pyplot(fig)'''
#---------------------------------------------------------------------------------------------------
# Matrice de corrélation
corr = data.corr(method = "spearman")
# Afficher la carte de chaleur
fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
ax.set_title("Heatmap de corrélation")
plt.show()
st.pyplot(fig)
#----------------------------------------------------------------------------------------------------


st.title("Classification")
# Charge les modèles
model_names = ['KNN', 'Arbre de décision', 'Régression logistique', 'Forêt aléatoire', 'SVM', 'Naive Bayes']
models = {}
for model_name in model_names:
    with open(model_name + '.pkl', 'rb') as file:
        models[model_name] = pickle.load(file)

# Standardise les données
scaler = StandardScaler()
X_selected = scaler.fit_transform(X)
X_train_selected = scaler.fit_transform(X_train)
X_test_selected = scaler.transform(X_test)

# Définit les paramètres de chaque modèle
model_params = {
    'KNN': {'n_neighbors': st.slider('Nombre de voisins pour KNN', 1, 20, 5)},
    'Arbre de décision': {'max_depth': st.slider('Profondeur maximale pour l\'arbre de décision', 1, 20, 5)},
    'Régression logistique': {'C': st.slider('Régression logistique - Force de régularisation', 0.1, 100.0, 10.0)},
    'Forêt aléatoire': {'n_estimators': st.slider('Nombre d\'estimateurs pour la forêt aléatoire', 1, 200, 50)},
    'SVM': {'C': st.slider('SVM - Force de régularisation', 0.1, 10.0, 1.0)},
    'Naive Bayes': {'var_smoothing': st.slider('Naive Bayes - Lissage', 1e-10, 1e-8, 1e-9)}
}

# Affiche l'interface utilisateur

st.write("Prédiction de la santé fœtale")
model_name = st.selectbox("Sélectionnez un modèle", model_names)
params = model_params[model_name]
if st.button("Prédire"):
    model = models[model_name]
    model.set_params(**params)
    Y_pred_train = model.predict(X_train_selected)
    Y_pred = model.predict(X_test_selected)

    plot_confusion_matrix(Y_test,Y_pred, 'PiYG')
    st.write("Précision sur les données d'entraînement :", accuracy_score(Y_train, Y_pred_train))
    # Création de deux colonnes
    col1, col2 = st.columns(2)

    # Affichage des prédictions dans la première colonne
    with col1:
        st.write("Prédiction :", Y_pred)

    # Affichage des valeurs réelles dans la deuxième colonne
    with col2:
        st.write("Valeurs réelles :", Y_test)

    # Affichage de la précision en dessous des colonnes
    st.write("Précision sur les données de test :", accuracy_score(Y_test, Y_pred))

st.title("Clustring")
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Charge les modèles
model_names = ['Kmeans', 'Agglomorative', 'DBSCAN']
models = {}
for model_name in model_names:
    with open(model_name + '.pkl', 'rb') as file:
        models[model_name] = pickle.load(file)



# Définit les paramètres de chaque modèle
model_params = {
    'Kmeans': {'n_clusters': st.slider('Nombre de clusters k pour Kmeans', 2, 11, 1)},
    'Agglomorative': {'n_clusters': st.slider('Nombre de clusters k pour Agglomorative', 2, 11, 1)},
    'DBSCAN': {'min_samples': st.slider('min_samples', 2, 5, 1),'eps': st.slider('Epsilon', 0.1, 1.1, 0.1)}
}

# Affiche l'interface utilisateur
st.write("Génerate clusters")
model_name = st.selectbox("Sélectionnez un modèle", model_names)
params = model_params[model_name]
if st.button("Générer"):
    model = models[model_name]
    model.set_params(**params)
    model_labels = model.fit_predict(X_selected)
    fig, ax = plt.subplots(figsize=(20, 15))
    # Afficher les résultats
    plt.scatter(X_selected[:, 0], X_selected[:, 1], c=model_labels, s=50)
    plt.title(model_name)
    plt.show()
    st.pyplot(fig)

st.write(" clusters of best modèles")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_selected)

hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_selected)

dbscan = DBSCAN(eps=0.4, min_samples=4)
dbscan_labels = dbscan.fit_predict(X_selected)
# Afficher les résultats
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(X_selected[:, 0], X_selected[:, 1], c=kmeans_labels, s=50)
axes[0].set_title("K-Means")
axes[1].scatter(X_selected[:, 0], X_selected[:, 1], c=hierarchical_labels, s=50)
axes[1].set_title("Clustering hiérarchique")
axes[2].scatter(X_selected[:, 0], X_selected[:, 1], c=dbscan_labels, s=50)
axes[2].set_title("DBSCAN")
plt.show()
st.pyplot(fig)
