import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Charger le jeu de données Iris
iris = sns.load_dataset("iris")

# Titre de l'application
st.title("Analyse du jeu de données Iris")

# Afficher un résumé du jeu de données
st.write("### Aperçu des premières lignes du jeu de données Iris :")
st.write(iris.head())

# Graphique 1 : Nuage de points (scatter plot)
st.write("### Relation entre la longueur et la largeur des pétales")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="petal_length", y="petal_width", hue="species", data=iris, palette="Set1", ax=ax1)
ax1.set_title("Relation entre Petal.Length et Petal.Width")
ax1.set_xlabel("Petal Length (cm)")
ax1.set_ylabel("Petal Width (cm)")
st.pyplot(fig1)

# Graphique 2 : Boîte à moustaches (boxplot)
st.write("### Comparaison de la longueur des sépales par espèce")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.boxplot(x="species", y="sepal_length", data=iris, palette="Pastel1", ax=ax2)
ax2.set_title("Comparaison de Sepal.Length par espèce")
ax2.set_xlabel("Espèce")
ax2.set_ylabel("Longueur des sépales (cm)")
st.pyplot(fig2)

# Analyse en Composantes Principales (PCA)
st.write("### Analyse en Composantes Principales (PCA)")

# Standardiser les données (supprimer la colonne 'species' pour PCA)
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.iloc[:, :-1])

# Appliquer PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(iris_scaled)

# Créer un DataFrame pour les composants PCA
pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
pca_df["species"] = iris["species"]

# Graphique PCA
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", hue="species", data=pca_df, palette="Set1", s=100, ax=ax3)
ax3.set_title("Analyse en Composantes Principales (PCA)")
ax3.set_xlabel("Composante Principale 1")
ax3.set_ylabel("Composante Principale 2")
st.pyplot(fig3)

# Lancer l'application avec Streamlit
st.write("### Application interactive avec Streamlit")
