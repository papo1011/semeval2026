import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

class Clustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.data_scaled = None
        self.labels = None

    def prepare_data(self, df):
        print("Scaling data")
        self.data_scaled = self.scaler.fit_transform(df)
        print("Data correctly scaled")
        return self.data_scaled
    
    def cluster(self):
        if self.data_scaled is None:
            raise ValueError("Before you must prepare the data")
        
        self.labels = self.model.fit_predict(self.data_scaled)
        return self.labels 
    
    def visualizza_2d(self, df, colonna_x, colonna_y):
        """
        Mostra un grafico a dispersione basato su due colonne scelte.
        """
        if self.labels is None:
            print("Nessun label trovato. Esegui prima il clustering.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(df[colonna_x], df[colonna_y], c=self.labels, cmap='viridis')
        plt.title(f"Visualizzazione Clustering: {colonna_x} vs {colonna_y}")
        plt.xlabel(colonna_x)
        plt.ylabel(colonna_y)
        plt.colorbar(label='Cluster ID')
        plt.show()

dataset = load_iris(as_frame=True).frame
clustering = Clustering(3)
clustering.prepare_data(dataset)
clustering.cluster()
clustering.visualizza_2d(dataset, colonna_x='petal length (cm)', colonna_y='petal width (cm)')
