import pandas as pd
import scipy.cluster.hierarchy as hc
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize


def clusterize_hierarchical():
    data = pd.read_csv('data/cc.csv')
    data = data.drop('CUST_ID', axis=1)
    data.ffill(inplace=True)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_normalized = normalize(data_scaled)

    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_normalized)
    data_pca = pd.DataFrame(data_pca)
    data_pca.columns = ['P1', 'P2', 'P3']

    dendrogram = hc.dendrogram((hc.linkage(data_pca, method='ward')))

    cluserize_agglomerative(5, 1, data_pca)
    cluserize_agglomerative(4, 2, data_pca)
    cluserize_agglomerative(3, 3, data_pca)
    cluserize_agglomerative(2, 4, data_pca)


def cluserize_agglomerative(k, l, data_pca):
    agc = AgglomerativeClustering(n_clusters=k)
    plt.figure(figsize=(10, 10))
    plt.scatter(data_pca['P1'], data_pca['P2'], c=agc.fit_predict(data_pca), cmap='rainbow')
    plt.title(f'Иерархическая кластеризация снизу вверх. Уровень {l}. Кластеров - {k}', fontsize=18)
    plt.show()
