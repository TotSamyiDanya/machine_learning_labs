import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score, davies_bouldin_score, mutual_info_score


def clusterize_k_means():
    data = pd.read_csv('data/heart.csv')

    x = data[['creatinine_phosphokinase', 'age']]
    n_clusters = 4

    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(x)

    prediction = model.predict(x)
    result = pd.DataFrame(x)
    result['cluster'] = prediction

    colors = ['black', 'red', 'blue', 'green']
    for k in range(0, n_clusters):
        plot = result[result['cluster'] == k]
        plt.scatter(plot['creatinine_phosphokinase'], plot['age'], color=colors[k])
    plt.ylabel('age')
    plt.xlabel('creatinine_phosphokinase')
    plt.show()

    ari = adjusted_rand_score(prediction, model.labels_)
    print(f'Скорректированная оценка по индексу рандомизации: {ari}')

    ris = rand_score(prediction, model.labels_)
    print(f'Оценка по индексу рандомизации: {ris}')

    ss = silhouette_score(x, model.labels_)
    print(f'Оценка силуэта: {ss}')

    dbs = davies_bouldin_score(x, model.labels_)
    print(f'Оценка сходства кластеров: {dbs}')

    mis = mutual_info_score(prediction, model.labels_)
    print(f'Оценка взаимоной информации: {mis}')
