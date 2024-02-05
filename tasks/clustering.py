import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score, davies_bouldin_score, mutual_info_score
from sklearn.mixture import GaussianMixture


def k_means_clustering():
    # чтение данных их csv файла
    data = pandas.read_csv('data/heart.csv')

    # выбор признаков для кластеризации
    x = data[['creatinine_phosphokinase', 'age']]

    # задание количества кластеров
    n_clusters = 4

    # создание модели
    model = KMeans(n_clusters=n_clusters, random_state=0)

    # обучение модели
    model.fit(x)

    # создание предсказания
    prediction = model.predict(x)

    # сбор результатов кластеризации
    result = pandas.DataFrame(x)
    result['cluster'] = prediction

    # определние цветов для кластеров
    colors = ['black', 'red', 'blue', 'green']

    # вывод графика результатов кластеризации
    for k in range(0, n_clusters):
        plot = result[result['cluster'] == k]
        plt.scatter(plot['creatinine_phosphokinase'], plot['age'], color=colors[k])
    plt.ylabel('age')
    plt.xlabel('creatinine_phosphokinase')
    plt.show()

    # скорректированная оценка по индексу рандомизации
    ari = adjusted_rand_score(prediction, model.labels_)
    print(f'Скорректированная оценка по индексу рандомизации: {ari}')

    # оценка по индексу рандомизации
    ris = rand_score(prediction, model.labels_)
    print(f'Оценка по индексу рандомизации: {ris}')

    # оценка силуэта
    ss = silhouette_score(x, model.labels_)
    print(f'Оценка силуэта: {ss}')

    # оценка сходства кластеров
    dbs = davies_bouldin_score(x, model.labels_)
    print(f'Оценка сходства кластеров: {dbs}')

    # оценка взаимоной информации
    mis = mutual_info_score(prediction, model.labels_)
    print(f'Оценка взаимоной информации: {mis}')


def gaussian_mixture_clustering():
    # чтение данных их csv файла
    data = pandas.read_csv('data/heart.csv')

    # выбор признаков для кластеризации
    x = data[['creatinine_phosphokinase', 'age']]

    # задание количества кластеров
    n_clusters = 4

    # создание модели
    model = GaussianMixture(n_components=n_clusters, random_state=0)

    # обучение модели
    model.fit(x)

    # создание предсказания
    prediction = model.predict(x)

    # сбор результатов кластеризации
    result = pandas.DataFrame(x)
    result['cluster'] = prediction

    # определние цветов для кластеров
    colors = ['black', 'red', 'blue', 'green']

    # вывод графика результатов кластеризации
    for k in range(0, n_clusters):
        plot = result[result['cluster'] == k]
        plt.scatter(plot['creatinine_phosphokinase'], plot['age'], color=colors[k])
    plt.ylabel('age')
    plt.xlabel('creatinine_phosphokinase')
    plt.show()

    # вывод информационного критерия акаике
    print(f'Информационный критерий Акаике: {model.aic(x)}')

    # вывод байесовского информационного критерия
    print(f'Байесовский информационный критерий: {model.bic(x)}')

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=model[["BIC", "AIC"]])
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.show()
