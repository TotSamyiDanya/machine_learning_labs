import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture


def clusterize_gaussian_mixture():
    data = pd.read_csv('data/heart.csv')

    x = data[['creatinine_phosphokinase', 'age']]
    n_clusters = 4

    model = GaussianMixture(n_components=n_clusters, random_state=0)
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

    print(f'Информационный критерий Акаике: {model.aic(x)}')
    print(f'Байесовский информационный критерий: {model.bic(x)}')
