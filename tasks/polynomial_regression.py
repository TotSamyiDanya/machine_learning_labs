import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def regression_polynomial():
    data = pd.read_csv(filepath_or_buffer='data/intrusion.csv')
    correlation_matrix = data.corr().round(2)

    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    sns.pairplot(data)
    plt.show()

    x = data[['Transmission Range']].values
    y = data[['Number of Barriers']].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.3)

    model = LinearRegression()
    poly = PolynomialFeatures(degree=3)

    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)
    x_fit_test = np.arange(x_test.min(), x_test.max(), 1)[:, np.newaxis]

    model.fit(x_train_poly, y_train)

    y_fit_poly = model.predict(poly.fit_transform(x_fit_test))
    r2 = r2_score(y_test, model.predict(x_test_poly))

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_fit_test, y_fit_poly, label=f'Коэффициент детерминации: {r2}', color='red', lw=2, linestyle='-')
    plt.title('Полиномиальная реграссия (3ая степень)')
    plt.legend(loc='upper right')
    plt.show()
