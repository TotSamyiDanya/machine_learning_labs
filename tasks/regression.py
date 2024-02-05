import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree


def linear_regression():
    # чтение данных из csv файла
    data = pandas.read_csv(filepath_or_buffer='data/intrusion.csv')

    # получение матрицы корреляции
    correlation_matrix = data.corr().round(2)

    # вывод матриц корреляции
    seaborn.heatmap(correlation_matrix, annot=True)
    plt.show()

    # вывод попарных графиков взаимосвязи
    seaborn.pairplot(data)
    plt.show()

    # создание модели
    model = LinearRegression()

    # выделение признака и метки
    x = data[['Transmission Range']]
    y = data[['Number of Barriers']]

    # деление признака и метки на тренировочные и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.3)

    # обучение модели
    model.fit(x_train, y_train)

    # создание предсказания
    prediction = model.predict(x_test)

    # полечение коэффициента детерминации
    r2 = r2_score(y_test, prediction)

    # вывод графика линейной регресии
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, prediction, label=f'Коэффициент детерминации: {r2}', color='red', lw=2, linestyle='-')
    plt.legend(loc='upper right')
    plt.show()


def polynomial_regression():
    # чтение данных из csv файла
    data = pandas.read_csv(filepath_or_buffer='data/intrusion.csv')

    # получение матрицы корреляции
    correlation_matrix = data.corr().round(2)

    # вывод матриц корреляции
    seaborn.heatmap(correlation_matrix, annot=True)
    plt.show()

    # вывод попарных графиков взаимосвязи
    seaborn.pairplot(data)
    plt.show()

    # создание модели
    model = LinearRegression()

    # выделение признака и метки
    x = data[['Transmission Range']].values
    y = data[['Number of Barriers']].values

    # деление признака и метки на тренировочные и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.3)

    # создание полиномиального признака
    poly = PolynomialFeatures(degree=3)

    # создание полиномиальных тренировочных и тестовых признаков
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.fit_transform(x_test)

    # создание полиномиального трансформированного признака
    x_fit_test = np.arange(x_test.min(), x_test.max(), 1)[:, np.newaxis]

    # обучение модели
    model.fit(x_train_poly, y_train)

    # нахождение значений полиномиальной функции
    y_fit_poly = model.predict(poly.fit_transform(x_fit_test))

    # нахождение коэффициента детерминации
    r2 = r2_score(y_test, model.predict(x_test_poly))

    # вывод графика полиномиальной регрессии
    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_fit_test, y_fit_poly, label=f'Коэффициент детерминации: {r2}', color='red', lw=2, linestyle='-')
    plt.legend(loc='upper right')
    plt.show()


def random_forest_regression():
    # чтение данных из csv файла
    data = pandas.read_csv('data/intrusion.csv')

    # получение матрицы корреляции
    correlation_matrix = data.corr().round(2)

    # вывод матриц корреляции
    seaborn.heatmap(correlation_matrix, annot=True)
    plt.show()

    # вывод попарных графиков взаимосвязи
    seaborn.pairplot(data)
    plt.show()

    # выделение признака и метки
    x = data.iloc[:, 1:2].values
    y = data.iloc[:, 4].values

    # создание модели
    model = RandomForestRegressor(n_estimators=10, random_state=0)

    # обучение модели
    model.fit(x, y)

    # создание предсказания
    prediction = model.predict(x)

    # нахождение коэффициента детерминации
    r2 = r2_score(y, prediction)

    # создание сетки признаков
    x_grid = np.arange(min(x), max(x), 0.01)
    x_grid = x_grid.reshape(len(x_grid), 1)

    # вывод графика случайного леса
    plt.scatter(x, y, color='black')
    plt.plot(x_grid, model.predict(x_grid), label=f'Коэффициент детерминации: {r2}', color='red')
    plt.legend(loc='upper right')
    plt.show()

    # создание дерева решений
    tree_to_plot = model.estimators_[0]

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, feature_names=data.columns.tolist(), filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()

