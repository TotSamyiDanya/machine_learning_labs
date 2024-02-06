import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def regression_ensemble():
    data = pd.read_csv('data/intrusion.csv')
    x = data.iloc[:, 1:2].values
    y = data.iloc[:, 4].values

    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(estimators=
    [
        ('svt', svr_lin),
        ('lr', ridge)
    ])

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    stregr.fit(x_train, y_train)
    stregr.predict(x_test)

    test_predict = stregr.predict(x_test)
    train_predict = stregr.predict(x_train)

    print(f'Коэффициент детерминации: {r2_score(y_test, test_predict)}')

    a = plt.axes(aspect='equal')
    lims = [0, 150]
    plt.scatter(y_test, test_predict)
    plt.xlabel('Истиные значения y')
    plt.ylabel('Предсказания для y')
    plt.title('Ансамблевая регрессия (стекинг)')
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

    print(f'Среднеквадратическая ошибка: {mean_squared_error(y_test, test_predict)}')
