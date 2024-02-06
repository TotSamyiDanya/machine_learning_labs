import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def regression_linear():
    data = pd.read_csv(filepath_or_buffer='data/intrusion.csv')
    correlation_matrix = data.corr().round(2)

    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

    sns.pairplot(data)
    plt.show()

    x = data[['Transmission Range']]
    y = data[['Number of Barriers']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.3)

    model = LinearRegression()
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    r2 = r2_score(y_test, prediction)

    plt.scatter(x_test, y_test, color='black')
    plt.plot(x_test, prediction, label=f'Коэффициент детерминации: {r2}', color='red', lw=2, linestyle='-')
    plt.title('Линейная регрессия')
    plt.legend(loc='upper right')
    plt.show()
