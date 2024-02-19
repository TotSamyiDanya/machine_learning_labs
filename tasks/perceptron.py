import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, \
    mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.inspection import PartialDependenceDisplay


def perceptron_single_layer_classificate():
    data = pd.read_csv('data/sonar.csv', header=None)

    data.hist(figsize=(12, 12))
    plt.show()

    data.plot(kind='density', subplots=True, layout=(8, 8), sharex=False, legend=False, figsize=(12, 12))
    plt.show()

    correlation_matrix = (data.select_dtypes(include=['number'])).corr()
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, linecolor="black", fmt=".4f", ax=ax)
    plt.show()

    array = data.values
    x, y = array[:, 0:60].astype(float), array[:, 60]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)
    perceptron.fit(x_train, y_train)

    y_pred = perceptron.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)

    print(y_pred)

    name = ['камень', 'металлический цилиндр']
    colors = {'R': 'red', 'M': 'blue'}
    for i in ['R', 'M']:
        xs = x_test[:, 0][y_pred == i]
        ys = x_test[:, 15][y_pred == i]
        plt.scatter(xs, ys, c=colors[i])
        plt.legend(name)
    plt.title('Различие камней и металлических цилиндров сонаром')
    plt.show()


def perceptron_multi_layer_xor():
    xs = np.array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]).reshape(4, 2)

    ys = np.array([0, 1, 1, 0]).reshape(4, )

    model = MLPClassifier(
        activation='relu', max_iter=10000, hidden_layer_sizes=(4, 2))
    model.fit(xs, ys)

    print('score:', model.score(xs, ys))
    print('predictions:', model.predict(xs))
    print('expected:', np.array([0, 1, 1, 0]))


def perceptron_multi_layer_and():
    xs = np.array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]).reshape(4, 2)

    ys = np.array([0, 0, 0, 1]).reshape(4, )

    model = MLPClassifier(
        activation='relu', max_iter=10000, hidden_layer_sizes=(4, 2))
    model.fit(xs, ys)

    print('score:', model.score(xs, ys))
    print('predictions:', model.predict(xs))
    print('expected:', np.array([0, 0, 0, 1]))


def perceptron_multi_layer_or():
    xs = np.array([
        0, 0,
        0, 1,
        1, 0,
        1, 1
    ]).reshape(4, 2)

    ys = np.array([0, 1, 1, 1]).reshape(4, )

    model = MLPClassifier(
        activation='relu', max_iter=10000, hidden_layer_sizes=(4, 2))
    model.fit(xs, ys)

    print('score:', model.score(xs, ys))
    print('predictions:', model.predict(xs))
    print('expected:', np.array([0, 1, 1, 1]))


def perceptron_multi_layer_binary_classficate():
    wine_red = pd.read_csv('data/winequality-red.csv', sep=';')
    wine_white = pd.read_csv('data/winequality-white.csv', sep=';')

    wine_red['type'] = 0
    wine_white['type'] = 1

    data = pd.concat([wine_red, wine_white], axis=0)
    data = data.dropna()

    x = data.drop('type', axis=1)
    y = data['type']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()

    scaler = sc.fit(train_x)
    train_x_scaled = scaler.transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                            max_iter=300, activation='relu',
                            solver='adam')

    mlp_clf.fit(train_x_scaled, train_y)

    y_pred = mlp_clf.predict(test_x_scaled)

    print('Accuracy: {:.2f}'.format(accuracy_score(test_y, y_pred)))

    matrix = confusion_matrix(test_y, y_pred, labels=mlp_clf.classes_)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=mlp_clf.classes_)
    matrix_display.plot()
    plt.show()

    print(classification_report(test_y, y_pred))

    plt.plot(mlp_clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    print('Веса между входным и первым скрытым слоем')
    print(mlp_clf.coefs_[0])

    print('Веса между входным и вторым скрытым слоем')
    print(mlp_clf.coefs_[1])

    print('Веса между входным и третьим скрытым слоем')
    print(mlp_clf.coefs_[2])


def perceptron_multi_layer_classficate():
    data = pd.read_excel('data/beans.xlsx')
    x = data.iloc[:, :-1]
    y = data['Class']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()

    scaler = sc.fit(train_x)
    trainX_scaled = scaler.transform(train_x)
    testX_scaled = scaler.transform(test_x)

    mlp_clf = MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                            max_iter=300, activation='relu',
                            solver='adam')

    mlp_clf.fit(trainX_scaled, train_y)

    y_pred = mlp_clf.predict(testX_scaled)

    print('Accuracy: {:.2f}'.format(accuracy_score(test_y, y_pred)))

    matrix = confusion_matrix(test_y, y_pred, labels=mlp_clf.classes_)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=mlp_clf.classes_)
    matrix_display.plot()
    plt.show()

    print(classification_report(test_y, y_pred))

    plt.plot(mlp_clf.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    print('Веса между входным и первым скрытым слоем')
    print(mlp_clf.coefs_[0])

    print('Веса между входным и вторым скрытым слоем')
    print(mlp_clf.coefs_[1])

    print('Веса между входным и третьим скрытым слоем')
    print(mlp_clf.coefs_[2])


def perceptron_digits():
    digits = datasets.load_digits()

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

    plt.show()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    mlp_clf = MLPClassifier(hidden_layer_sizes=(10, 30, 20),
                            max_iter=1000, activation='relu')

    x_train, x_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    mlp_clf.fit(x_train, y_train)
    predicted = mlp_clf.predict(x_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    print(
        f"Classification report for classifier {mlp_clf}:\n"
        f"{classification_report(y_test, predicted)}\n"
    )

    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()


def perceptron_regression():
    x_train = np.arange(0.0, 1, 0.001).reshape(-1, 1)
    y_train = np.sin(2 * np.pi * x_train).ravel()

    nn = MLPRegressor(hidden_layer_sizes=(3),
                      activation='tanh', solver='lbfgs')
    n = nn.fit(x_train, y_train)

    x_test = np.arange(0.0, 1, 0.05).reshape(-1, 1)
    y_test = nn.predict(x_test)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x_train, y_train, s=1, c='b', marker="s", label='real')
    ax1.scatter(x_test, y_test, s=10, c='r', marker="o", label='NN Prediction')
    plt.show()


def perceptron_regression():
    data = pd.read_csv('data/house_prices.csv')
    data = data.drop(['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'], axis=1)

    x = data.drop('price', axis=1)
    y = data['price']

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()

    scaler = sc.fit(train_x)
    train_x_scaled = scaler.transform(train_x)
    test_x_scaled = scaler.transform(test_x)

    mlp_reg = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                           max_iter=300, activation='relu',
                           solver='adam')
    mlp_reg.fit(train_x_scaled, train_y)

    y_pred = mlp_reg.predict(test_x_scaled)
    data_temp = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

    data_temp = data_temp.head(30)
    data_temp.plot(kind='bar', figsize=(10, 6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()

    print('Mean Absolute Error:', mean_absolute_error(test_y, y_pred))
    print('Mean Squared Error:', mean_squared_error(test_y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(test_y, y_pred)))

    plt.plot(mlp_reg.loss_curve_)
    plt.title("Loss Curve", fontsize=14)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

    features = ['sqft_above', 'sqft_living', 'sqft_basement', 'sqft_lot']
    PartialDependenceDisplay.from_estimator(mlp_reg, train_x, features)
    fig = plt.gcf()

    fig.suptitle('Зависимость стоимости дома от параметров.\n'
                 'Используется ИНС MLPRegressor')
    fig.subplots_adjust(hspace=0.3)
    plt.show()

