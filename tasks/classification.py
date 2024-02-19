import pandas
from sklearn import tree
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier


def k_neighbors_classification():
    # чтение данных из excel файла
    data = pandas.read_excel('data/beans.xlsx')

    # распределение переменных
    plt.rcParams['figure.figsize'] = (30, 25)
    (data.select_dtypes(include=['number'])).plot(kind='hist', bins=17, subplots=True, layout=(9, 2))
    plt.show()

    # выделение признаков и метки
    x = data.iloc[:, :-1]
    y = data['Class']

    # деление признаков и метки на тренировочные и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100)

    # создание модели
    model = KNeighborsClassifier(15)

    # обучение модели
    model.fit(x_train, y_train)

    # созднание предсказания
    prediction = model.predict(x_test)

    # создание матрицы ошибок
    matrix = confusion_matrix(y_test, prediction, labels=model.classes_)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)

    # вывод матрицы ошибок
    matrix_display.plot()
    plt.show()

    # получение оценки модели
    score = accuracy_score(y_test, prediction)

    # вывод оценки модели
    print(score)

    # вывод оценки качества модели
    print(classification_report(y_test, prediction))

    # получение матрицы корреляции
    correlation_matrix = (data.select_dtypes(include=['number'])).corr()

    # вывод матрицы корреляции
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, linecolor="black", fmt=".4f", ax=ax)
    plt.show()

    # вывод графика семян по значениям их периметра и формы
    barbunya = data.loc[data['Class'] == 'BARBUNYA']
    dermason = data.loc[data['Class'] == 'DERMASON']
    sira = data.loc[data['Class'] == 'SIRA']
    bombay = data.loc[data['Class'] == 'BOMBAY']
    horoz = data.loc[data['Class'] == 'HOROZ']
    seker = data.loc[data['Class'] == 'SEKER']
    cali = data.loc[data['Class'] == 'CALI']

    plt.figure(figsize=(10, 10))
    plt.scatter(barbunya.Perimeter, barbunya.ShapeFactor3, color='red', label='BARBUNYA')
    plt.scatter(dermason.Perimeter, dermason.ShapeFactor3, color='green', label='DERMASON')
    plt.scatter(sira.Perimeter, sira.ShapeFactor3, color='blue', label='SIRA')
    plt.scatter(bombay.Perimeter, bombay.ShapeFactor3, color='orange', label='BOMBAY')
    plt.scatter(horoz.Perimeter, horoz.ShapeFactor3, color='purple', label='HOROZ')
    plt.scatter(seker.Perimeter, seker.ShapeFactor3, color='pink', label='SEKER')
    plt.scatter(cali.Perimeter, cali.ShapeFactor3, color='black', label='CALI')
    plt.xlabel('MajorAxisLength')
    plt.ylabel('MinorAxisLength')
    plt.legend()
    plt.show()


def decision_tree_classification():
    # чтение данных из excel файла
    data = pandas.read_excel('data/beans.xlsx')

    # вывод графика распределения по классам
    sns.countplot(data['Class'])
    plt.show()

    # распределение переменных
    plt.rcParams['figure.figsize'] = (30, 25)
    (data.select_dtypes(include=['number'])).plot(kind='hist', bins=17, subplots=True, layout=(9, 2))
    plt.show()

    # выделение признаков и метки
    x = data.iloc[:, :-1]
    y = data['Class']

    # деление признаков и метки на тренировочные и тестовые наборы
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100)

    # создание модели
    model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)

    # обучение модели
    model.fit(x_train, y_train)

    # создание предсказания
    prediction = model.predict(x_test)

    # создание матрицы ошибок
    matrix = confusion_matrix(y_test, prediction, labels=model.classes_)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)

    # вывод матрицы ошибок
    matrix_display.plot()
    plt.show()

    # вывод оценки точности модели
    print(accuracy_score(y_test, prediction))

    # вывод оценки качества модели
    print(classification_report(y_test, prediction, zero_division=0))

    # получение уникальных значений метки и названий признаков соответственно
    target = list(data['Class'].unique())
    feature_names = list(x.columns)

    # вывод дерева решений
    plt.figure(figsize=(20, 10))  # Устанавливаем размер фигуры (по желанию)
    tree.plot_tree(model, fontsize=12)
    plt.show()

