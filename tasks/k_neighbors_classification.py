import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def k_neighbors_classify():
    data = pd.read_excel('data/beans.xlsx')

    plt.rcParams['figure.figsize'] = (8, 8)
    (data.select_dtypes(include=['number'])).plot(kind='hist', bins=17, subplots=True, layout=(9, 2))
    plt.show()

    x = data.iloc[:, :-1]
    y = data['Class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=100)

    model = KNeighborsClassifier(15)
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)

    matrix = confusion_matrix(y_test, prediction, labels=model.classes_)
    matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=model.classes_)
    matrix_display.plot()
    plt.show()

    score = accuracy_score(y_test, prediction)
    print(score)
    print(classification_report(y_test, prediction))

    correlation_matrix = (data.select_dtypes(include=['number'])).corr()
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, linecolor="black", fmt=".4f", ax=ax)
    plt.show()

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
