from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier


def random_forest_classify():
    data = load_iris()

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=23)

    clf = OneVsRestClassifier(RandomForestClassifier())
    clf.fit(x_train, y_train)
    y_prediction_proba = clf.predict_proba(x_test)

    roc_auc = roc_auc_score(y_test, y_prediction_proba, multi_class='ovr')
    print('Roc Auc оценка: ', roc_auc)

    colors = ['orange', 'red', 'green']
    for i in range(len(data.target_names)):
        fpr, tpr, thresh = roc_curve(y_test, y_prediction_proba[:, i], pos_label=i)
        plt.plot(fpr, tpr, linestyle='--', color=colors[i], label=data.target_names[i] + ' против Остальных')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Мультикласс Бобы ROC кривая (Случайный лес)')
    plt.xlabel('Частота ложноположительных результатов')
    plt.ylabel('Истинный положительный показатель')
    plt.legend()
    plt.show()


def decision_tree_classify():
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=23)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    y_prediction_proba = clf.predict_proba(x_test)

    roc_auc = roc_auc_score(y_test, y_prediction_proba, multi_class='ovr')
    print(f'Roc Auc оценка: {roc_auc}')

    colors = ['orange', 'red', 'green']
    for i in range(len(data.target_names)):
        fpr, tpr, thresh = roc_curve(y_test, y_prediction_proba[:, i], pos_label=i)
        plt.plot(fpr, tpr, linestyle='--', color=colors[i], label=data.target_names[i] + ' против Остальных')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Мультикласс Бобы ROC кривая (Дерево решений)')
    plt.xlabel('Частота ложноположительных результатов')
    plt.ylabel('Истинный положительный показатель')
    plt.legend()
    plt.show()