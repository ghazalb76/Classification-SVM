from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA


# data_columns = ['months_as_customer', 'age,policy_number', 'policy_bind_date', 'policy_state', 'policy_csl',
#                 'policy_deductable', 'policy_annual_premium', 'umbrella_limit', 'insured_zip', 'insured_sex',
#                 'insured_education_level', 'insured_occupation,insured_hobbies', 'insured_relationship',
#                 'capital-gains', 'capital-loss', 'incident_date', 'incident_type',
#                 'collision_type', 'incident_severity', 'authorities_contacted',
#                 'incident_state', 'incident_city', 'incident_location', 'incident_hour_of_the_day',
#                 'number_of_vehicles_involved', 'property_damage', 'bodily_injuries,witnesses',
#                 'police_report_available', 'total_claim_amount,injury_claim',
#                 'property_claim,vehicle_claim', 'auto_make', 'auto_model', 'auto_year', 'fraud_reported']
def main():
    interesting_features = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
                            'umbrella_limit', 'insured_zip', 'insured_sex', 'insured_education_level',
                            'insured_occupation', 'insured_hobbies', 'insured_relationship',
                            'capital-gains', 'capital-loss', 'incident_type', 'incident_severity',
                            'authorities_contacted', 'number_of_vehicles_involved', 'property_damage',
                            'bodily_injuries', 'witnesses', 'police_report_available',
                            'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
                            'auto_year', 'fraud_reported']

    data = pd.read_csv('insurance_claims.csv', header=0)
    data = data[interesting_features]

    # نمونه برداری
    yes_rows = data[data.fraud_reported == 'Y']
    no_rows = data[(data.fraud_reported == 'N')]
    no_rows = no_rows.sample(len(yes_rows.values), replace=False)
    yes_rows = yes_rows.append(no_rows, ignore_index=True)
    print('تعداد کل داده ها {0} است و تعداد نمونه های انتخاب شد {1}'.format(len(data), len(yes_rows)))
    data = yes_rows
    print(data)

    # پیش پردازشی
    data['capital-loss'] = pd.cut(data['capital-loss'], bins=2)
    data['capital-gains'] = pd.cut(data['capital-gains'], bins=2)
    data.umbrella_limit = pd.cut(data.umbrella_limit, bins=2)
    data.insured_zip = pd.cut(data.insured_zip, bins=3)
    data.policy_deductable = pd.cut(data.policy_deductable, bins=3)
    data.number_of_vehicles_involved = pd.cut(data.number_of_vehicles_involved, bins=3)
    data.months_as_customer = pd.qcut(data.months_as_customer, 5)
    data.age = pd.qcut(data.age, 5)
    data.vehicle_claim = pd.qcut(data.vehicle_claim, 7)
    data.total_claim_amount = pd.qcut(data.total_claim_amount, 7)
    data.policy_annual_premium = pd.qcut(data.policy_annual_premium, 7)
    data.injury_claim = pd.qcut(data.injury_claim, 7)
    data.property_claim = pd.qcut(data.property_claim, 7)

    # انتخاب داده های آموزشی و آزمون
    columns = list(data.columns.values)
    X = data[columns[:-1]]
    y = data[columns[-1]].replace('Y', 1).replace('N', 0)
    X = pd.get_dummies(X, columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=42)

    svm_kernel_comparison(X_train, X_test, y_train, y_test)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for k in kernels:
     clf = svm.SVC(kernel=k)
     clf.fit(X_train, y_train)
     disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap=plt.cm.Blues)
     disp.ax_.set_title('Normalized Confusion Matrix\nkernel = {0}'.format(k))

     # GridSearchCV
    best_param = svc_param_selection(X_train, y_train)
    print(best_param)
    clf = svm.SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
    clf.fit(X_train, y_train)
    confusion_matrix(clf, X_test, y_test, 'Best Params\n{0}'.format(best_param))
    default_clf = svm.SVC()
    default_clf.fit(X_train, y_train)
    plot_roc_curves(X_test, y_test, default_clf, clf)
    plot_principal_component_analysis(X, y)
    plt.show()

def svm_kernel_comparison(X_train, X_test, y_train, y_test):
    kernel = ('linear', 'poly', 'rbf', 'sigmoid')
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    precs = []
    for k in kernel:
        for c in Cs:
            for g in gammas:
                clf = svm.SVC(kernel=k, C=c, gamma=g)
                clf.fit(X_train, y_train)
                test_score = clf.score(X_test, y_test)
                train_score = clf.score(X_train, y_train)
                precs.append([k, c, g, round(test_score, 2), round(train_score, 2)])
    for p in precs:
        print(p)

def svc_param_selection(X, y):
    kernel = ('linear', 'poly', 'rbf', 'sigmoid')
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3)
    grid_search.fit(X, y)
    return grid_search.best_params_

def plot_roc_curves(X_test, y_test, clf, best_clf):
    lw = 2
    y_score1 = clf.decision_function(X_test)
    fpr1, tpr1, _ = roc_curve(y_test, y_score1, pos_label=clf.classes_[1])
    roc_auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, color='orange', lw=lw, label='ROC curve (area = {0})'.format(round(roc_auc1, 2)))
    y_score2 = best_clf.decision_function(X_test)
    fpr2, tpr2, _ = roc_curve(y_test, y_score2, pos_label=best_clf.classes_[1])
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, color='green', lw=lw, label='ROC curve (area = {0})'.format(round(roc_auc2, 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.title('Receiver operating characteristic')
    plt.legend()
    plt.show()


def plot_principal_component_analysis(X, y):
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)
    X = MinMaxScaler().fit_transform(X=X)
    h = .02
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, shading='auto', cmap=plt.cm.Accent, alpha=0.8)
    colors = []
    for i in y_train:
        if i == 1:
            colors.append('green')
        else:
            colors.append('red')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolors='k')
    plt.title('Decision Surface of 2-D Principal Component Analysis')
    plt.axis('tight')
    red_patch = mpatches.Patch(color='red', label='NO')
    green_patch = mpatches.Patch(color='green', label='YES')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()


def confusion_matrix(clf, X_test, y_test, desc):
    disp = plot_confusion_matrix(clf, X_test, y_test, normalize='true', cmap=plt.cm.Greens)
    title = "Normalized confusion matrix\n{0}".format(desc)
    disp.ax_.set_title(title)
    plt.show()


if __name__ == '__main__':
    main()
