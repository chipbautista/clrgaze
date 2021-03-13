from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


log_reg_search = [
    {'log_reg__penalty': ['l2'],  # l1 is not supported by lbfgs
     'log_reg__class_weight': ['balanced'], 'log_reg__C': [0.1, 1, 10],
     'log_reg__solver': ['lbfgs'], 'log_reg__max_iter': [20000],
     'log_reg__multi_class': ['multinomial']}
]

svm_grid_search = [
    {'svm_rbf__kernel': ['rbf'], 'svm_rbf__gamma': ['auto'],
     'svm_rbf__C': [0.1, 1, 10, 100], 'svm_rbf__class_weight': ['balanced']}
]

lin_svm_grid_search = [
    {'svm_linear__kernel': ['linear'], 'svm_linear__gamma': ['auto'],
     'svm_linear__C': [0.1, 1, 10], 'svm_linear__class_weight': ['balanced']}
]

poly_svm_grid_search = [
    {'svm_poly__kernel': ['poly'], 'svm_poly__degree': [2],
     'svm_poly__gamma': ['auto'],
     'svm_poly__C': [0.1, 1, 10], 'svm_poly__class_weight': ['balanced']}
]

rf_grid_search = [
    {'rf__n_estimators': [50, 100, 500], 'rf__class_weight': ['balanced'],
     'rf__criterion': ['gini', 'entropy']}
]

dt_grid_search = [
    {'dt__criterion': ['gini', 'entropy'], 'dt__class_weight': ['balanced']}
]

nb_grid_search = [{'nb__alpha': [0.1, 1.0]}]

mlp_grid_search = [
    {'mlp__hidden_layer_sizes': [(1000,), (2000,), (1000, 1000,)],
     'mlp__activation': ['relu'], 'mlp__learning_rate': ['adaptive'],
     'mlp__solver': ['adam'], 'mlp__early_stopping': [True]}
]

CLASSIFIER_PARAMS = {
    'svm_linear': (('svm_linear', SVC()), lin_svm_grid_search),
    # 'svm_poly': (('svm_poly', SVC()), poly_svm_grid_search),
    # 'svm_rbf': (('svm_rbf', SVC()), svm_grid_search),
    # 'mlp': (('mlp', MLPClassifier()), mlp_grid_search),
    # 'rf': (('rf', RandomForestClassifier()), rf_grid_search),
    # 'log_reg': (('log_reg', LogisticRegression()), log_reg_search)
}
