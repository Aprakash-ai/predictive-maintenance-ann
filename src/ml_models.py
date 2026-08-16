from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def build_logistic_regression():

    #logistic regression classifier

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    return model

def build_decision_tree():

    # decision tree classifier

    model = DecisionTreeClassifier(
        random_state=42
    )

    return model

def build_random_forest():

    # random forest classifier

    model = RandomForestClassifier(
        random_state=42
    )

    return model

def build_svm():
    # svm classifier

    model = SVC(
        random_state=42
    )

    return model

def build_xgboost():

    # xgboost classifier

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    return model