"""
Teste 2 - Selecao de features via SHAP, implementada como um sklearn Transformer para
que o k (numero de features selecionadas) seja escolhido dentro do treino de cada fold
via GridSearchCV (a mesma maquina de CV aninhada ja usada para PCA em common_cv.py) -
nunca observando o dataset inteiro. fit() treina um modelo base logreg no treino do
fold, calcula SHAP nesse MESMO treino, ranqueia por |SHAP| medio e guarda os indices
top-k; transform() so aplica o subconjunto de colunas ja decidido no fit.
"""
import numpy as np
import shap
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
K_GRID = [10, 20, 50, 100]


class SHAPTopKSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=20, random_state=SEED):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        base = LogisticRegression(max_iter=5000, random_state=self.random_state)
        base.fit(X, y)
        explainer = shap.LinearExplainer(base, X)
        sv = explainer(X)
        mean_abs_shap = np.abs(sv.values).mean(axis=0)
        k = min(self.k, X.shape[1])
        self.selected_idx_ = np.argsort(mean_abs_shap)[::-1][:k]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]


def build_shap_classifier_grids():
    """
    Fresh Pipeline/grid each call (GridSearchCV clones internally anyway, but avoids any
    shared-mutable-state surprise across repeated CLI invocations in the same process).
    """
    return {
        "shap_logreg": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SHAPTopKSelector()),
                ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
            ]),
            {"selector__k": K_GRID, "clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }
