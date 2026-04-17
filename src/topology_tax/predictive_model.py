import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

FEATURE_KEYS = [
    "edge_density", "diameter", "clustering_coefficient",
    "algebraic_connectivity", "spectral_gap", "avg_path_length",
    "degree_entropy", "max_betweenness",
]


class TopologyPredictor:
    def __init__(self, model_type="linear"):
        self.model_type = model_type
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.r_squared = None
        self.feature_names = FEATURE_KEYS

    def _to_matrix(self, features):
        rows = []
        for feat in features:
            row = [100.0 if feat.get(key, 0.0) == float("inf") else feat.get(key, 0.0) for key in FEATURE_KEYS]
            rows.append(row)
        return np.array(rows)

    def fit(self, features, targets):
        X = self._to_matrix(features)
        y = np.array(targets)
        self.model.fit(X, y)
        self.r_squared = float(r2_score(y, self.model.predict(X)))

    def predict(self, features):
        return self.model.predict(self._to_matrix(features))

    def feature_importance(self):
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)
        else:
            return {}
        return dict(zip(FEATURE_KEYS, importances.tolist()))

    def cross_validate(self, features, targets, cv=5):
        X = self._to_matrix(features)
        y = np.array(targets)
        return cross_val_score(self.model, X, y, cv=cv, scoring="r2").tolist()
