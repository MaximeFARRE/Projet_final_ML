import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_models(
    prices,
    features,
    tickers,
    train_end_date,
    C=1.0,
    max_iter=1000,
    use_grid_search=True,
):
    models = {}
    meta = {}

    # temporal separation 
    for ticker in tickers:

        # Columns of features related to this ticker
        prefix = f"{ticker}_"
        feature_cols = [c for c in features.columns if c.startswith(prefix)]
        feature_cols = [c for c in feature_cols if not c.endswith("_PRICE")]

        if len(feature_cols) == 0:
            continue

        X = features[feature_cols]

        # Label : 1 if the next day's yield is positive
        ret_col = f"{ticker}_RET"
        if ret_col not in features.columns:
            continue

        y = (features[ret_col].shift(-1) > 0).astype(int)

        data = pd.concat([X, y], axis=1).dropna()
        X = data[feature_cols]
        y = data[ret_col]

        X_train = X[X.index <= train_end_date]
        y_train = y[y.index <= train_end_date]

        if X_train.empty:
            continue

        if use_grid_search:
            param_grid = {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            }

            base_model = LogisticRegression(max_iter=max_iter)
            grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)

        models[ticker] = model
        meta[ticker] = {
            "features": feature_cols,
            "train_end_date": train_end_date,
        }

    return models, meta
