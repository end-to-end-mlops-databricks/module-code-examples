from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class PriceModel:
    def __init__(self, preprocessor, config):
        self.config = config
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=config['parameters']['n_estimators'],
                max_depth=config['parameters']['max_depth'],
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def get_feature_importance(self):
        feature_importance = self.model.named_steps['regressor'].feature_importances_
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        return feature_importance, feature_names