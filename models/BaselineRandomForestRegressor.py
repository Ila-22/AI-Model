import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

class BaselineRandomForestRegressor:
    def __init__(self, df, target_col, feature_cols=None, test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = feature_cols if feature_cols else [col for col in df.columns if col != target_col]
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self._prepare_data()
        
    def _prepare_data(self):
        self.df = self.df.dropna(subset=self.feature_cols + [self.target_col])
        self.X = self.df[self.feature_cols]
        self.y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train(self):
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [col for col in self.X.columns if col not in categorical_cols]

        preprocessor = ColumnTransformer(transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_cols)
        ])

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=self.random_state))
        ])

        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return {
            "MSE": mse,
            "R^2 Score": r2
        }

    def predict(self, X_new):
        return self.model.predict(X_new)

    def get_model(self):
        return self.model
