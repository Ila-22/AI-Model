from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



class GradBoostModel():

    def __init__(self,
                 categorical_cols,
                 numeric_cols,
                 n_estimators=500,
                 learning_rate=0.05,
                 max_depth=3,
                 validation_fraction=0.05,
                 n_iter_no_change=10,
                 tol=1e-4):
        
        # Store column settings
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        
        # 1) define the preprocessor
        self.preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('scale', StandardScaler()),
            ]), numeric_cols),
        ], remainder='drop')

        # 2) define the estimator
        self.estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol
        )

        # 3) stitch into a pipeline
        self.pipeline = Pipeline([
            ('prep', self.preprocessor),
            ('est',  self.estimator)
        ])

    def fit(self, X, y):
        """Fit the full pipeline."""
        return self.pipeline.fit(X, y)

    def predict(self, X):
        """Generate predictions (one‐step ahead)."""
        return self.pipeline.predict(X)
    



    def get_model(self):
        model = GradientBoostingRegressor(n_estimators=500, 
                                          learning_rate=0.05, 
                                          max_depth=3,
                                          validation_fraction=0.05, 
                                          n_iter_no_change=10, 
                                          tol=1e-4)
        return model