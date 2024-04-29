import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

class ChurnPredictionModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()

    def prepare_data(self):
        # Drop specified columns
        self.df = self.df.drop(['id','Unnamed: 0', 'CustomerId', 'Surname', 'Geography'], axis=1)
        
        # Convert Age to integer
        self.df['Age'] = self.df['Age'].astype('int')
        
        # Fill missing values in CreditScore with median
        median_credit_score = self.df['CreditScore'].median()
        self.df['CreditScore'].fillna(median_credit_score, inplace=True)
        
        # Replace Gender with numerical values
        self.df['Gender'] = self.df['Gender'].replace({'Female': 0, 'Male': 1})
        
        # Separate features and target
        self.input_df = self.df.drop('churn', axis=1)
        self.output_df = self.df['churn']

        # Scale features
        self.scaler = RobustScaler()
        self.input_df = self.scaler.fit_transform(self.input_df)

        # Oversample using SMOTE
        smote = SMOTE(random_state=42)
        self.df_input_oversampled, self.df_output_oversampled = smote.fit_resample(self.input_df, self.output_df)

        # Split data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df_input_oversampled,
                                                                                self.df_output_oversampled,
                                                                                test_size=0.2,
                                                                                random_state=42)

    def train_random_forest(self):
        # Define hyperparameters for Random Forest
        parameters = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2, 4, 6, 8],
        }

        # Grid search for best parameters
        rf = RandomForestClassifier()
        self.rf_grid_search = GridSearchCV(rf, param_grid=parameters, scoring='accuracy', cv=5)
        self.rf_grid_search.fit(self.x_train, self.y_train)

        # Train Random Forest with best parameters
        best_params_rf = self.rf_grid_search.best_params_
        self.rf_best = RandomForestClassifier(**best_params_rf)
        self.rf_best.fit(self.x_train, self.y_train)

    def train_xgboost(self):
        # Define hyperparameters for XGBoost
        param_grid_xgb = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
        }

        # Grid search for best parameters
        xgb = XGBClassifier()
        self.xgb_grid_search = GridSearchCV(xgb, param_grid=param_grid_xgb, scoring='accuracy', cv=3)
        self.xgb_grid_search.fit(self.x_train, self.y_train)

        # Train XGBoost with best parameters
        best_params_xgb = self.xgb_grid_search.best_params_
        self.xgb_best = XGBClassifier(**best_params_xgb)
        self.xgb_best.fit(self.x_train, self.y_train)

    def evaluate_models(self):
        # Evaluate Random Forest
        y_pred_rf = self.rf_best.predict(self.x_test)
        print("\nRandom Forest Classification Report:\n")
        print(classification_report(self.y_test, y_pred_rf, target_names=['0', '1']))

        # Evaluate XGBoost
        y_pred_xgb = self.xgb_best.predict(self.x_test)
        print("\nXGBoost Classification Report:\n")
        print(classification_report(self.y_test, y_pred_xgb, target_names=['0', '1']))

