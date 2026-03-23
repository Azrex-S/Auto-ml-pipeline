import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
##classif models:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
##regression models:
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
## eval metrics:
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings('ignore')


class AdvancedMLPipeline:
    def __init__(self):
        self.filepath=csv_filepath
        self.df= None
        self.X_train = None
        self.X_test= None
        self.Y_train= None
        self.Y_test= None
        self.results ={}
    ##loading the data :
    def load_data(self):
        print("\nloading data")
        try:
            self.df=pd.read_csv(self.filepath)
            print(f"file loaded:  {self.filepath}")
            print(f"shape : {self.df.shape[0]}rows x {self.df.shape[1]}columns")
            return True
        except FileNotFoundError:
            print(f"file {self.filepath} not found")
            return False
    ## auto detecting the problem type(regression or classif):
    def auto_detect_problem(self):
        self.target_column= self.df.columns[-1]
        target= self.df[self.target_column]
        n_unique= target.nunique()

        if target.dtype=='object':#if text
            self.problem_type='classification'
        elif n_unique<=20:
            self.problem_type='classification'
        else:
            self.problem_type='regression'

    ##slpitting and scaling data
    def split_scale(self):
        X= self.df.drop(self.target_column, axis=1) # all cols except target
        y= self.df[self.target_column]

        self.X_train, self.X_test, self.y_train,self.y_test=train_test_split(
            X,y , test_size=0.2, random_state=42
        )
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_classification_models(self):
        models_to_train={
            'logistic regression ': LogisticRegression(max_iter=1000,random_state=42),
            'random forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'grad boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm(support vector machine)': SVC(probability=True, random_state=42),
            "naive bayes": GaussianNB()
        }
        self.results={}

        for name, model in models_to_train.items():
            print(f"training:{name}")

            model.fit(self.X_train,self.y_train)
            y_pred= model.predict(self.X_test)

        accuracys = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

        self.results[name]={
            'model':model,
            'accuracys': accuracys,
            'precision': recall,
            'f1':f1
        }

    def compare_select(self):

        sorted_res= sorted(
            self.results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for rank ,(name,metrics) in enumerate(sorted_res,1):
               print(f"{rank:<6} {name:<30} {metrics['accuracy']:<12.4f}")
    
        self.best_model_name = sorted_res[0][0]
        self.best_model = sorted_res[0][1]['model']

    def create_visualizations(self):
        import os
        os.makedirs('ml_pipeline_results', exist_ok=True)
    
        if self.problem_type == 'classification':
            self._visualize_classification()
        else:
            self._visualize_regression()

    def save_results_csv(self):
        if self.problem_type=='classification':
            results_list=[]
            for name , metrics in self.results.items():
                results_list.append({
                    'model':name,
                    'accuracys': metrics['accuracys'],
                    'precision': metrics['recall'],
                    'f1 score': metrics['f1']

                })
            results_df = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False)
            results_df.to_csv('ml_pipeline_results/classification_results.csv', index=False)


            