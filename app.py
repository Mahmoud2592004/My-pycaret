import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
import seaborn as sns
import plotly.express as px

class my_pycaret:
    def __init__(self, df):
        """
        Initialize the my_pycaret object with a DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be processed.
        """
        self.df = df

    def remove_nulls(self, method=None):
        """
        Remove or handle null values in the DataFrame.

        Parameters:
        method (str, optional): The method to handle null values. Defaults to None.
            Options:
            - 'drop': Drops rows containing null values.
            - 'forward': Forward fills null values.
            - 'backward': Backward fills null values.
            - 'interpolate': Interpolates numeric columns.
            - 'mean': Imputes missing values using the mean of each column.
            - 'mode': Imputes missing values using the mode (most frequent value) of each column.
            - auto: Fills missing values with the mean for numeric columns, then fills remaining null values with the mode (most frequent value).

        Returns:
        pandas.DataFrame: The DataFrame with null values handled as specified.

        Raises:
        ValueError: If an invalid method is provided.

        Example:
        >>> df_handler = my_pycaret(df)
        >>> df_cleaned = df_handler.remove_nulls(method='drop')
        """
        
        if method == 'drop':
            self.df.dropna(inplace=True)
        elif method == 'forward':
            self.df.fillna(method='ffill', inplace=True)
        elif method == 'backward':
            self.df.fillna(method='bfill', inplace=True)
        elif method == 'interpolate':
            self.df.interpolate(inplace=True)
        elif method == 'mean':
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                self.df[column].fillna(np.mean(self.df[column]),inplace=True)
        elif method == 'mode':
            for column in self.df.columns:
                if self.df[column].isnull().any():
                    mode_value = self.df[column].mode()[0] 
                    self.df[column].fillna(mode_value, inplace=True)
        elif method == 'auto':
            # Fill numeric columns with the mean
            numeric_columns = []
            for column in self.df.columns:
                if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                    numeric_columns.append(column)
            for column in numeric_columns:
                self.df[column].fillna(np.mean(self.df[column]),inplace=True)
            
            # Fill categorical columns with the mode
            for column in self.df.columns:
                if self.df[column].isnull().any():
                    mode_value = self.df[column].mode()[0] 
                    self.df[column].fillna(mode_value, inplace=True)  
        else:
            st.exception(ValueError("Invalid method specified. Choose from the following options:\n"
                            "drop: Drops the rows containing null values\n"
                            "forward: Forward fills null values\n"
                            "backward: Backward fills null values\n"
                            "interpolate: Interpolates numeric columns\n"
                            "mean: Imputes missing values using the mean of each column\n"
                            "mode: Imputes missing values using the mode (most frequent value) of each column\n"
                            "auto: Fills missing values with the mean for numeric columns and the mode for remaining null values"))

        return self.df

    def remove_outliers(self, column_name=None):
        """
        Remove outliers from the DataFrame based on the Interquartile Range (IQR) method.

        This method detects and removes outliers in a specified column or all numeric columns
        of the DataFrame using the IQR method. Outliers are defined as values that fall outside
        of 1.5 times the IQR below the first quartile (Q1) or above the third quartile (Q3).

        Parameters:
        column_name (str, optional): The name of the column to remove outliers from. 
                                    If None, outliers will be removed from all numeric columns. 

        Returns:
        pandas.DataFrame: The DataFrame with outliers removed.

        Raises:
        ValueError: If the specified column does not exist or is not numeric.

        Example:
        >>> df_handler = my_pycaret(df)
        >>> df_cleaned = df_handler.remove_outliers(column_name='A')
        """
        if column_name is None:
            # Remove outliers from all numeric columns
            numeric_columns = []
            for column in self.df.columns:
                if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                    numeric_columns.append(column)
            for column in numeric_columns:
                Q1 = np.quantile(self.df[column], 0.25)
                Q3 = np.quantile(self.df[column], 0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        else:
            if column_name not in self.df.columns:
                st.exception(ValueError(f"Column '{column_name}' does not exist in the DataFrame."))

            if not pd.api.types.is_numeric_dtype(self.df[column_name]):
                st.exception(ValueError(f"Column '{column_name}' is not numeric. Please select a numeric column."))

            # Remove outliers from the specified numeric column
            Q1 = np.quantile(self.df[column_name], 0.25)
            Q3 = np.quantile(self.df[column_name], 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column_name] >= lower_bound) & (self.df[column_name] <= upper_bound)]

        return self.df

    def encode(self, method, column_name=None):
        """
        Encode categorical variables in the DataFrame.

        Parameters:
        method (str): The encoding method. Options are 'label' or 'one hot'.
        column_name (str, optional): The name of the column to encode. If None, all categorical columns will be encoded.

        Returns:
        pandas.DataFrame: The DataFrame with encoded categorical variables.

        Raises:
        ValueError: If an invalid method is provided or the specified column does not exist or is not categorical.

        Example:
        >>> df_handler = my_pycaret(df)
        >>> df_encoded = df_handler.encode(method='label', column_name='category_column')
        """
        categorical_columns = []
        for column in self.df.columns:
            if was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                categorical_columns.append(column)
        if column_name is None:
            for column in categorical_columns:
                if method == 'label':
                    label = LabelEncoder()
                    self.df[column] = label.fit_transform(self.df[column])
                    label_mapping = {label: code for label, code in zip(label.classes_, label.transform(label.classes_))}
                    st.write(f'For column {column}:\n{label_mapping}')

                elif method == 'one hot':
                    ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
                    ohe_encoded = ohe.fit_transform(self.df[[column]])
                    ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out([column]))
                    self.df = pd.concat([self.df, ohe_encoded_df], axis=1).drop(columns=[column])

                else:
                    st.exception(ValueError(f"Method: {method} is not supported, please choose 'label' or 'one hot'."))
        else:
            if column_name not in self.df.columns:
                st.exception(ValueError(f"Column '{column_name}' does not exist in the DataFrame."))
            elif column_name not in categorical_columns:
                st.exception(ValueError(f"Column '{column_name}' is not categorical or boolean. Please select a categorical column."))
            if method == 'label':
                label = LabelEncoder()
                self.df[column_name] = label.fit_transform(self.df[column_name])
                label_mapping = {label: code for label, code in zip(label.classes_, label.transform(label.classes_))}
                st.write(f'For column {column_name}:\n{label_mapping}')

            elif method == 'one hot':
                ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
                ohe_encoded = ohe.fit_transform(self.df[[column_name]])
                ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out([column_name]))
                self.df = pd.concat([self.df, ohe_encoded_df], axis=1).drop(columns=[column_name])
                
            else:
                st.exception(ValueError(f"Method: {method} is not supported, please choose 'label' or 'one hot'."))

        return self.df

    
    def scale (self):
        numeric_columns = []
        scl = StandardScaler()
        for column in self.df.columns:
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                numeric_columns.append(column)
        self.df[numeric_columns] = scl.fit_transform(self.df[numeric_columns])
        return self.df

    def preprocess(self, encode_method, na_method=None):
        """
        Preprocess the DataFrame by handling null values, removing outliers, encoding categorical variables, and scaling numeric variables.

        Parameters:
        encode_method (str): The encoding method for categorical variables. Options are 'label' or 'one hot'.
        na_method (str, optional): The method to handle null values. Defaults to None.
            Options:
            - 'drop': Drops rows containing null values.
            - 'forward': Forward fills null values.
            - 'backward': Backward fills null values.
            - 'interpolate': Interpolates numeric columns.
            - 'mean': Imputes missing values using the mean of each column.
            - 'mode': Imputes missing values using the mode (most frequent value) of each column.
            - auto: Fills missing values with the mean for numeric columns, then fills remaining null values with the mode (most frequent value).

        Returns:
        pandas.DataFrame: The preprocessed DataFrame.

        Raises:
        ValueError: If an invalid method is provided for handling null values or encoding.

        Example:
        >>> df_handler = my_pycaret(df)
        >>> df_preprocessed = df_handler.preprocess(encode_method='label', na_method='mean')
        """
        numeric_columns = []
        for column in self.df.columns:
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                numeric_columns.append(column)

        categorical_columns = []
        for column in self.df.columns:
            if was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                categorical_columns.append(column)

        # Handle null values
        if na_method == 'auto':
            # Remove nan values with mean for numeric columns and mode for categorical columns
            for column in numeric_columns:
                self.df[column].fillna(np.mean(self.df[column]),inplace=True)
            for column in categorical_columns:
                if self.df[column].isnull().any():
                    mode_value = self.df[column].mode()[0] 
                    self.df[column].fillna(mode_value, inplace=True)
        elif na_method == 'drop':
            self.df.dropna(inplace=True)
        elif na_method == 'forward':
            self.df.fillna(method='ffill', inplace=True)
        elif na_method == 'backward':
            self.df.fillna(method='bfill', inplace=True)
        elif na_method == 'interpolate':
            self.df.interpolate(inplace=True)
        elif na_method == 'mean':
            for column in numeric_columns:
                self.df[column].fillna(np.mean(self.df[column]),inplace=True)
        elif na_method == 'mode':
            for column in self.df.columns:
                if self.df[column].isnull().any():
                    mode_value = self.df[column].mode()[0] 
                    self.df[column].fillna(mode_value, inplace=True)
        else:
            st.exception(ValueError(f"Method: {na_method} is not defined. Please choose one of the following:\n'drop', 'forward', 'backward', 'interpolate', 'mean', 'mode', or auto"))

        # Remove outliers
        for column in numeric_columns:
            Q1 = np.quantile(self.df[column], 0.25)
            Q3 = np.quantile(self.df[column], 0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
        
        # Scale the data
        scl = StandardScaler()
        self.df[numeric_columns] = scl.fit_transform(self.df[numeric_columns])

        # Encode using Label or One hot
        if encode_method == 'label':
            for column in categorical_columns:
                label = LabelEncoder()
                encoded_labels = label.fit_transform(self.df[column])
                self.df.drop(column, axis=1, inplace=True)
                self.df[column] = encoded_labels
                label_mapping = {label: code for label, code in zip(label.classes_, label.transform(label.classes_))}
                st.write(f'For column {column}:\n{label_mapping}')
        elif encode_method == 'one hot':
            for column in categorical_columns:
                ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')
                ohe_encoded = ohe.fit_transform(self.df[[column]])
                ohe_encoded_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out([column]))
                self.df = pd.concat([self.df, ohe_encoded_df], axis=1).drop(columns=[column])
        else:
            st.exception(ValueError(f"Method: {encode_method} is not defined. Please choose 'label' or 'one hot'"))
            
        return self.df
    
    def eda(self,chart_type,x_axis_column=None,y_axis_column=None,hue=None,bins=None,annot=False,fit_reg = False):
        if chart_type == 'Scatter':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column) and not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=y_axis_column):
                fig, ax = plt.subplots()
                ax.scatter(self.df[x_axis_column],self.df[y_axis_column])
                ax.set_xlabel(f'{x_axis_column}')
                ax.set_ylabel(f'{y_axis_column}')
                ax.set_title(f'{x_axis_column} and {y_axis_column} scatter plot')
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"Both x and y axis must be numeric. Please choose numeric columns"))
        elif chart_type == 'Plot':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column) and not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=y_axis_column):
                fig, ax = plt.subplots()
                ax.plot(self.df[x_axis_column],self.df[y_axis_column])
                ax.set_xlabel(f'{x_axis_column}')
                ax.set_ylabel(f'{y_axis_column}')
                ax.set_title(f'{x_axis_column} and {y_axis_column} plot')
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"Both x and y axis must be numeric. Please choose numeric columns"))
        elif chart_type == 'Count plot':
            if was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column) and was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=hue):
                fig, ax = plt.subplots()
                sns.countplot(x=x_axis_column, hue=hue, data=self.df, ax=ax)
                ax.set_title(f'Count plot of {x_axis_column}' + (f' by {hue}' if hue != 'None' else ''))
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"x and hue must be categorical. PLease choose categorical columns"))
        elif chart_type == 'Histogram':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column):
                fig, ax = plt.subplots()
                ax.hist(self.df[x_axis_column], bins=bins)
                ax.set_xlabel(f'{x_axis_column}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Histogram of {x_axis_column}')
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"Feature must be numeric. PLease choose numeric column"))
        elif chart_type == 'Box plot':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column):
                fig, ax = plt.subplots()
                ax.boxplot(self.df[x_axis_column])
                ax.set_xlabel(f'{x_axis_column}')
                ax.set_ylabel('Values')
                ax.set_title(f'Box plot of {x_axis_column}')
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"Feature must be numeric. PLease choose numeric column"))
        elif chart_type == 'Violinplot':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column):
                fig, ax = plt.subplots()
                ax.violinplot(self.df[x_axis_column])
                ax.set_xlabel(f'{x_axis_column}')
                ax.set_ylabel('Values')
                ax.set_title(f'Box plot of {x_axis_column}')
                st.pyplot(fig)
            else:
                st.exception(ValueError(f"Feature must be numeric. PLease choose numeric column"))
        elif chart_type == 'Pie chart':
            if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column) and was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=y_axis_column):
                fig = px.pie(data_frame=self.df,values=self.df[x_axis_column],names=self.df[y_axis_column])
                st.plotly_chart(fig)
            else:
                st.error(ValueError(f"Values must be numeric and categories must be categorical."))
        elif chart_type == 'Bar chart':
            if was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=x_axis_column) and not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=y_axis_column):
                fig = px.bar(data_frame=self.df,x=self.df[x_axis_column],y=self.df[y_axis_column])
                st.plotly_chart(fig)
            else:
                st.error(ValueError(f"Values must be numeric and categories must be categorical."))
        elif chart_type == 'Heat map':
            numeric_columns = []
            for column in self.df.columns:
                if not was_categorical(df=self.df,original_categorical_columns=original_categorical_columns,target_column=column):
                    numeric_columns.append(column)
            new_df = self.df[numeric_columns]
            if not new_df.empty:
                fig,ax = plt.subplots()
                sns.heatmap(new_df.corr(), annot=annot, ax=ax)
                st.pyplot(fig)
            else:
                st.error(ValueError(f"Your dataframe does not contain numeric columns"))
        else:
            st.error(f"{chart_type} is not defined, please choose one of the following chart types: 'Scatter','Plot','Histogram','Box plot','Violinplot','Count plot','Pie chart','Bar chart','Heat map'")

    
    
    
    def classification_model(self, model_name, target_column):
        """
        Build and train a classification model.

        Parameters:
        model_name (str): The name of the classification model.
            Options are 'logistic regression', 'random forest', 'support vector',
            'k-nearest neighbor', 'decision tree', or 'xgb'.
        target_column (str): The name of the target column in the DataFrame.

        Returns:
        model: The best trained classification model.

        Raises:
        ValueError: If an invalid model name is provided or if the target_column
            does not exist in the DataFrame.
        """
        if target_column not in self.df.columns:
            st.exception(ValueError(f"Column '{target_column}' does not exist in the DataFrame."))
        
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        if model_name == 'logistic regression':
            model = LogisticRegression()
        elif model_name == 'random forest':
            param = {
                    'max_depth': list(range(1,10)),
                    'n_estimators': [1,10,100]
                    }
            clf = RandomForestClassifier()
            gsclf = GridSearchCV(clf,param)
            gsclf.fit(x_train, y_train)
            model = gsclf.best_estimator_
        elif model_name == 'support vector':
            param = {'kernel': ['linear', 'poly', 'rbf']}
            clf = SVC()
            gsclf = GridSearchCV(clf,param)
            gsclf.fit(x_train, y_train)
            model = gsclf.best_estimator_
        elif model_name == 'k-nearest neighbor':
            param = {'n_neighbors': [1,3,5,7,9,11,13,15],
                    'weights': ['uniform','distance']}
            clf = KNeighborsClassifier()
            gsclf = GridSearchCV(clf,param)
            gsclf.fit(x_train, y_train)
            model = gsclf.best_estimator_
        elif model_name == 'decision tree':
            param = {'criterion': ['gini', 'entropy'],
                    'max_depth': list(range(1,12))
                    }
            clf = DecisionTreeClassifier()
            gsclf = GridSearchCV(clf,param)
            gsclf.fit(x_train, y_train)
            model = gsclf.best_estimator_
        elif model_name == 'xgb':
            param = {'eta': [0.1,0.3,0.5,0.7,0.9],
                    'max_depth': list(range(1,10)),
                    'gamma': [0.5,0,1.5]
                    }
            clf = XGBClassifier()
            gsclf = GridSearchCV(clf,param)
            gsclf.fit(x_train, y_train)
            model = gsclf.best_estimator_
        else:
            st.exception(ValueError(f"Model '{model_name}' is not supported. Please choose 'logistic regression', 'random forest', 'support vector', 'k-nearest neighbor', 'desicion tree', or 'xgb'."))

        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        st.write(f'Accuracy for {model_name}: {accuracy}')
        st.write(f'Precision for {model_name}: {precision}')
        st.write(f"Recall for {model_name}: {recall}")
        st.write(f"Confusion Matrix for {model_name}:")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cmd.plot(ax=ax)
        st.pyplot(fig)
        return model
    
    def best_calssification_model(self, target_column):
        """
        Trains multiple classification models and returns the best performing model based on accuracy.

        Parameters:
        target_column (str): The name of the target column in the DataFrame.

        Returns:
        model: The best performing classification model based on accuracy.

        Raises:
        ValueError: If the specified target_column does not exist in the DataFrame.
        """
        if target_column not in self.df.columns:
            st.exception((f"Column '{target_column}' does not exist in the DataFrame."))
        
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        accuracies = {}
        
        # Train logistic regression model 
        lg = LogisticRegression()
        lg.fit(x_train, y_train)
        lg_pred = lg.predict(x_test)
        lg_accuracy = accuracy_score(y_test, lg_pred)
        accuracies[lg] = lg_accuracy
        
        # Train random forest
        param = {
                'max_depth': list(range(1,10)),
                'n_estimators': [1,10,100]
                }
        clf = RandomForestClassifier()
        gsclf = GridSearchCV(clf,param)
        gsclf.fit(x_train, y_train)
        rf = gsclf.best_estimator_
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        accuracies[rf] = rf_accuracy
        
        # Train support vector machine
        param = {'kernel': ['linear', 'poly', 'rbf']}
        clf = SVC()
        gsclf = GridSearchCV(clf,param)
        gsclf.fit(x_train, y_train)
        svm = gsclf.best_estimator_
        svm.fit(x_train, y_train)
        svm_pred = svm.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        accuracies[svm] = svm_accuracy
        
        # Train k-nearest neighbor
        param = {'n_neighbors': [1,3,5,7,9,11,13,15]}
        clf = KNeighborsClassifier()
        gsclf = GridSearchCV(clf,param)
        gsclf.fit(x_train, y_train)
        knn = gsclf.best_estimator_
        knn.fit(x_train, y_train)
        knn_pred = knn.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_pred)
        accuracies[knn] = knn_accuracy
        
        # Train decision tree
        param = {'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1,12))
        }
        clf = DecisionTreeClassifier()
        gsclf = GridSearchCV(clf,param)
        gsclf.fit(x_train, y_train)
        dt = gsclf.best_estimator_
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        dt_accuracy = accuracy_score(y_test, dt_pred)
        accuracies[dt] = dt_accuracy
        
        # Train xgb
        param = {'eta': [0.1,0.3,0.5,0.7,0.9],
        'max_depth': list(range(1,10)),
        'gamma': [0.5,0,1.5]
        }
        clf = XGBClassifier()
        gsclf = GridSearchCV(clf,param)
        gsclf.fit(x_train, y_train)
        xgb = gsclf.best_estimator_
        xgb.fit(x_train, y_train)
        xgb_pred = xgb.predict(x_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        accuracies[xgb] = xgb_accuracy
        
        # Choose the best model in terms of accuracy
        sorted_accuracies = sorted(accuracies.items(), key=lambda item: item[1], reverse=True)
        container = st.container(border=True)
        for model, accuracy in sorted_accuracies:
            container.write(f"{model}: {accuracy}")
        model, highest_accuracy = sorted_accuracies[0]
        return model
    
    def regression_model(self, model_name, target_column):
        """
        Build and train a regression model.

        Parameters:
        model_name (str): The name of the regression model.
            Options are 'linear regression', 'random forest', 'lasso', 'ridge'
            'k-nearest neighbor', 'decision tree', 'gradient descent', or 'xgb'.
        target_column (str): The name of the target column in the DataFrame.

        Returns:
        model: The best trained regression model.

        Raises:
        ValueError: If an invalid model name is provided or if the target_column
            does not exist in the DataFrame.
        """
        if target_column not in self.df.columns:
            st.exception((f"Column '{target_column}' does not exist in the DataFrame."))
        
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        if model_name == 'linear regression':
            model = LinearRegression()
        elif model_name == 'random forest':
            param = {
                    'max_depth': list(range(1,10)),
                    'n_estimators': [1,10,100]
                    }
            rf = RandomForestRegressor()
            gsreg = GridSearchCV(rf,param)
            gsreg.fit(x_train, y_train)
            model = gsreg.best_estimator_
        elif model_name == 'lasso':
            param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]}
            lr = Lasso()
            gsreg = GridSearchCV(lr,param)
            gsreg.fit(x_train,y_train)
            model = gsreg.best_estimator_
        elif model_name == 'ridge':
            param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]}
            lr = Ridge()
            gsreg = GridSearchCV(lr,param)
            gsreg.fit(x_train,y_train)
            model = gsreg.best_estimator_
        elif model_name == 'gradient descent':
            param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5],
                    'max_iter' : [1,5,10,50,100,200]
                    }
            sgdr = SGDRegressor()
            gsreg = GridSearchCV(sgdr,param)
            gsreg.fit(x_train,y_train)
            model = gsreg.best_estimator_
        elif model_name == 'k-nearest neighbor':
            param = {'n_neighbors': [1,3,5,7,9,11,13,15],
                    'weights': ['uniform','distance']}
            knn = KNeighborsRegressor()
            gsreg = GridSearchCV(knn,param)
            gsreg.fit(x_train,y_train)
            model = gsreg.best_estimator_
        elif model_name == 'decision tree':
            param = {'criterion': ['absolute_error', 'squared_error','poisson'],
            'max_depth': list(range(1,12))
            }
            dt = DecisionTreeRegressor()
            gsreg = GridSearchCV(dt,param)
            gsreg.fit(x_train,y_train)
            model = gsreg.best_estimator_
        elif model_name == 'xgb':
            param = {'eta': [0.1,0.3,0.5,0.7,0.9],
                    'max_depth': list(range(1,10)),
                    'gamma': [0.5,0,1.5]
                    }
            xgb = XGBRegressor()
            gsreg = GridSearchCV(xgb,param)
            gsreg.fit(x_train, y_train)
            model = gsreg.best_estimator_
        else:
            st.exception((f"Model '{model_name}' is not supported. Please choose 'linear regression', 'random forest', 'lasso', 'ridge', 'gradient descent','k-nearest neighbor', 'desicion tree', or 'xgb'."))
        
        model.fit(x_train ,y_train)
        pred = model.predict(x_test)
        meanSquaredError = mean_squared_error(y_test,pred)
        r2score = r2_score(y_test,pred)
        st.write(f"Mean squared error for {model_name}: {meanSquaredError}")
        st.write(f"R2 score for {model_name}: {r2score}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(y_test)), y_test, color='red', label='Actual')
        ax.plot(range(len(pred)), pred, color='blue', label='Predictions')
        ax.set_title('Actual vs Predictions')
        ax.set_xlabel('Index')
        ax.set_ylabel('Values')
        ax.legend()
        st.pyplot(fig)
        return model
    
    def best_regression_model(self,target_column):
        """
        Trains multiple Regression models and returns the best performing model based on mean squared error.

        Parameters:
        target_column (str): The name of the target column in the DataFrame.

        Returns:
        model: The best performing regression model based on mean squared error.

        Raises:
        ValueError: If the specified target_column does not exist in the DataFrame.
        """
        if target_column not in self.df.columns:
            st.exception((f"Column '{target_column}' does not exist in the DataFrame."))
        
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        meanSquaredErrors = {}
        
        # Train linear regression model 
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        lr_pred = lr.predict(x_test)
        lr_error = mean_squared_error(y_test, lr_pred)
        meanSquaredErrors[lr] = lr_error
        
        # Train lasso
        param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]}
        reg = Lasso()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train,y_train)
        ls = gsreg.best_estimator_
        ls.fit(x_train ,y_train)
        ls_pred = ls.predict(x_test)
        ls_error = mean_squared_error(y_test, ls_pred)
        meanSquaredErrors[ls] = ls_error
        
        # Train ridge
        param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5]}
        reg = Ridge()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train,y_train)
        ridge = gsreg.best_estimator_
        ridge.fit(x_train ,y_train)
        ridge_pred = ridge.predict(x_test)
        ridge_error = mean_squared_error(y_test, ridge_pred)
        meanSquaredErrors[ridge] = ridge_error  
        
        # Train gradient descent
        param = {'alpha': [0.1,0.3,0.5,0.7,0.9,1,1.5],
        'max_iter' : [1,5,10,50,100,200]
        }
        reg = SGDRegressor()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train,y_train)
        sgdr = gsreg.best_estimator_
        sgdr.fit(x_train, y_train)
        sgdr_pred = sgdr.predict(x_test)
        sgdr_error = mean_squared_error(y_test, sgdr_pred)
        meanSquaredErrors[sgdr] = sgdr_error
        
        # Train random forest
        param = {
                'max_depth': list(range(1,10)),
                'n_estimators': [1,10,100]
                }
        reg = RandomForestRegressor()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train, y_train)
        rf = gsreg.best_estimator_
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        rf_error = mean_squared_error(y_test, rf_pred)
        meanSquaredErrors[rf] = rf_error
                
        # Train k-nearest neighbor
        param = {'n_neighbors': [1,3,5,7,9,11,13,15]}
        reg = KNeighborsRegressor()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train, y_train)
        knn = gsreg.best_estimator_
        knn.fit(x_train, y_train)
        knn_pred = knn.predict(x_test)
        knn_error = mean_squared_error(y_test, knn_pred)
        meanSquaredErrors[knn] = knn_error
        
        # Train decision tree
        param = {'criterion': ['absolute_error', 'squared_error','poisson'],
            'max_depth': list(range(1,12))
            }
        reg = DecisionTreeRegressor()
        gscreg = GridSearchCV(reg,param)
        gsreg.fit(x_train, y_train)
        dt = gsreg.best_estimator_
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        dt_error = mean_squared_error(y_test, dt_pred)
        meanSquaredErrors[dt] = dt_error
        
        # Train xgb
        param = {'eta': [0.1,0.3,0.5,0.7,0.9],
        'max_depth': list(range(1,10)),
        'gamma': [0.5,0,1.5]
        }
        reg = XGBRegressor()
        gsreg = GridSearchCV(reg,param)
        gsreg.fit(x_train, y_train)
        xgb = gsreg.best_estimator_
        xgb.fit(x_train, y_train)
        xgb_pred = xgb.predict(x_test)
        xgb_error = mean_squared_error(y_test, xgb_pred)
        meanSquaredErrors[xgb] = xgb_error
        
        # Choose the best model in terms of mean squared error
        sorted_meanSquaredErrors = sorted(meanSquaredErrors.items(), key=lambda item: item[1], reverse=False)
        container = st.container(border=True)
        for model, error in sorted_meanSquaredErrors:
            container.write(f"{model}: {error}")
        model, lowest_error = sorted_meanSquaredErrors[0]
        return model


def was_categorical(df, target_column, original_categorical_columns):
    """
    Determine if the target_column is categorical.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the target_column.
    target_column (str): The column name to check.
    original_categorical_columns (list): List of original categorical columns.

    Returns:
    bool: True if the column is categorical, False otherwise.
    """
    # Check if the column is in the list of known categorical columns
    if target_column in original_categorical_columns or "_".join(target_column.split("_")[:-1]) in original_categorical_columns:
        return True
    
    # Check the data type of the column
    if target_column in df.columns:
        col = df[target_column]
        # Check if the column is of object or category type
        if col.dtype == 'object' or col.dtype.name == 'category':
            return True
        
        # Heuristic based on unique values
        num_unique_values = col.nunique()
        total_values = len(col)
        if num_unique_values < total_values * 0.1:
            return True

    return False


original_categorical_columns = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'first_run' not in st.session_state:
    st.session_state.first_run = True

# File uploader
dataset = st.file_uploader("Upload your dataset", type=['xlsx', 'csv', 'json'])
if dataset is not None:
    if dataset.name.split('.')[-1] == 'xlsx':
        df = pd.read_excel(dataset)
    elif dataset.name.split('.')[-1] == 'csv':
        df = pd.read_csv(dataset)
    elif dataset.name.split('.')[-1] == 'json':
        df = pd.read_json(dataset)
    else:
        st.exception('Unsupported file type')
    
    # Update session state with the uploaded dataset
    if st.session_state.first_run:
        st.session_state.df = df
        st.session_state.first_run = False
        original_categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
        st.success("File uploaded successfully")
        st.write(st.session_state.df.head())
    
    preprocess_tab, eda_tab, build_model_tab = st.tabs(['Preprocessing','EDA','Build your model'])


    pyc = my_pycaret(df=st.session_state.df)


    with preprocess_tab:
        method = st.selectbox("Choose preprocessing method", ['Remove nulls', 'Remove outliers', 'Scale the data', 'Encode', 'Auto preprocess'])

        if method:
            preprocess_function = None

            if method == 'Remove nulls':
                remove_nulls = st.selectbox("Choose nulls removal method", ['drop', 'forward', 'backward', 'interpolate', 'mean', 'mode', 'auto'])
                preprocess_function = lambda: pyc.remove_nulls(method=remove_nulls)

            elif method == 'Remove outliers':
                remove_outliers = st.selectbox("Choose outliers removal method", ['all columns', 'specific column'])
                if remove_outliers == 'all columns':
                    preprocess_function = lambda: pyc.remove_outliers()
                elif remove_outliers == 'specific column':
                    column_name = st.selectbox("Choose the column you want to remove outliers from", df.select_dtypes(include=['number']).columns)
                    preprocess_function = lambda: pyc.remove_outliers(column_name=column_name)

            elif method == 'Scale the data':
                preprocess_function = lambda: pyc.scale()

            elif method == 'Encode':
                encode = st.selectbox("Choose encoding method", ['label', 'one hot'])
                preprocess_function = lambda: pyc.encode(method=encode)

            elif method == 'Auto preprocess':
                col1, col2 = st.columns(2)
                with col1:
                    remove_nulls = st.selectbox("Choose nulls removal method", ['drop', 'forward', 'backward', 'interpolate', 'mean', 'mode', 'auto'])
                with col2:
                    encode = st.selectbox("Choose encoding method", ['label', 'one hot'])
                preprocess_function = lambda: pyc.preprocess(encode_method=encode, na_method=remove_nulls)

            start_button = st.button("Do it")
            if start_button and preprocess_function:
                with st.spinner("Processing your data frame"):
                    st.session_state.df = preprocess_function()
                    st.write(st.session_state.df.head())
            elif start_button:
                st.error("No preprocessing method selected")

    # EDA tab
    with eda_tab:
        st.write(st.session_state.df.head())
        chart_type = st.selectbox('Choose the type of chart you want to see',['Scatter','Plot','Histogram','Box plot','Violinplot','Count plot','Pie chart','Bar chart','Heat map'])
        if chart_type is not None:
            if chart_type == 'Scatter' or chart_type == 'Plot':
                col1,col2 = st.columns(2)
                with col1 :
                    x = st.selectbox('Choose the x axis',st.session_state.df.columns)
                with col2:
                    y = st.selectbox('Choose the y axis',st.session_state.df.columns)
                draw_chart = st.button("Get your chart")
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,y_axis_column=y)
            elif chart_type == 'Count plot':
                col1,col2 = st.columns(2)
                with col1 :
                    x = st.selectbox('Choose the x axis',st.session_state.df.columns)
                with col2:
                    hue = st.selectbox('Choose the hue',st.session_state.df.columns)
                draw_chart = st.button("Get your chart")
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,hue=hue)
            elif chart_type == 'Histogram':
                x = st.selectbox('Choose the feature',st.session_state.df.columns)
                bins = st.slider('Choose the number of bins',min_value=1,max_value=100)
                draw_chart = st.button("Get your chart")
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,bins=bins)
            elif chart_type == 'Box plot' or chart_type == 'Violinplot':
                x = st.selectbox('Choose the feature',st.session_state.df.columns)
                draw_chart = st.button("Get your chart")
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x)
            elif chart_type == 'Pie chart':
                col1,col2 = st.columns(2)
                with col1:
                    x = st.selectbox('Choose the values (Numeric)',st.session_state.df.columns)
                with col2:
                    y = st.selectbox('Choose the categories (Categorical)',st.session_state.df.columns)
                draw_chart = st.button('Get your chart')
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,y_axis_column=y)
            elif chart_type == 'Bar chart':
                col1,col2 = st.columns(2)
                with col1:
                    x = st.selectbox('Choose the categories (Categorical)',st.session_state.df.columns)
                with col2:
                    y = st.selectbox('Choose the values (Numeric)',st.session_state.df.columns)
                draw_chart = st.button('Get your chart')
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,y_axis_column=y)
            elif chart_type == 'Heat map':
                annot = st.toggle(label='Annotations')
                draw_chart = st.button('Get your chart')
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,annot=annot)
            elif chart_type == 'lmplot':
                col1,col2,col3 = st.columns(3)
                with col1:
                    x = st.selectbox("Choose the x axis",st.session_state.df.columns)
                with col2:
                    y = st.selectbox("Choose the y axis",st.session_state.df.columns)
                with col3:
                    hue = st.selectbox("Choose the hue",st.session_state.df.columns)
                fit_reg = st.toggle(label='Fit regression line')
                draw_chart = st.button('Get your chart')
                if draw_chart:
                    with st.spinner("Drawing your chart"):
                        pyc.eda(chart_type=chart_type,x_axis_column=x,y_axis_column=y,hue=hue,fit_reg=fit_reg)

    # Build Model tab
    with build_model_tab:
        st.write(st.session_state.df.head())
        classification_model_options = ['logistic regression', 'random forest', 'support vector', 'k-nearest neighbor', 'desicion tree', 'xgb']
        regression_model_options = ['linear regression', 'random forest', 'lasso', 'ridge', 'gradient descent','k-nearest neighbor', 'decision tree', 'xgb']
        target_column = st.selectbox("Target column",df.columns)
        if target_column is not None:
            model_building_method = st.selectbox('Choose how you want to build the model',['Create Specific model','Best model'])
            categorical = was_categorical(df=df,original_categorical_columns=original_categorical_columns,target_column=target_column)
            if model_building_method == 'Create Specific model' and categorical:
                model_name = st.selectbox("Choose a classifcation model to build", classification_model_options)
                build_model_button = st.button("Get your model")
                if build_model_button:
                    with st.spinner("Building your model"):
                        model = pyc.classification_model(model_name=model_name,target_column=target_column)
                    st.success(f"Your best model is {model}")
            elif model_building_method == 'Best model' and categorical:
                build_model_button = st.button("Get your model")
                if build_model_button:
                    with st.spinner("Building the best model for you"):
                        model = pyc.best_calssification_model(target_column=target_column)
                    st.success(f"Your best model is {model}")
            elif model_building_method == 'Create Specific model' and not categorical:
                model_name = st.selectbox("Choose a regression model to build", regression_model_options)
                build_model_button = st.button("Get your model")
                if build_model_button:
                    with st.spinner("Building your model"):
                        model = pyc.regression_model(model_name=model_name,target_column=target_column)
                    st.success(f"Your best model is {model}")
            elif model_building_method == 'Best model' and not categorical:
                build_model_button = st.button("Get your model")
                if build_model_button:
                    with st.spinner("Building the best model for you"):
                        model = pyc.best_regression_model(target_column=target_column)
                    st.success(f"Your best model is {model}")

else:
    st.session_state.df = None
    st.session_state.first_run = True


