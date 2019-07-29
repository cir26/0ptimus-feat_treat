# Feature treatment class
# Use to optimize feature selection, sampling, and hyperparameter tuning for model testing by creating instances of feature treatments
# Stream-lines testing process and brings together packages like scikit-learn, skopt, pandas, numpy
# returns performance metrics for treatments with visual

# Run Time:
# RandomForestClassifier = 4.5 iterations/min
# XGBClassifier = 5.2 iterations/min

# required arguments:
# X = feature pandas dataframe long format
# y = target pandas dataframe long format
# random_state = seed for pseudo random number generator to be used throughout treatments

# main dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import floor, ceil, pi
import copy
from random import randint
# scikit tools
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve,auc,cohen_kappa_score,accuracy_score,roc_auc_score,roc_curve,brier_score_loss,confusion_matrix,f1_score,recall_score,precision_score,matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# sampling tools
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN

# self dependencies
import Feat_Treat.static
import Feat_Treat.validation

class feat_treat(Feat_Treat.validation.validation, Feat_Treat.static.static):
    def __init__(self,X,y,random_state):
        self.X=X
        self.y=y
        self.random_state=random_state
        self.encoded=[]
        self.na_col='update with check_na(percent_threshold,axis=1)'
        self.na_row='update with check_na(percent_threshold,axis=0)'
        self.metrics='No metrics available'
        self.hyperparameters='No hyperparameters available'


        # remove observations where DV is missing
        mask = y.isnull()
        missing_dv = [i for i in range(0, y.shape[0]) if mask[i]==True]
        if len(missing_dv)>0:
            self.y = self.y.drop(index=missing_dv)
            self.X = self.X.drop(index=missing_dv)
            print("The following observations were removed due to missing dependent variable: ", missing_dv)

        # Low variance filter
        # check variance of each column
        # remove column if variance is less than var_threshold
        drop=[]
        var_threshold = 0.004975
        for col in self.X.columns:
            if self.X[col].dtype!='object':
                if np.var(self.X[col]) < var_threshold:
                    drop.append(col)
        self.X=self.X.drop(columns=drop)
        # GET TO KNOW YOUR DATA SET
        # return dataframe indices for categorical datatypes
        obj = [self.X.columns.get_loc(i) for i in self.X.columns if self.X[i].dtype == 'object']
        boolean = [self.X.columns.get_loc(i) for i in self.X.columns if self.X[i].dtype == 'bool']
        date = [self.X.columns.get_loc(i) for i in self.X.columns if self.X[i].dtype == 'datetime64']
        print("non-numeric column indices:")
        print('object type: ', obj)
        print('boolean type: ', boolean)
        print('date type: ', date, " \n")


        # return base rate
        if y.dtype=='object' or y.dtype=='dtype('O')':
            y_class_count = y.astype(str)
        unique_elements, counts_elements = np.unique(y_class_count, return_counts=True)
        DV_classes_df = pd.DataFrame({'Class': unique_elements,
                                      'Count': counts_elements})
        print(DV_classes_df.to_string(index=False),"\n ")
        print("Base rate: ",min(counts_elements)/sum(counts_elements))


    def copy(self,n=1):
        if n==1:
            return copy.deepcopy(self)
        elif n>1:
            # copy instance n times
            copies=[]
            for i in range(0,n):
                copies.append(copy.deepcopy(self))
            return tuple(copies)


    def check_na(self,percent_threshold=0, axis=1):
        if axis==1:
            num_col=self.X.shape[1]
            num_obs=self.X.shape[0]
            # column check for missing values
            percent_missing = self.X.isnull().sum() * 100 / num_obs
            column_name = [percent_missing.index[i] for i in range(0,num_col) if percent_missing[i]>percent_threshold]
            percent_na = [percent_missing[i] for i in range(0,num_col) if percent_missing[i]>percent_threshold]
            missing_value_df = pd.DataFrame({'column_name': column_name,
                                             'percent_na': percent_na})
            missing_value_df.sort_values('percent_na', inplace=True)
            print(missing_value_df, "\n ")
            self.na_col=column_name
            print("Index stored in: self.na_col")
        if axis==0:
            num_col=self.X.shape[1]
            num_obs=self.X.shape[0]
            # row check for missing values
            percent_missing = self.X.isnull().sum(axis=1) * 100 / (num_col)
            row_index = [percent_missing.index[i] for i in range(0,num_obs) if percent_missing[i]>percent_threshold]
            percent_na = [percent_missing[i] for i in range(0,num_obs) if percent_missing[i]>percent_threshold]
            missing_value_df = pd.DataFrame({'row_index': row_index,
                                             'percent_na': percent_na})
            missing_value_df.sort_values('percent_na', inplace=True)
            print(missing_value_df, "\n ")
            self.na_row=row_index
            print("Index stored in: self.na_row")


    def handle_na(self,strategy=None,index=None,axis=1):
        # handle missing values
        # methods: mean, median, mode, value, remove
        if axis==1:
            if isinstance(strategy,int)==True or isinstance(strategy,float)==True:
                for i in index:
                    self.X[i]=self.X[i].fillna(strategy)
            else:
                strategy1=''.join(strategy.split()).lower()
                if strategy1=='mean':
                    for i in index:
                        mean=self.X[i].mean()
                        self.X[i].fillna(mean,inplace=True)
                elif strategy1=='median':
                    for i in index:
                        median=self.X[i].median()
                        self.X[i].fillna(median,inplace=True)
                elif strategy1=='mode':
                    for i in index:
                        mode = self.X[i].mode()[0]
                        self.X[i].fillna(mode,inplace=True)
                elif strategy1=='remove':
                    self.X.drop(columns=index,inplace=True)
                elif strategy1=='random':
                    for i in index:
                        uniques = np.unique(self.X[i])
                        uniques = uniques[~np.isnan(uniques)]
                        for j in range(0,len(self.X[i])):
                            if np.isnan(self.X.loc[self.X.index[j],i])==True:
                                self.X.loc[self.X.index[j],i] = np.random.choice(uniques)
    #             elif strategy1=='mice': # mice will perform impute on all columns
    #                 from impyute.imputation.cs import mice
    #                 col=self.X.columns
    #                 X_imp=mice(self.X)
    #                 self.X=pd.DataFrame(X_imp, columns=col)
                else:
                    for i in index:
                        self.X[i].fillna(strategy,inplace=True)
        elif axis==0:
             self.X.drop(index=index, axis=0,inplace=True)


    def pcc_filter(self,k):
        if isinstance(k,int)==True or isinstance(k,float)==True:
            # check pearson coefficient for linear correlation between DV and each feature
            # remove column if PCC is less than correlation limit
            df=pd.concat([self.y, self.X], axis=1, ignore_index=True)
#           create pearson correlation coefficient matrix
            cor = df.corr()
            corr_limit=k
            cor_target = abs(cor.iloc[:,0])
            relevant_features = cor_target[cor_target>corr_limit]
            relevant_features=relevant_features.iloc[1:]
            rel_col =[]
            for i in relevant_features.index:
                rel_col.append(i-1)
#           drop all columns except relevant ones
            self.X=self.X.iloc[:,rel_col]
#           check which columns are kept
            keep_col=self.X.columns
            print(keep_col)
            print(len(keep_col))
#       section to include pcc optimization loop
        else:
            k = ''.join(k.split()).lower()
            k = k[:3]
            if k == 'opt':
                pass



    def collinear(self,k):
#       check pearson coefficient for linear correlation between features (collinearity)
#       remove if PCC is greater than collinear limit
#       recreate pearson correlation matrix
        cor = self.X.corr()
        feat_targets=[]
        feat_remove=[]
        ignore=[]
#       set correlation limit
        colinear_corr_limit=k
#       check for collinearity
        for i in self.X.columns:
            cor_target = abs(cor.loc[:,i])
            feat_targets.append((i,cor_target[cor_target>colinear_corr_limit]))
        # remove collinear features
        for i in feat_targets:
            ignore.append(i[0])
            for j in i[1].index:
                if j not in ignore:
                    feat_remove.append(j)

        feat_remove=set(feat_remove)
        self.X=self.X.drop(columns=feat_remove)
        # check which columns are kept
        keep_col=self.X.columns
        print("columns remaining: ",keep_col)
        print(len(keep_col), " columns")

    def check_cat(self):
        categorical_columns=[i for i in self.X.columns if (self.X[i].dtype != 'float64' and self.X[i].dtype != 'int64')]
        print(categorical_columns)

    def encode(self, strategy=None, cat_col=None):
        if cat_col==None:
            cat_col=[i for i in self.X.columns if (self.X[i].dtype != 'float64' and self.X[i].dtype != 'int64')]
        else:
            pass
        if(strategy=='dummy'):
            self.X=pd.get_dummies(self.X, columns=cat_col)
            cat_col = [i for i in cat_col if i not in self.encoded]
            for i in cat_col:
                self.encoded.append(i)
        elif strategy == None or strategy == 'label':
            if len(cat_col)>0:
                le = preprocessing.LabelEncoder()
                for col in cat_col:
                    self.X[col]=le.fit_transform(self.X[col])
                cat_col = [i for i in cat_col if i not in self.encoded]
                for i in cat_col:
                    self.encoded.append(i)
            else:
                print("No categorical variables identified in data set")


    def rfe(self,n=None,cum=None,rfe_model=None):
        if isinstance(n,int)==True and cum==None:
            if rfe_model==None:
                from sklearn.linear_model import LogisticRegression
                rfe_model=LogisticRegression(solver='lbfgs')
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                selector = RFECV(estimator=rfe_model, min_features_to_select = n, cv=kfold, n_jobs=-1).fit(self.X,self.y)
                keep = [i for i in range(0,len(selector.support_)) if selector.support_[i]==True]
                self.X=self.X.iloc[:,keep]
            else:
                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                selector = RFECV(estimator=rfe_model, min_features_to_select = n, cv=kfold, n_jobs=-1).fit(self.X,self.y)
                keep = [i for i in range(0,len(selector.support_)) if selector.support_[i]==True]
                self.X=self.iloc[:,keep]
        elif isinstance(cum,float)==True and n==None:
            # cumulative feature importance
            pass
        else:
            n = ''.join(n.split()).lower()
            n = n[:3]
            if n == 'opt':
                if rfe_model==None:
                    if self.X.shape[1]>1000:
                        steps=int((round(floor(self.X.shape[1]),-3)/1000)*8)
                    else:
                        steps=1
                    nof_list=np.arange(self.X.shape[1]-1,1,step=-steps)
                    #print(nof_list)
                    check_point=np.arange(1,self.X.shape[1]-1,step=floor(0.1*self.X.shape[1]))
                    #print(check_point)
                    high_score=0
                    #Variable to store the optimum features
                    n_best=0
                    score_list =[]
                    X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size = 0.2, random_state = self.random_state)
                    print("Optimizing...")
                    for i in nof_list:
                        #print("Testing n = ",i)
                        num_col=X_train.shape[1]
                        from sklearn.linear_model import LogisticRegression
                        rfe_model=LogisticRegression(solver='lbfgs')
                        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                        selector = RFECV(estimator=rfe_model, min_features_to_select = i, cv=kfold, n_jobs=-1).fit(X_train,y_train)
                        X_train_rfe=selector.transform(X_train)
                        cols_kept = selector.get_support(indices=True)
                        drop_cols=set(np.arange(0,num_col))-set(cols_kept)
                        drop_cols=list(drop_cols)
                        X_test_rfe=selector.transform(X_test)
                        model=LogisticRegression(solver='lbfgs')
                        model.fit(X_train_rfe,y_train)
                        preds = model.predict(X_test_rfe)
                        score = roc_auc_score(y_test, preds)
                        score_list.append(score)
                        if(score>high_score):
                            high_score = score
                            n_best = i
                        if i in check_point:
                            print(" Best n so far: {} \n Score: {} \n".format(n_best,high_score))
                        X_train=X_train.drop(X_train.columns[drop_cols],axis=1)
                        X_test=X_test.drop(X_test.columns[drop_cols],axis=1)
                    print("Optimal n: {} \n Score: {} \n".format(n_best,high_score))
                    rfe_model=LogisticRegression(solver='lbfgs')
                    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                    selector = RFECV(estimator=rfe_model, min_features_to_select = n_best, cv=kfold, n_jobs=-1).fit(self.X,self.y)
                    keep = [i for i in range(0,len(selector.support_)) if selector.support_[i]==True]
                    self.X=self.X.iloc[:,keep]


                else:
                    #allow for other model support
                    pass


    def skb(self,k,score_func=None):
        cat_col=[self.X.columns.get_loc(i) for i in self.X.columns if self.X[i].dtype == 'object']
        if isinstance(k,int)==True or isinstance(k,float)==True:
            if score_func==None:
                from sklearn.feature_selection import chi2
                skb =SelectKBest(score_func=chi2, k=k).fit(self.X,self.y)
                self.X = self.X.iloc[:,skb.get_support(indices=True)]
            else:
                from sklearn.feature_selection import score_func
                skb = SelectKBest(score_func=score_func, k=k).fit(self.X,self.y)
                self.X = self.X.iloc[:,skb.get_support(indices=True)]
#           section to include optimization loop
        else:
            k = ''.join(k.split()).lower()
            k = k[:3]
            if k == 'opt':
                if self.X.shape[1]>1000:
                    steps=int((round(floor(self.X.shape[1]),-3)/1000)*2)
                else:
                    steps=1
                kof_list=np.arange(self.X.shape[1]-1,1,step=-steps)
                print(kof_list)
                check_point=np.arange(1,self.X.shape[1]-1,step=floor(0.1*self.X.shape[1]))
                print(check_point)
                high_score=0
                #Variable to store the optimum features
                k_best=0
                score_list =[]
                X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size = 0.2, random_state = self.random_state)
                print("Optimizing...")
                for i in kof_list:
                    print("Testing n = ",i)
                    if score_func==None:
                        from sklearn.feature_selection import chi2
                        skb = SelectKBest(score_func=chi2, k=i).fit(X_train,y_train)
                    else:
                        from sklearn.feature_selection import score_func
                        skb = SelectKBest(score_func=chi2, k=i).fit(X_train,y_train)
                    X_train_skb = X_train.iloc[:,skb.get_support(indices=True)]
                    from sklearn.linear_model import LogisticRegression
                    model=LogisticRegression(solver='liblinear')
                    model.fit(X_train_skb,y_train)
                    preds = model.predict(X_test_skb)
                    score = roc_auc_score(y_test, preds)
                    score_list.append(score)
                    if(score>high_score):
                        high_score = score
                        k_best = i
                    if i in check_point:
                        print(" Best k so far: {} \n Score: {} \n".format(k_best,high_score))
                print("Optimal k: {} \n Score: {} \n".format(k_best,high_score))
                skb = SelectKBest(score_func=chi2, k=k_best).fit(self.X,self.y)
                self.X = self.X.iloc[:,skb.get_support(indices=True)]



    def svd(self,n=None, var=None):
        if var==None and isinstance(n,int)==True:
            col = self.X.columns
            svd = TruncatedSVD(n_components=n, n_iter=5, random_state=self.random_state).fit(self.X)
            self.X = pd.DataFrame(svd.transform(self.X), columns=['SV %i' % i for i in range(n)], index=self.X.index)
        elif n==None and isinstance(var,float)==True:
            n=min(self.X.shape[1], self.X.shape[0])
            exp_var = 0
            svd = TruncatedSVD(n_components=n, n_iter=5, random_state=self.random_state).fit(self.X)
            j=0
            exp_var=0
            while exp_var<var and j<n:
                exp_var = exp_var + svd.explained_variance_ratio_[j]
                j=j+1
            svd = TruncatedSVD(n_components=j, n_iter=5, random_state=self.random_state).fit(self.X)
            self.X = pd.DataFrame(svd.transform(self.X), columns=['SV %i' % i for i in range(j)], index=self.X.index)
#       section to include optimization loop
        else:
            n = ''.join(n.split()).lower()
            n = n[:3]
            if n == 'opt':
                max_n=floor(0.8*min(self.X.shape[1], self.X.shape[0]))
                nof_list=np.arange(max_n,1,step=-1)
                #print(nof_list)
                check_point=np.arange(1,max_n-1,step=floor(0.1*max_n))
                #print(check_point)
                high_score=0
                #Variable to store the optimum features
                n_best=0
                score_list =[]
                X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size = 0.2, random_state = self.random_state)
                print("Optimizing...")
                for i in nof_list:
                    svd = TruncatedSVD(n_components=i, n_iter=5, random_state=self.random_state)
                    X_train_svd=svd.fit_transform(X_train)
                    X_test_svd=svd.transform(X_test)
                    from sklearn.linear_model import LogisticRegression
                    model=LogisticRegression(solver='liblinear')
                    model.fit(X_train_svd,y_train)
                    preds = model.predict(X_test_svd)
                    score = roc_auc_score(y_test, preds)
                    score_list.append(score)
                    if(score>high_score):
                        high_score = score
                        n_best = i
                    if i in check_point:
                        print(" Best n so far: {} \n Score: {} \n".format(n_best,high_score))
                print("Optimal n: {} \n Score: {} \n".format(n_best,high_score))
                svd = TruncatedSVD(n_components=n_best, n_iter=5, random_state=self.random_state).fit(self.X)
                self.X = pd.DataFrame(svd.transform(self.X), columns=['SV %i' % i for i in range(n_best)], index=self.X.index)


    def pca(self,n=None,var=None):
        if var==None and isinstance(n,int)==True:
            pca = PCA(n_components=n, svd_solver='auto', random_state=self.random_state).fit(self.X)
            self.X = pd.DataFrame(pca.transform(self.X), columns=['PCA %i' % i for i in range(n)], index=self.X.index)
        elif n==None and isinstance(var,float)==True:
            n=min(self.X.shape[1], self.X.shape[0])
            exp_var = 0
            pca = PCA(n_components=n, svd_solver='auto', random_state=self.random_state).fit(self.X)
            j=0
            exp_var=0
            while exp_var<var and j<n:
                exp_var = exp_var + pca.explained_variance_ratio_[j]
                j=j+1
            pca = PCA(n_components=j, random_state=self.random_state).fit(self.X)
            self.X = pd.DataFrame(pca.transform(self.X), columns=['PCA %i' % i for i in range(j)], index=self.X.index)
#       section to include optimization loop
        else:
            n = ''.join(n.split()).lower()
            n = n[:3]
            if n == 'opt':
                max_n=floor(0.8*min(self.X.shape[1], self.X.shape[0]))
                nof_list=np.arange(max_n-1,1,step=-1)
                #print(nof_list)
                check_point=np.arange(1,max_n-1,step=floor(0.1*max_n))
                #print(check_point)
                high_score=0
                #Variable to store the optimum features
                n_best=0
                score_list =[]
                X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, test_size = 0.2, random_state = self.random_state)
                print("Optimizing...")
                for i in nof_list:
                    #print("Testing n = ",i)
                    pca = PCA(n_components=i, random_state=self.random_state)
                    X_train_pca=pca.fit_transform(X_train)
                    X_test_pca=pca.transform(X_test)
                    from sklearn.linear_model import LogisticRegression
                    model=LogisticRegression(solver='liblinear')
                    model.fit(X_train_pca,y_train)
                    preds = model.predict(X_test_pca)
                    score = roc_auc_score(y_test, preds)
                    score_list.append(score)
                    if(score>high_score):
                        high_score = score
                        n_best = i
                    if i in check_point:
                        print(" Best n so far: {} \n Score: {} \n".format(n_best,high_score))
                print("Optimal n: {} \n Score: {} \n".format(n_best,high_score))
                pca = PCA(n_components=n_best, random_state=self.random_state).fit(self.X)
                self.X = pd.DataFrame(pca.transform(self.X), columns=['PCA %i' % i for i in range(n_best)], index=self.X.index)



    def match_features(self, to):
#       match object features to argument features
#       identify columns to drop and add
        match_columns = to.columns.to_list()
        drop_columns = [col for col in self.X.columns.to_list() if col not in match_columns]
        add_columns = [col for col in match_columns if col not in self.X.columns.to_list()]
#       first pass: remove extra features
        self.X = self.X.drop(columns = drop_columns)
#       second pass: add missing features as empty columns
        self.X = self.X.reindex(columns = self.X.columns.to_list() + add_columns)
