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


class static:


    @staticmethod
    def performance_metrics(y_test, probs, pred_threshold=0.5, sample_method_label='None', index=0, verbose=True):
#       returns dataframe of various performance metrics
        probs1 = probs[:,1]
        preds=[]
        for prediction in probs1:
            if prediction >= pred_threshold:
                preds.append(1)
            else:
                preds.append(0)
        fpr, tpr, threshold = roc_curve(y_test, probs1)
        auc_score = round(auc(fpr,tpr),5)
        prec, rec, threshold_pr = precision_recall_curve(y_test, probs1)
        auc_pr_curve = round(auc(rec, prec),5)
        conf_mat = confusion_matrix(y_true=y_test, y_pred=preds)
        accuracy = round(accuracy_score(y_test, preds),5)
        ck = round(cohen_kappa_score(y_test,preds),5)
        brier = round(brier_score_loss(y_test,probs1) ,5)
        f1 = round(f1_score(y_test, preds),5)
    #               jaccard = jaccard_score(y_test, preds)
        recall = round(recall_score(y_test, preds),5)
        precision = round(precision_score(y_test, preds),5)
        mcc = round(matthews_corrcoef(y_test, preds),5)
        specificity=round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1]),5)
        neg_pred= round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0]),5)
        f2=round(5*((precision*recall)/((4*precision)+recall)),5)
        g1=round(2*((specificity*neg_pred)/(specificity+neg_pred)),5)
        conf_sum =round(precision+recall+specificity+neg_pred,5)
    #               rmse = np.sqrt(mean_squared_error(y_test, preds))
        if verbose==True:
            print("Accuracy:          ", accuracy)
            print('Precision:         ', precision)
            print('Recall:            ', recall)
            print('Specificity:       ', specificity)
            print('Neg Pred Val:      ', neg_pred)
            print('Confusion Sum:     ', conf_sum)
            print(' ')
            print('F1 score:          ', f1)
            print('F2 score:          ', f2)
            print('G1 score:          ', g1)
            print('Cohen kappa score: ', ck)
        #               print("RMSE:              ", rmse)
            print(' ')
        #               print("Jaccard score:     ", jaccard)
            print("Brier score loss:  ", brier)
            print('MCC:               ', mcc)
            print("AUC:               ", auc_score, "\n")
        #-------- ROC CURVE --------------
            plt.figure()
            lw=2
            plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = {})'.format(auc_score))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(' Sampling: {} \n Receiver Operating Characteristic'.format(sample_method_label))
            plt.legend(loc="lower right")
            plt.show()
        #-----------------------------------
        #-------- Precision-Recall CURVE --------------
            unique_elements, counts_elements = np.unique(y_test, return_counts=True)
            no_skill=counts_elements[1]/sum(counts_elements)
            plt.figure()
            plt.plot(rec, prec, color='darkred',lw=lw, label='Precision-Recall curve (area = {})'.format(auc_pr_curve))
            plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(' Sampling: {} \n Precision-Recall Curve'.format(sample_method_label))
            plt.legend(loc="lower right")
            plt.show()
            print("\n ","\n ","\n ")
        else:
            pass
#       update self.metrics data
        df=pd.DataFrame({"Sampling" : sample_method_label,
                    "Accuracy" : accuracy,
                    'Precision' : precision,
                    'Recall' : recall,
                    'Specificity' : specificity,
                    'Neg Pred Val' : neg_pred,
                    'Confusion Sum' : conf_sum,
                    'F1' : f1,
                    'F2' : f2,
                    'G1' : g1,
                    'Cohen kappa' : ck,
#                   "RMSE" : rmse,
#                   "Jaccard score" : jaccard,
                    "Brier score loss" : brier,
                    'MCC' : mcc,
                    "AUC" : auc_score},index=[index])
        df=df.fillna(0)
        return df

    @staticmethod
    def select_hyperparameter_grid(model,feature_set, random_state=42):
        col_length=len(feature_set.columns)
#       IMPORTANT when inputting default hyperparameters:
#       wrap non-int or non-float types with Categorical() function
#       NEVER input int or float types with only one parameter option... this will not work with bayes search (works fine with randomized search)
        if("RandomForest" in str(model)):
#           default hyperparameter testing range
            bootstrap = Categorical([True, False])
            n_estimators = [300,500,700]
            criterion= Categorical(['gini','entropy'])
            max_depth =np.arange(1,floor(col_length),1)
            max_features = np.random.uniform(0.01,1,10000)
            min_samples_split = np.random.uniform(0.01,1,10000)
            min_samples_leaf = np.random.uniform(0.0001,0.5,10000)
            class_weight = Categorical(['balanced','balanced_subsample',None])
            n_jobs = [-1]
            random_state = [random_state]
#           input hyperparameters into dictionary
            param_grid = dict(n_estimators=n_estimators,
                              bootstrap=bootstrap,
                              criterion=criterion,
                              min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,
                              max_features=max_features,
                              max_depth=max_depth,
                              class_weight=class_weight,
                              n_jobs=n_jobs,
                              random_state=random_state)

        elif("XGB" in str(model)):
#           default hyperparameter testing range
            booster = Categorical(['gbtree','gblinear','dart'])
            n_estimators = [300,500,700]
            learning_rate = np.random.uniform(0.000001,1,10000)
            max_depth = np.arange(1,floor(col_length),1)
            gamma = np.random.uniform(0,15,10000)
            reg_alpha = np.random.uniform(0,1,10000)
            reg_lambda = np.random.uniform(0,1,10000)
            objective = Categorical(['reg:logistic'])
            #subsample = np.random.beta(2,5,10000)
            #colsample_bytree = np.random.beta(2,5,10000)
            scale_pos_weight = np.random.uniform(0,20,10000)
            min_child_weight = np.random.uniform(0,floor(0.5*col_length),10000)
            max_delta_step = floor(0.2*col_length)*np.random.beta(2,5,10000)
            n_jobs = [-1]
            random_state = [random_state]
#           input hyperparameters into dictionary
            param_grid = dict(n_estimators=n_estimators,
                              booster=booster,
                              learning_rate=learning_rate,
                              max_depth=max_depth,
                              gamma=gamma,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              objective=objective,
                              #subsample=subsample,
                              #colsample_bytree=colsample_bytree,
                              scale_pos_weight=scale_pos_weight,
                              min_child_weight=min_child_weight,
                              max_delta_step=max_delta_step,
                              n_jobs=n_jobs,
                              random_state=random_state)

        elif("LogisticRegression" in str(model)):
#           default hyperparameter testing range
            penalty = Categorical(['l2'])
            tol = np.random.uniform(0.00000001,0.001,10000)
            C = np.random.uniform(0.000001,100,10000)
            fit_intercept = Categorical([True,False])
            intercept_scaling = np.random.uniform(0.01,10,10000)
            class_weight = Categorical(['balanced',None])
            solver = Categorical(['newton-cg', 'lbfgs','sag'])
            max_iter = np.arange(100,1000,10)
            n_jobs = [-1]
            random_state = [random_state]
#           input hyperparameters into dictionary
            param_grid = dict(penalty=penalty,
                              tol=tol,
                              C=C,
                              fit_intercept=fit_intercept,
                              intercept_scaling=intercept_scaling,
                              class_weight=class_weight,
                              solver=solver,
                              max_iter=max_iter,
                              n_jobs=n_jobs,
                              random_state=random_state)
        elif("SVC" in str(model)):
#           default hyperparameter testing range
            C=np.random.gamma(1.4,9,10000)
            kernel=Categorical(['rbf','poly','sigmoid'])
            degree=np.arange(1,20,1)
            gamma=np.random.uniform(0,25,10000)
            coef0=np.random.gamma(0.1,1,10000)
            shrinking=Categorical([True,False])
            probability=Categorical([True])
            tol=np.random.uniform(0.00000001,0.001,10000)
            class_weight=Categorical(['balanced',None])
            random_state=[random_state]
#           input hyperparameters into dictionary
            param_grid = dict(C=C,
                              kernel=kernel,
                              degree=degree,
                              gamma=gamma,
                              coef0=coef0,
                              shrinking=shrinking,
                              probability=probability,
                              tol=tol,
                              class_weight=class_weight,
                              random_state=random_state)

        #elif("Bagging" in str(model_rep)):
        elif("multilayer_perceptron" in str(model)):
#           default hyperparameter testing range
            neurons = [i for i in range(floor(col_length*0.05),floor(col_length*0.9),3)]
            hidden_layer_sizes = [(i,)*j for i in neurons for j in range(1,4)]
            activation = Categorical(['relu'])
            solver = Categorical(['sgd','adam'])
            tol = np.random.uniform(0.00000001,0.001,1000)
            alpha = 10.0 ** -np.random.uniform(1, 11,10000)
            learning_rate = Categorical(['constant', 'invscaling', 'adaptive'])
            learning_rate_init = 10.0 ** -np.random.uniform(2, 11,10000)
            momentum = np.random.uniform(0.01,1,10000)
            nesterovs_momentum = Categorical([True, False])
            power_t = np.random.uniform(0.01,2,10000)
            max_iter = np.arange(500,1000,100)
            random_state = [random_state]
            beta_1 = np.random.uniform(0,1,10000)
            beta_2 = np.random.uniform(0,1,10000)
#           input hyperparameters into dictionary
            param_grid = dict(hidden_layer_sizes=hidden_layer_sizes,
                              activation=activation,
                              solver=solver,
                              tol=tol,
                              alpha=alpha,
                              learning_rate=learning_rate,
                              learning_rate_init=learning_rate_init,
                              nesterovs_momentum=nesterovs_momentum,
                              momentum=momentum,
                              power_t=power_t,
                              max_iter=max_iter,
                              beta_1=beta_1,
                              beta_2=beta_2,
                              random_state=random_state)
        return param_grid

    @staticmethod
    def create_radar_chart(radar_df):
        # max display 4 best plots
        # more plots than this becomes confusing
        while (radar_df.shape[0])>4:
            # drop row with lowest AUC score for display purposes
            low_score_row=[radar_df['AUC'].idxmin()]
            radar_df = radar_df.drop(index=low_score_row)
            radar_df.reset_index(drop=True,inplace=True)
        # ------- RADAR CHARTS PART 1: Create background
        # number of variables
        categories=list(radar_df)[1:]
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(111, polar=True)
        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="blue", size=7)
        plt.ylim(0,1)
        # ------- PART 2: Add plots
        # Plot each individual = each line of the data
        labels=[i for i in radar_df['Sampling']]
        colors=['blue', 'orange', 'green', 'red']
        colors=colors[:len(labels)]
        for i in range(0,len(labels)):
            values=radar_df.loc[i].drop('Sampling').values.flatten().tolist()
            values += values[:1]
            values = [abs(number) for number in values]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=labels[i])
            ax.fill(angles, values, colors[i], alpha=0.1)
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        return plt

    @staticmethod
    def resampler(sample, X_train, y_train, encoded_columns):
        col=X_train.columns
        if sample==False:
            samples = [(X_train,y_train,"None")]
        else:
            if sample==True:
                sample=['rus','ros','smote','smotenc','smote+tl','tl','smote+enn', 'neighborhoodcleaning','onesidedselection']
            else:
                pass
            samples = [(X_train,y_train,"None")]
            for i in sample:
                i=''.join(i.split()).lower()
                if i in ['rus','randomundersampling','randomundersample']:
                    rus = RandomUnderSampler(sampling_strategy='auto',n_jobs=-1)
                    X_rus, y_rus = rus.fit_sample(X_train, y_train)
                    X_rus = pd.DataFrame(X_rus, columns = col)
                    samples.append(tuple([X_rus,y_rus,"Random under sampling"]))

                elif i in ['ros','randomoversampling','randomoversample']:
                    ros = RandomOverSampler(sampling_strategy='auto',n_jobs=-1)
                    X_ros, y_ros = ros.fit_sample(X_train, y_train)
                    X_ros = pd.DataFrame(X_ros, columns = col)
                    samples.append(tuple([X_ros,y_ros,"Random over sampling"]))

                elif i in ['smote']:
                    smote = SMOTE(sampling_strategy='minority',n_jobs=-1)
                    X_sm, y_sm = smote.fit_sample(X_train, y_train)
                    X_sm = pd.DataFrame(X_sm, columns = col)
                    samples.append(tuple([X_sm,y_sm,"SMOTE"]))

                elif i in ['smotenc']:
                    cat_index = [X_train.columns.get_loc(i) for i in encoded_columns]
                    smotenc = SMOTENC(categorical_features=cat_index, sampling_strategy='auto',n_jobs=-1)
                    X_smnc, y_smnc = smotenc.fit_sample(X_train, y_train)
                    X_smnc = pd.DataFrame(X_smnc, columns = col)
                    samples.append(tuple([X_smnc,y_smnc,"SMOTE"]))

                elif i in ['smotetl','smote+tl','tlsmote','tl+smote']:
                    smt = SMOTETomek(sampling_strategy='auto',n_jobs=-1)
                    X_smt, y_smt = smt.fit_sample(X_train, y_train)
                    X_smt = pd.DataFrame(X_smt, columns = col)
                    samples.append(tuple([X_smt,y_smt,"SMOTE + TL"]))

                elif i in ['smoteenn','smotenn','ennsmote','smote+enn','enn+smote']:
                    sme = SMOTEENN(sampling_strategy='auto',n_jobs=-1)
                    X_sme, y_sme = sme.fit_sample(X_train, y_train)
                    X_sme = pd.DataFrame(X_sme, columns = col)
                    samples.append(tuple([X_sme,y_sme,"SMOTE + ENN"]))

                elif i in ['tl','tomek','tomeklink','tomeklinks']:
                    tl=TomekLinks(sampling_strategy='all',n_jobs=-1)
                    X_tl, y_tl = tl.fit_sample(X_train,y_train)
                    X_tl = pd.DataFrame(X_tl, columns = col)
                    samples.append(tuple([X_tl,y_tl,"Tomek link"]))

                elif i in ['neighborhoodcleaning', 'neighborhoodcleaningrule','neighbourhoodcleaning', 'neighbourhoodcleaningrule', 'ncr', 'nc', 'ncl']:
                    ncl=NeighbourhoodCleaningRule(sampling_strategy='auto',n_jobs=-1)
                    X_ncl, y_ncl = ncl.fit_sample(X_train,y_train)
                    X_ncl = pd.DataFrame(X_ncl, columns = col)
                    samples.append(tuple([X_ncl,y_ncl,"Neighborhood Cleaning"]))

                elif i in ['onesidedselection', 'oss', 'one-sidedselection']:
                    oss=OneSidedSelection(sampling_strategy='auto',n_jobs=-1)
                    X_oss, y_oss = oss.fit_sample(X_train,y_train)
                    X_oss = pd.DataFrame(X_oss, columns = col)
                    samples.append(tuple([X_oss,y_oss,"Neighborhood Cleaning"]))

        return samples
