# main dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import floor, ceil, pi
import copy
from random import randint, choice
from itertools import cycle
# scikit tools
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, precision_recall_curve,auc,cohen_kappa_score,accuracy_score,roc_auc_score,roc_curve,brier_score_loss,confusion_matrix,f1_score,recall_score,precision_score,matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# sampling tools
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
# timer tool
from timeit import default_timer as timer


class static:

    @staticmethod
    def performance_metrics_binary(y_test, probs, pred_threshold=0.5, sample_method_label='None', index=0, verbose=True):
#       returns dataframe of various performance metrics
        logloss = log_loss(y_test, probs)
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
        if verbose == True:
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
            print("\n ")
        else:
            pass
        conf_mat = confusion_matrix(y_true=y_test, y_pred=preds)
        specificity=round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1]),5)
        neg_pred= round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0]),5)
        g1=round(2*((specificity*neg_pred)/(specificity+neg_pred)),5)
        accuracy = round(accuracy_score(y_test, preds),5)
        ck = round(cohen_kappa_score(y_test,preds),5)
        mcc = round(matthews_corrcoef(y_test, preds),5)
        f1 = round(f1_score(y_test, preds),5)
        recall = round(recall_score(y_test, preds),5)
        precision = round(precision_score(y_test, preds),5)
        f2=round(5*((precision*recall)/((4*precision)+recall)),5)
        conf_sum =round(precision+recall+specificity+neg_pred,5)
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
            print(' ')
            print("Log loss: ", logloss)
            print('MCC:               ', mcc)
            print("AUC:               ", auc_score, "\n")
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
                      "Log loss" : logloss,
                      'MCC' : mcc,
                      "AUC" : auc_score},index=[index])
        df=df.fillna(0)
        return df

    @staticmethod
    def performance_metrics_multiclass(y_test, probs, preds, classes, weights, sample_method_label='None', index=0, verbose=True):
        num_classes=len(classes)
        print(classes)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probs[:, i])
            roc_auc[i] = round(auc(fpr[i], tpr[i]),3)
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probs.ravel())
        roc_auc["micro"] = round(auc(fpr["micro"], tpr["micro"]),3)
        #auc_score = round(auc(fpr["micro"], tpr["micro"]),5)
        if verbose == True:
            #plt.figure()
            plt.figure(figsize=(8,8))
            lw = 2
            plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                     lw=lw, label='Micro-average (AUC = {})'.format(roc_auc["micro"]))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            colors = cycle(['xkcd:sun yellow','aqua', 'darkorange', 'cornflowerblue', 'red','green'])
            for i, color in zip(range(num_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                         label='Class {} (AUC = {})'.format(classes[i], roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.show()
        else:
            pass
        # return actual final predictions (shape(:,1))
        # final_preds=[]
        # for i in range(0,len(y_test)):
        #     decision = np.where(probs[i] == np.amax(probs[i]))
        #     if len(decision[0])>1:
        #         decision=choice(decision[0])
        #     else:
        #         decision=decision[0][0]
        #     final_preds.append(classes[decision])
        specificity_multi = dict()
        neg_pred_multi = dict()
        g1_multi = dict()
        accuracy_multi = dict()
        ck_multi = dict()
        mcc_multi = dict()
        f1_multi = dict()
        recall_multi = dict()
        precision_multi = dict()
        f2_multi = dict()
        conf_sum_multi = dict()
        auc_score_multi = dict()
        logloss_multi = dict()
#       calculate metrics per class
        for i in range(num_classes):
            logloss_multi[i] = log_loss(y_test[:,i], probs[:,i])
            conf_mat = confusion_matrix(y_true=y_test[:,i], y_pred=preds[:,i])
            specificity_multi[i]=round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[0][1]),5)
            neg_pred_multi[i]= round(conf_mat[0][0] / (conf_mat[0][0]+conf_mat[1][0]),5)
            g1_multi[i]=round(2*((specificity_multi[i]*neg_pred_multi[i])/(specificity_multi[i]+neg_pred_multi[i])),5)
            accuracy_multi[i] = round(accuracy_score(y_test[:,i], preds[:,i]),5)
            ck_multi[i] = round(cohen_kappa_score(y_test[:,i],preds[:,i]),5)
            mcc_multi[i] = round(matthews_corrcoef(y_test[:,i], preds[:,i]),5)
            f1_multi[i] = round(f1_score(y_test[:,i], preds[:,i]),5)
            recall_multi[i] = round(recall_score(y_test[:,i], preds[:,i]),5)
            precision_multi[i] = round(precision_score(y_test[:,i], preds[:,i]),5)
            f2_multi[i]=round(5*((precision_multi[i]*recall_multi[i])/((4*precision_multi[i])+recall_multi[i])),5)
            conf_sum_multi[i] =round(precision_multi[i]+recall_multi[i]+specificity_multi[i]+neg_pred_multi[i],5)
            auc_score_multi[i] = round(roc_auc_score(y_test[:,i], preds[:,i]),5)

        Raw_accuracy= sum(accuracy_multi.values())
#       return sum of metrics weighted by class size
        accuracy = [x*w for x,w in zip(accuracy_multi.values(),weights)]
        accuracy= sum(accuracy)
        precision = [x*w for x,w in zip(precision_multi.values(),weights)]
        precision = sum(precision)
        recall = [x*w for x,w in zip(recall_multi.values(),weights)]
        recall= sum(recall)
        specificity = [x*w for x,w in zip(specificity_multi.values(),weights)]
        specificity= sum(specificity)
        neg_pred = [x*w for x,w in zip(neg_pred_multi.values(),weights)]
        neg_pred= sum(neg_pred)
        conf_sum = [x*w for x,w in zip(conf_sum_multi.values(),weights)]
        conf_sum= sum(conf_sum)
        f1 = [x*w for x,w in zip(f1_multi.values(),weights)]
        f1= sum(f1)
        g1 = [x*w for x,w in zip(g1_multi.values(),weights)]
        g1= sum(g1)
        f2 = [x*w for x,w in zip(f2_multi.values(),weights)]
        f2= sum(f2)
        ck = [x*w for x,w in zip(ck_multi.values(),weights)]
        ck= sum(ck)
        mcc = [x*w for x,w in zip(mcc_multi.values(),weights)]
        mcc= sum(mcc)
        auc_score = [x*w for x,w in zip(auc_score_multi.values(),weights)]
        auc_score= sum(auc_score)
        logloss = [x*w for x,w in zip(logloss_multi.values(),weights)]
        logloss= sum(logloss)
        if verbose==True:
            print("Raw Accuracy:      ", Raw_accuracy)
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
            print(' ')
            print("Log loss:          ", logloss)
            print('MCC:               ', mcc)
            print("AUC:               ", auc_score, "\n")
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
                    "Log loss" : logloss,
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
            max_depth = np.arange(1,col_length,1)
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
        elif("MLP" in str(model)):
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
            low_score_row=[radar_df['Log loss'].idxmax()]
            radar_df = radar_df.drop(index=low_score_row)
            radar_df = radar_df.reset_index(drop=True)
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
                    smote = SMOTE(sampling_strategy='auto',n_jobs=-1)
                    X_sm, y_sm = smote.fit_sample(X_train, y_train)
                    X_sm = pd.DataFrame(X_sm, columns = col)
                    samples.append(tuple([X_sm,y_sm,"SMOTE"]))

                elif i in ['smotenc']:
                    cat_index = [X_train.columns.get_loc(i) for i in encoded_columns]
                    smotenc = SMOTENC(categorical_features=cat_index, sampling_strategy='auto',n_jobs=-1)
                    X_smnc, y_smnc = smotenc.fit_sample(X_train, y_train)
                    X_smnc = pd.DataFrame(X_smnc, columns = col)
                    samples.append(tuple([X_smnc,y_smnc,"SMOTENC"]))

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
                    samples.append(tuple([X_oss,y_oss,"One-Sided Selection"]))

        return samples
