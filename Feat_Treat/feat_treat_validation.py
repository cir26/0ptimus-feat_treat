class validation:
    
    def tune_test(self,model,tuning_iter,sample=True,tuning_strategy='randomized',tuning_metric='roc_auc',test_size=0.2, skfold=8):
#       split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
#       test set will remain untouched until making predictions with tuned model
        col = X_train.columns
        col_length=len(X_train.columns)
#       instantiate cv stratified fold
        inner_kfold = StratifiedKFold(n_splits=skfold, shuffle=True, random_state=self.random_state)
        #outer_kfold = StratifiedKFold(n_splits=skfold, shuffle=True, random_state=self.random_state)
        if("multilayer_perceptron" in str(model)):
            #standardize data sets
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=col, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=col, index=X_test.index)
        else:
            pass

        param_grid = self.select_hyperparameter_grid(model=model,feature_set=X_train,random_state=self.random_state)
        samples = self.resampler(sample=sample,X_train=X_train,y_train=y_train)

#       tune and test
        N = len(samples)
#       check that tuning_iter and number of samples agree
        if N > tuning_iter:
            tuning_iter = N
        best_param=[0]*N
        self.metrics=pd.DataFrame(columns=["Sampling",
                                           "Accuracy",
                                           'Precision',
                                           'Recall',
                                           'Specificity',
                                           'Neg Pred Val',
                                           'Confusion Sum',
                                           'F1',
                                           'F2',
                                           'G1',
                                           'Cohen kappa',
#                                              "RMSE" : rmse,
#                                              "Jaccard score" : jaccard,
                                           "Brier score loss",
                                           'MCC',
                                           "AUC"])
#       copy uninstantiated model for replacement within loop
        model_rep=model
        print("Estimator: ",model_rep, " \n ")
        for i in range(0,N):
#           instantiate model for tuning purposes
            model_inst=model_rep()
            print("Sampling technique: ",samples[i][2], "\n ")
            tuning_strategy=''.join(tuning_strategy.split()).lower()
#           check for bayes or random search
            if tuning_strategy=='bayes':
                # unpack hyperparameters
                param_grid_cat = {k: v for k, v in param_grid.items() if isinstance(v,Categorical)==True or k=='n_jobs' or k=='random_state'}
                param_grid_num = {k:(min(v),max(v)) for k,v in param_grid.items() if k not in param_grid_cat}
                param_grid={**param_grid_num,**param_grid_cat}
                import warnings
                warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
                grid_search = BayesSearchCV(model_inst, param_grid, scoring=tuning_metric, n_jobs=-1, n_points=4, cv=inner_kfold, n_iter=floor(tuning_iter/N),verbose=0)
            elif tuning_strategy=='randomized':
                grid_search = RandomizedSearchCV(model_inst, param_grid, scoring=tuning_metric, n_jobs=-1, pre_dispatch='2*n_jobs', refit=True, cv=inner_kfold, n_iter=floor(tuning_iter/N),verbose=0)
            print('Tuning...')
            grid_results=grid_search.fit(samples[i][0],samples[i][1])
            best_param[i]=(samples[i][2], grid_results.best_estimator_.get_params())
            print("Best {}: {} using {} \n".format(tuning_metric, grid_results.best_score_, best_param[i][1]))
            print("Validating model...")
#           Train model
            #model = grid_search.best_estimator_
            del grid_results
            model=model_rep(**best_param[i][1])
            model.fit(samples[i][0],samples[i][1])
            probs = model.predict_proba(X_test[samples[i][0].columns])
            #return performance metrics
            df = self.performance_metrics(y_test=y_test,probs=probs, pred_threshold=0.5, sample_method=samples[i][2],index=i)
            self.metrics=self.metrics.append(df)
#           end of loop
        self.hyperparameters=best_param
        #self.metrics=self.metrics.fillna(0)
        radar_df=self.metrics[['Sampling',
                                'Accuracy',
                                'F1',
                                'G1',
                                'Cohen kappa',
                                'MCC',
                                'AUC']]
        print("Performance Metrics Summary")
        radar_plot = self.create_radar_chart(radar_df=radar_df)
        radar_plot.show()




    def multi_test_split_validation(self, model, params, iterations, sample = False, test_size = 0.2):
#       use to validate best hyperparameters by averaging results of multiple random test split iterations
#       initialize metrics dataframe
        metrics_log=pd.DataFrame(columns=["Sampling",
                                           "Accuracy",
                                           'Precision',
                                           'Recall',
                                           'Specificity',
                                           'Neg Pred Val',
                                           'Confusion Sum',
                                           'F1',
                                           'F2',
                                           'G1',
                                           'Cohen kappa',
#                                              "RMSE" : rmse,
#                                              "Jaccard score" : jaccard,
                                           "Brier score loss",
                                           'MCC',
                                           "AUC"])
        model_rep=model
        random = np.arange(self.random_state,self.random_state+iterations)
        random = [num*randint(1,self.random_state*iterations) for num in random]
        print('Random states: ',random)
        for j in range(0,iterations):
#           split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random[j])
            samples = self.resampler(sample=sample,X_train=X_train,y_train=y_train)
            N=len(samples)
            for i in range(0,N):
                if isinstance(params, list)==True:
                    fitted=0
                    for k in params:
                        if k[0] == samples[i][2]:
                            param = k[1]
                            param.update({'random_state':random[j]})
                            model=model_rep(**param)
                            model.fit(samples[i][0],samples[i][1])
                            probs = model.predict_proba(X_test[samples[i][0].columns])
                            df = self.performance_metrics(y_test=y_test,probs=probs, pred_threshold=0.5, sample_method=samples[i][2],index=(N*j)+i, verbose=False)
                            metrics_log=metrics_log.append(df)
                            fitted=fitted+1
                        else:
                            pass
                    if fitted==0:
                        param = params[0][1]
                        param.update({'random_state':random[j]})
                        model=model_rep(**param)
                        model.fit(samples[i][0],samples[i][1])
                        probs = model.predict_proba(X_test[samples[i][0].columns])
                        df = self.performance_metrics(y_test=y_test,probs=probs, pred_threshold=0.5, sample_method=samples[i][2],index=(N*j)+i, verbose=False)
                        metrics_log=metrics_log.append(df)
                    else:
                        pass
                else:
                    params.update({'random_state':random[j]})
                    model=model_rep(**params)
                    model.fit(samples[i][0],samples[i][1])
                    probs = model.predict_proba(X_test[samples[i][0].columns])
                    df = self.performance_metrics(y_test=y_test,probs=probs, pred_threshold=0.5, sample_method=samples[i][2],index=(N*j)+i, verbose=False)
                    metrics_log=metrics_log.append(df)
#       return dataframe of average column scores of each sampling method
        ave_metrics = metrics_log.groupby("Sampling").mean()
        ave_metrics.reset_index(level="Sampling", inplace=True)
        self.metrics = ave_metrics
        simple_ave_metrics = ave_metrics[['Sampling',
                                 'Accuracy',
                                 'F1',
                                 'G1',
                                 'Cohen kappa',
                                 'MCC',
                                 'AUC']]
        radar_plot = self.create_radar_chart(radar_df=simple_ave_metrics)
        radar_plot.show()
