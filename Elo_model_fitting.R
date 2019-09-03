library(dplyr)
library(xgboost)
library(Matrix)
library(caret)
library(glmnet)
library(ranger)
library(doParallel)
library(foreach)
library(imputeMissings)

cores <- 1
threads <- if(Sys.getenv('PBS_NUM_PPN')!=''){cores <- as.numeric(Sys.getenv('PBS_NUM_PPN'))-1} else {cores <- detectCores()-1}
cl <- makeForkCluster(cores)

load("modelfit.RData")

#three data sets
# train set #1
fulltrain_upd<- fulltrain_upd %>% left_join(hist2upd, by="card_id")
fulltrain_upd<- fulltrain_upd[!is.infinite(rowSums(fulltrain_upd[,-c(1,2,7,10)])),]

fulltrain_upd$feature_1<-as.factor(fulltrain_upd$feature_1)
fulltrain_upd$feature_2<-as.factor(fulltrain_upd$feature_2)
fulltrain_upd$feature_3<-as.factor(fulltrain_upd$feature_3)
#fulltrain_upd$rectday<-as.factor(fulltrain_upd$rectday)
#fulltrain_upd$rectmonth<-as.factor(fulltrain_upd$rectmonth)
#fulltrain_upd$rectyear<-as.factor(fulltrain_upd$rectyear)
#fulltrain_upd$year<-as.factor(fulltrain_upd$year)
#fulltrain_upd$month<-as.factor(fulltrain_upd$month)

#train set #2

rm_feat[14,3]="c2_2"
rm_feat[15,3]="c2_4"

fulltrain_upd2<-fulltrain_upd[,-which(colnames(fulltrain_upd)==rm_feat$X3[1])]

for (j in 2:nrow(rm_feat)){
  
  fulltrain_upd2<-fulltrain_upd2[,-which(colnames(fulltrain_upd2)==rm_feat$X3[j])]
  
}

#train set #3

fulltrain_upd3<-fulltrain_upd2[,-which(colnames(fulltrain_upd2)==final_rm[1])]

for (j in 2:length(final_rm)){
  
  fulltrain_upd3<-fulltrain_upd3[,-which(colnames(fulltrain_upd3)==final_rm[j])]
  
}

#test set
fulltest<-fulltest %>% left_join(hist2upd, by="card_id")

fulltest$feature_1<-as.factor(fulltest$feature_1)
fulltest$feature_2<-as.factor(fulltest$feature_2)
fulltest$feature_3<-as.factor(fulltest$feature_3)
fulltest$rectday<-as.integer(fulltest$rectday)
fulltest$rectmonth<-as.integer(fulltest$rectmonth)
fulltest$rectyear<-as.integer(fulltest$rectyear)
fulltest$year<-as.integer(fulltest$year)
fulltest$month<-as.integer(fulltest$month)

#impute missing values in fulltest set
fulltest<-impute(fulltest,object=NULL, method="median/mode",flag=FALSE)

#fulltest_2

fulltest_2<-fulltest[,-which(colnames(fulltest)==rm_feat$X3[1])]

for (j in 2:nrow(rm_feat)){
  
  fulltest_2<-fulltest_2[,-which(colnames(fulltest_2)==rm_feat$X3[j])]
  
}

#fulltest_3

fulltest_3<-fulltest_2[,-which(colnames(fulltest_2)==final_rm[1])]

for (j in 2:length(final_rm)){
  
  fulltest_3<-fulltest_3[,-which(colnames(fulltest_3)==final_rm[j])]
  
}


##################
#cross validation
##################

datasplit<- list()

set.seed(100)
dp1<-createDataPartition(fulltrain_upd$target,times=5, p=0.8,list=TRUE)
set.seed(200)
dp2<-createDataPartition(fulltrain_upd$target,times=5, p=0.8,list=TRUE)
set.seed(300)
dp3<-createDataPartition(fulltrain_upd$target,times=5,p=0.8,list=TRUE)

datasplit[["dp_0.8"]]<-cbind(dp1,dp2,dp3)


# 5-folds cv
set.seed(100)
cv_5f_1<-createFolds(fulltrain_upd$target,k=5,list = TRUE, returnTrain=TRUE)
set.seed(200)
cv_5f_2<-createFolds(fulltrain_upd$target,k=5,list = TRUE, returnTrain=TRUE)
set.seed(300)
cv_5f_3<-createFolds(fulltrain_upd$target,k=5,list = TRUE, returnTrain=TRUE)

datasplit[["5k"]]<-cbind(cv_5f_1,cv_5f_2,cv_5f_3)


# 10 folds cv
set.seed(100)
cv_10f_1<-createFolds(fulltrain_upd$target,k=10,list=TRUE,returnTrain=TRUE)
set.seed(200)
cv_10f_2<-createFolds(fulltrain_upd$target,k=10,list=TRUE,returnTrain=TRUE)
set.seed(300)
cv_10f_3<-createFolds(fulltrain_upd$target,k=10,list=TRUE,returnTrain=TRUE)

datasplit[["10k"]]<-cbind(cv_10f_1,cv_10f_2,cv_10f_3)


###########################################
#Model Fitting without considering outliers
###########################################


#GLM

fit_glm=function(train,test,index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  tc<-trainControl(method="cv", number=10,verboseIter = FALSE,summaryFunction=defaultSummary,selectionFunction="best")
  fm_glm<-as.formula(paste("target","~",paste(names(subtrain)[!names(subtrain) %in% c("target","rectday","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  
  lambda=seq(0.001,0.01,length=10)
  alpha=seq(0.3,0.7,length=10)
  glm_grid=expand.grid(lambda,alpha)
  colnames(glm_grid)=c("lambda","alpha")
  
  set.seed(100)
  m_glm<-train(fm_glm,data=subtrain,
               method='glmnet',
               metric="RMSE",
               trControl=tc,
               tuneGrid=glm_grid,
               na.action=na.pass)
  
  glm_tr_te_pred<-predict(m_glm, subtest, na.action=na.pass)
  glm_test_pred<-predict(m_glm, test, na.action=na.pass)
  
  truedata<-subtest[,6]
  err_glm<-sqrt(mean((truedata-glm_tr_te_pred)^2))
  
  return(list(fit=m_glm, tr_te_err=err_glm, test_pred=glm_test_pred))
  
}



#Ridge

fit_ridge=function (train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  tc<-trainControl(method="cv", number=10,verboseIter = FALSE,summaryFunction=defaultSummary,selectionFunction="best")
  fm_ridge<-as.formula(paste("target","~",paste(names(subtrain)[!names(subtrain) %in% c("target","rectday","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  ridge_grid=data.frame(.lambda=seq(0,.2,length=20),.alpha=0)
  set.seed(100)
  m_ridge<-train(fm_ridge,data=subtrain,
                 method='glmnet',
                 metric="RMSE",
                 trControl=tc,
                 tuneGrid=ridge_grid,
                 preProcess=c("center","scale"),
                 na.action=na.pass)
  
  ridge_tr_te_pred<-predict(m_ridge, subtest, na.action=na.pass)
  ridge_test_pred<-predict(m_ridge, test, na.action=na.pass)
  
  truedata<-subtest[,6]
  err_ridge<-sqrt(mean((truedata-ridge_tr_te_pred)^2))
  
  return(list(fit=m_ridge,tr_te_err=err_ridge, test_pred=ridge_test_pred))
  
}


#Random Forest

fit_rf<-function(train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  tc<-trainControl(method="cv", number=10,summaryFunction=defaultSummary,selectionFunction="best")
  fm_rf<-as.formula(paste("target","~",paste(names(subtrain)[!names(subtrain) %in% c("target","rectday","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  
  #rf_grid<- data.frame("mtry"=seq(60,100,5),"min.node.size"= 5, "splitrule"="variance")
  mtry=seq(20,60,10)
  splitrule="variance"
  min.node.size=5
  rf_grid=expand.grid(mtry,splitrule,min.node.size)
  colnames(rf_grid)=c("mtry","splitrule","min.node.size")
  
  #fit in clusters with parallel
  set.seed(100)
  m_rf<-train(fm_rf,subtrain,
              method='ranger',
              metric="RMSE",
              trControl=tc,
              tuneGrid=rf_grid,
              num.trees=500,
              na.action=na.pass, num.threads=threads)
  
  #make prediction
  rf_tr_te_pred<-predict(m_rf, subtest,na.action=na.pass)
  rf_test_pred<-predict(m_rf, test, na.action=na.pass)
  
  truedata<-subtest[,6]
  err_rf<-sqrt(mean((truedata-rf_tr_te_pred)^2))
  
  return(list(fit=m_rf,tr_te_err=err_rf, test_pred=rf_test_pred))
}


#XGBOOST

fit_xgb<- function(train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  formu<-as.formula(paste("target","~-1+",paste(names(subtrain)[!names(subtrain) %in% c("target","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  
  tr_data<- sparse.model.matrix(object=formu,drop.unused.levels = FALSE,data = subtrain)
  tr_ts_data<- sparse.model.matrix(object=formu,drop.unused.levels = FALSE,data = subtest)
  
  formu_ts<-as.formula(paste("~-1+",paste(names(test)[!names(test) %in% c("first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  ts_data<-sparse.model.matrix(object=formu_ts,drop.unused.levels = FALSE,data = test)
  
  tr_label=subtrain[complete.cases(subtrain),6]
  tr_ts_label=subtest[complete.cases(subtest),6]
  
  tr_ma<-xgb.DMatrix(data=as.matrix(tr_data),label=tr_label,missing = c(NA,NaN,Inf))
  tr_ts_ma<-xgb.DMatrix(data=as.matrix(tr_ts_data),label=tr_ts_label,missing = c(NA,NaN,Inf))
  ts_ma<-xgb.DMatrix(data=as.matrix(ts_data),missing = c(NA,NaN,Inf))
  
  
  # grid search on hyperparameters of xgb model with 5 folds cv
  
  booster = "gbtree"
  objective = "reg:linear"
  eval_metric = "rmse"
  eta=0.01 
  gamma=c(0,1)
  alpha=0
  lambda=c(0,5)
  max_depth=c(5,15,25) 
  min_child_weight=c(5,10) 
  subsample=0.8 
  colsample_bytree = 0.7
  colsample_bylevel = 0.6
  
  param_grid <-expand.grid(booster,objective,eval_metric,eta,gamma,
                           alpha,lambda,max_depth, min_child_weight,subsample, colsample_bytree,colsample_bylevel)
  colnames(param_grid) <- c("booster","objective","eval_metric","eta",
                            "gamma","alpha","lambda","max_depth","min_child_weight","subsample","colsample_bytree",
                            "colsample_bylevel")
  
  param_grid$booster<-as.character(param_grid$booster)
  param_grid$objective<-as.character(param_grid$objective)
  param_grid$eval_metric<-as.character(param_grid$eval_metric)
  
  grid_cv_error<-matrix(0,nrow=nrow(param_grid),ncol=2)
  
  
  foreach (k = 1:nrow(param_grid)) %dopar% {
    
    xgbcv_grid <- xgb.cv(params = as.list(param_grid[k,]), data = tr_ma, nrounds = 1000, nfold = 5, showsd = T, stratified = T, early_stop_round = 100, maximize = F,verbose=0)
    
    grid_cv_error[k,1]<- min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    grid_cv_error[k,2]<- which.min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    
  }
  
  #compare rmse of each model and select the best parameter combination
  
  params=as.list(param_grid[which.min(grid_cv_error),])
  
  m_xgb <- xgb.train(params=params, data=tr_ma, nrounds=1000, watchlist=list(val=tr_ts_ma,train=tr_ma),
                     print_every_n = 100, early_stopping_rounds = 100)
  
  xgb_tr_te_pred<-predict(m_xgb, tr_ts_ma, na.action=na.pass)
  xgb_test_pred<-predict(m_xgb, ts_ma, na.action=na.pass)
  
  truedata<-subtest[,6]
  err_xgb<-sqrt(mean((truedata-xgb_tr_te_pred)^2))
  
  return(list(fit=m_xgb,tr_te_err=err_xgb, test_pred=xgb_test_pred
  ))
  
  
}


##################################
#Model Fitting considering outliers
##################################


#GLM

fit_glm_ens=function(train,test,index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  #separate outlier targets and non outlier targets
  
  ds_out<-subtrain[which(subtrain$target>=quantile(subtrain$target,probs=1) |subtrain$target<=quantile(subtrain$target, probs=0)),]
  ds_nout<-subtrain[-which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  
  
  #build outlier model
  tc<-trainControl(method="cv", number=10,verboseIter = FALSE,summaryFunction=defaultSummary,selectionFunction="best")
  fm_glm_out<-as.formula(paste("target","~",paste(names(ds_out)[!names(ds_out) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  
  lambda=seq(0.001,0.01,length=10)
  alpha=seq(0.3,0.7,length=10)
  glm_grid=expand.grid(lambda,alpha)
  colnames(glm_grid)=c("lambda","alpha")
  
  set.seed(100)
  m_glm_out<-train(fm_glm_out,data=ds_out,
                   method='glmnet',
                   metric="RMSE",
                   trControl=tc,
                   tuneGrid=glm_grid,
                   na.action=na.pass)
  
  #build non outlier model
  fm_glm_nout<-as.formula(paste("target","~",paste(names(ds_nout)[!names(ds_nout) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  set.seed(100)
  m_glm_nout<-train(fm_glm_nout,data=ds_nout,
                    method='glmnet',
                    metric="RMSE",
                    trControl=tc,
                    tuneGrid=glm_grid,
                    na.action=na.pass)
  
  
  #make separate predictions with each model
  glm_tr_te_prd_out<-predict(m_glm_out, subtest, na.action=na.pass)
  glm_tr_te_prd_nout<-predict(m_glm_nout, subtest, na.action=na.pass)
  
  #ensumble predictions by adjusting weights
  
  rmse={}
  ens_weight={}
  
  for (w in seq(0,1, 0.1) ){
    
    ens_weight=rbind(ens_weight,w)
    glm_tr_te_prd= w*glm_tr_te_prd_out+(1-w)*glm_tr_te_prd_nout
    rmse_trte=sqrt(mean((glm_tr_te_prd-subtest$target)^2))
    rmse=rbind(rmse,rmse_trte)
  }
  
  best_w=cbind(rmse,ens_weight)[which.min(cbind(rmse,ens_weight)[,1]),2]
  
  glm_tr_te_pred= best_w*glm_tr_te_prd_out+(1-best_w)*glm_tr_te_prd_nout
  
  #predict test set
  glm_test_pred_out<-predict(m_glm_out, test, na.action=na.pass)
  glm_test_pred_nout<-predict(m_glm_nout, test, na.action=na.pass)
  
  glm_test_pred= best_w*glm_test_pred_out+(1-best_w)*glm_test_pred_nout
  
  truedata<-subtest[,6]
  err_glm<-sqrt(mean((truedata-glm_tr_te_pred)^2))
  
  return(list(best_w=best_w, tr_te_err=err_glm, test_pred=glm_test_pred))
  
}



#Ridge

fit_ridge_ens=function (train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  #separate outlier targets and non outlier targets
  
  ds_out<-subtrain[which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  ds_nout<-subtrain[-which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  
  
  #build outlier model
  tc<-trainControl(method="cv", number=10,verboseIter = FALSE,summaryFunction=defaultSummary,selectionFunction="best")
  fm_ridge_out<-as.formula(paste("target","~",paste(names(ds_out)[!names(ds_out) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  
  ridge_grid=data.frame(.lambda=seq(0,.2,length=20),.alpha=0)
  set.seed(100)
  m_ridge_out<-train(fm_ridge_out,data=ds_out,
                     method='glmnet',
                     metric="RMSE",
                     trControl=tc,
                     tuneGrid=ridge_grid,
                     preProcess=c("center","scale"),
                     na.action=na.pass)
  #build non outlier model
  fm_ridge_nout<-as.formula(paste("target","~",paste(names(ds_nout)[!names(ds_nout) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  
  set.seed(100)
  m_ridge_nout<-train(fm_ridge_nout,data=ds_nout,
                      method='glmnet',
                      metric="RMSE",
                      trControl=tc,
                      tuneGrid=ridge_grid,
                      preProcess=c("center","scale"),
                      na.action=na.pass)
  
  ##predict train test set
  ridge_tr_te_prd_out<-predict(m_ridge_out, subtest, na.action=na.pass)
  ridge_tr_te_prd_nout<-predict(m_ridge_nout, subtest, na.action=na.pass)
  
  #ensumble predictions by adjusting weights
  
  rmse={}
  ens_weight={}
  
  for (w in seq(0,1, 0.1) ){
    
    ens_weight=rbind(ens_weight,w)
    ridge_tr_te_prd= w*ridge_tr_te_prd_out+(1-w)*ridge_tr_te_prd_nout
    rmse_trte=sqrt(mean((ridge_tr_te_prd-subtest$target)^2))
    rmse=rbind(rmse,rmse_trte)
  }
  
  best_w=cbind(rmse,ens_weight)[which.min(cbind(rmse,ens_weight)[,1]),2]
  
  ridge_tr_te_pred= best_w*ridge_tr_te_prd_out+(1-best_w)*ridge_tr_te_prd_nout
  
  
  #predict test set
  ridge_test_pred_out<-predict(m_ridge_out, test, na.action=na.pass)
  ridge_test_pred_nout<-predict(m_ridge_nout, test, na.action=na.pass)
  
  ridge_test_pred= best_w*ridge_test_pred_out+(1-best_w)*ridge_test_pred_nout
  
  truedata<-subtest[,6]
  err_ridge<-sqrt(mean((truedata-ridge_tr_te_pred)^2))
  
  return(list(best_w=best_w, tr_te_err=err_ridge, test_pred=ridge_test_pred))
  
}


#Random Forest

fit_rf_ens<-function(train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  #separate outlier targets and non outlier targets
  
  ds_out<-subtrain[which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  ds_nout<-subtrain[-which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  
  
  #building outlier model
  tc<-trainControl(method="cv", number=10,summaryFunction=defaultSummary,selectionFunction="best")
  fm_rf_out<-as.formula(paste("target","~",paste(names(ds_out)[!names(ds_out) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  
  #rf_grid<- data.frame("mtry"=seq(60,100,5),"min.node.size"= 5, "splitrule"="variance")
  mtry=seq(20,60,10)
  splitrule="variance"
  min.node.size=5
  rf_grid=expand.grid(mtry,splitrule,min.node.size)
  colnames(rf_grid)=c("mtry","splitrule","min.node.size")
  
  set.seed(100)
  m_rf_out<-train(fm_rf_out,ds_out,
                  method='ranger',
                  metric="RMSE",
                  trControl=tc,
                  tuneGrid=rf_grid,
                  num.trees=500,
                  na.action=na.pass, num.threads=threads)
  
  # build non-outlier model
  
  tc<-trainControl(method="cv", number=10,summaryFunction=defaultSummary,selectionFunction="best")
  fm_rf_nout<-as.formula(paste("target","~",paste(names(ds_nout)[!names(ds_nout) %in% c("target","rectday","first_active_month","card_id","mrec_trans","outlier")],collapse = "+"),sep=""))
  
  set.seed(100)
  m_rf_nout<-train(fm_rf_nout,ds_nout,
                   method='ranger',
                   metric="RMSE",
                   trControl=tc,
                   tuneGrid=rf_grid,
                   num.trees=500,
                   na.action=na.pass, num.threads=threads)
  
  ##predict train test set
  rf_tr_te_prd_out<-predict(m_rf_out, subtest,na.action=na.pass)
  rf_tr_te_prd_nout<-predict(m_rf_nout, subtest,na.action=na.pass)
  
  #ensumble predictions by adjusting weights
  
  rmse={}
  ens_weight={}
  
  for (w in seq(0,1, 0.1) ){
    
    ens_weight=rbind(ens_weight,w)
    rf_tr_te_prd= w*rf_tr_te_prd_out+(1-w)*rf_tr_te_prd_nout
    rmse_trte=sqrt(mean((rf_tr_te_prd-subtest$target)^2))
    rmse=rbind(rmse,rmse_trte)
  }
  
  best_w=cbind(rmse,ens_weight)[which.min(cbind(rmse,ens_weight)[,1]),2]
  
  rf_tr_te_pred= best_w*rf_tr_te_prd_out+(1-best_w)*rf_tr_te_prd_nout
  
  
  #predict test set
  rf_test_pred_out<-predict(m_rf_out, test, na.action=na.pass)
  rf_test_pred_nout<-predict(m_rf_nout, test, na.action=na.pass)
  
  rf_test_pred= best_w*rf_test_pred_out+(1-best_w)*rf_test_pred_nout
  
  truedata<-subtest[,6]
  err_rf<-sqrt(mean((truedata-rf_tr_te_pred)^2))
  
  return(list(best_w=best_w, tr_te_err=err_rf, test_pred=rf_test_pred))
}


#XGBOOST

fit_xgb_ens<- function(train, test, index){
  
  subtrain=train[index,]
  subtest=train[-index,]
  
  #separate outlier targets and non outlier targets
  
  ds_out<-subtrain[which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  ds_nout<-subtrain[-which(subtrain$target>=quantile(subtrain$target,probs=1) | subtrain$target<=quantile(subtrain$target, probs=0)),]
  
  formu<-as.formula(paste("target","~-1+",paste(names(subtrain)[!names(subtrain) %in% c("target","first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  
  tr_data_out<- sparse.model.matrix(object=formu,drop.unused.levels = FALSE,data = ds_out)
  tr_data_nout<-sparse.model.matrix(object=formu,drop.unused.levels = FALSE,data = ds_nout)
  tr_ts_data<- sparse.model.matrix(object=formu,drop.unused.levels = FALSE,data = subtest)
  
  formu_ts<-as.formula(paste("~-1+",paste(names(test)[!names(test) %in% c("first_active_month","card_id","mrec_trans")],collapse = "+"),sep=""))
  ts_data<-sparse.model.matrix(object=formu_ts,drop.unused.levels = FALSE,data = fulltest)
  
  tr_label_out=ds_out[,6]
  tr_label_nout=ds_nout[,6]
  tr_ts_label=subtest[,6]
  
  tr_ma_out<-xgb.DMatrix(data=as.matrix(tr_data_out),label=tr_label_out,missing = c(NA,NaN,Inf))
  tr_ma_nout<-xgb.DMatrix(data=as.matrix(tr_data_nout),label=tr_label_nout,missing = c(NA,NaN,Inf))
  tr_ts_ma<-xgb.DMatrix(data=as.matrix(tr_ts_data),label=tr_ts_label,missing = c(NA,NaN,Inf))
  
  ts_ma<-xgb.DMatrix(data=as.matrix(ts_data),missing = c(NA,NaN,Inf))
  
  # grid search on hyperparameters of xgb model with 5 folds cv
  
  booster = "gbtree"
  objective = "reg:linear"
  eval_metric = "rmse"
  eta=0.01 
  gamma=c(0,1)
  alpha=0
  lambda=c(0,5)
  max_depth=c(5,15,25) 
  min_child_weight=c(5,10) 
  subsample=0.8 
  colsample_bytree = 0.7
  colsample_bylevel = 0.6
  
  param_grid <-expand.grid(booster,objective,eval_metric,eta,gamma,alpha,lambda,
                           max_depth, min_child_weight,subsample, colsample_bytree,colsample_bylevel)
  colnames(param_grid) <- c("booster","objective","eval_metric","eta","gamma","alpha",
                            "lambda","max_depth","min_child_weight","subsample","colsample_bytree",
                            "colsample_bylevel")
  
  grid_cv_error<-matrix(0,nrow=nrow(param_grid),ncol=2)
  
  param_grid$booster<-as.character(param_grid$booster)
  param_grid$objective<-as.character(param_grid$objective)
  param_grid$eval_metric<-as.character(param_grid$eval_metric)
  
  #set parallel backend
  #library(parallel)
  #library(parallelMap) 
  #parallelStartSocket(cpus = detectCores())
  
  
  # fit outlier model on grid search 
  foreach (k = 1:nrow(param_grid)) %dopar% {
    
    xgbcv_grid <- xgb.cv(params = as.list(param_grid[k,]), data = tr_ma_out, nrounds = 1000, nfold = 5, showsd = T, 
                         stratified = T, early_stop_round = 100, maximize = F,verbose=0)
    
    grid_cv_error[k,1]<- min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    grid_cv_error[k,2]<- which.min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    
  }
  
  
  params=as.list(param_grid[which.min(grid_cv_error),])
  
  
  m_xgb_out <- xgb.train(params=params, data=tr_ma_out, nrounds=1000, watchlist=list(val=tr_ts_ma,train=tr_ma_out),print_every_n = 100, 
                         early_stopping_rounds = 100)
  
  # fit non-outlier model on grid search 
  foreach (k = 1:nrow(param_grid)) %dopar% {
    
    xgbcv_grid <- xgb.cv(params = as.list(param_grid[k,]), data = tr_ma_nout, nrounds = 1000, nfold = 5, showsd = T, 
                         stratified = T, early_stop_round = 100, maximize = F,verbose=0)
    
    grid_cv_error[k,1]<- min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    grid_cv_error[k,2]<- which.min(xgbcv_grid$evaluation_log[,test_rmse_mean])
    
  }
  
  
  params=as.list(param_grid[which.min(grid_cv_error),])
  
  
  m_xgb_nout <- xgb.train(params=params, data=tr_ma_nout, nrounds=1000, watchlist=list(val=tr_ts_ma,train=tr_ma_nout),print_every_n = 100, 
                          early_stopping_rounds = 100)
  
  
  ##predict train test set
  xgb_tr_te_prd_out<-predict(m_xgb_out, tr_ts_ma, na.action=na.pass)
  xgb_tr_te_prd_nout<-predict(m_xgb_nout, tr_ts_ma, na.action=na.pass)
  
  #ensumble predictions by adjusting weights
  
  rmse={}
  ens_weight={}
  
  for (w in seq(0,1, 0.1) ){
    
    ens_weight=rbind(ens_weight,w)
    xgb_tr_te_prd= w*xgb_tr_te_prd_out+(1-w)*xgb_tr_te_prd_nout
    rmse_trte=sqrt(mean((xgb_tr_te_prd-subtest$target)^2))
    rmse=rbind(rmse,rmse_trte)
  }
  
  best_w=cbind(rmse,ens_weight)[which.min(cbind(rmse,ens_weight)[,1]),2]
  
  xgb_tr_te_pred= best_w*xgb_tr_te_prd_out+(1-best_w)*xgb_tr_te_prd_nout
  
  
  #predict test set
  xgb_test_prd_out<-predict(m_xgb_out, ts_ma, na.action=na.pass)
  xgb_test_prd_nout<-predict(m_xgb_nout, ts_ma,na.action=na.pass)
  
  xgb_test_pred= best_w*xgb_test_prd_out+(1-best_w)*xgb_test_prd_nout
  
  truedata<-subtest[,6]
  err_xgb<-sqrt(mean((truedata-xgb_tr_te_pred)^2))
  
  return(list(best_w=best_w, tr_te_err=err_xgb, test_pred=xgb_test_pred))
  
  
}


##########################
#Evaluation and Prediction
#########################

################
#fulltrain set 1

err_model1=list()
fit_model1=list()

for (i in 1: length(datasplit)){
  split_name <- names(datasplit)[i]
  subsplit <- datasplit[[i]]
  print(paste("Evaluatin on Split-Scheme ",i,"/",length(datasplit),sep=""))
  tr_te_err <- list()
  fit_model<- list()

  # loop through all data splits in trainset
  for (j in 1:length(subsplit)){
    
    # train model on current split
    print(paste("  Fitting Models on Split-Number: ",j,"/",length(subsplit),sep=""))
    
    pred_glm=fit_glm(fulltrain_upd, fulltest, index=subsplit[[j]])
    pred_ridge=fit_ridge(fulltrain_upd, fulltest, index=subsplit[[j]])
    pred_rf=fit_rf(fulltrain_upd, fulltest, index=subsplit[[j]])
    pred_xgb=fit_xgb(fulltrain_upd, fulltest, index=subsplit[[j]])
   
    pred_glm_ens=fit_glm_ens(fulltrain_upd,fulltest,index=subsplit[[j]])
    pred_ridge_ens=fit_ridge_ens(fulltrain_upd, fulltest,index=subsplit[[j]])
    pred_rf_ens=fit_rf_ens(fulltrain_upd,fulltest,index=subsplit[[j]])
    pred_xgb_ens=fit_xgb_ens(fulltrain_upd,fulltest,index=subsplit[[j]])

    te_rf<-pred_rf$test_pred
    te_rfens<-pred_rf_ens$test_pred	
    te_glm<-pred_glm$test_pred
    te_glmens<-pred_glm_ens$test_pred		
    te_ridge<-pred_ridge$test_pred
    te_ridgeens<-pred_ridge_ens$test_pred
    te_xgb<-pred_xgb$test_pred
    te_xgbens<-pred_xgb_ens$test_pred	
    
    write.csv(te_glm, file=paste("te_glm",i,"_",j,".csv",sep=""))
    write.csv(te_glmens, file=paste("te_glmens",i,"_",j,".csv",sep=""))
    write.csv(te_ridge, file=paste("te_ridge",i,"_",j,".csv",sep=""))
    write.csv(te_ridgeens, file=paste("te_ridgeens",i,"_",j,".csv",sep=""))
    write.csv(te_rf, file=paste("te_rf",i,"_",j,".csv",sep=""))
    write.csv(te_rfens, file=paste("te_rfens",i,"_",j,".csv",sep=""))
    write.csv(te_xgb, file=paste("te_xgb",i,"_",j,".csv",sep=""))
    write.csv(te_xgbens, file=paste("te_xgbens",i,"_",j,".csv",sep=""))
    
    
    tr_te_err[[j]]=list(glm=pred_glm$tr_te_err, ridge=pred_ridge$tr_te_err,rf=pred_rf$tr_te_err,
                        xgb=pred_xgb$tr_te_err,glm_ensumble=pred_glm_ens$tr_te_err,
                        ridge_ensumble=pred_ridge_ens$tr_te_err,
                        rf_ensumble=pred_rf_ens$tr_te_err,xgb_ensumble=pred_xgb_ens$tr_te_err)
    
    fit_model[[j]]=list(glm=pred_glm$fit, ridge=pred_ridge$fit,rf=pred_rf$fit,xgb=pred_xgb$fit,
                        glm_ensumble=pred_glm_ens$best_w,ridge_ensumble=pred_ridge_ens$best_w,
                        rf_ensumble=pred_rf_ens$best_w, xgb_ensumble=pred_xgb_ens$best_w)
    
  }
  
  err_model1[[i]]=tr_te_err
  fit_model1[[i]]=fit_model
  
}

save(file='err_model1.RData', list="err_model1")
save(file='fit_model1.RData', list="fit_model1")

##################
#fulltrain set 2

err_model2=list()
fit_model2=list()

for (i in 1: length(datasplit)){
  split_name <- names(datasplit)[i]
  subsplit <- datasplit[[i]]
  print(paste("Evaluatin on Split-Scheme ",i,"/",length(datasplit),sep=""))
  tr_te_err <- list()
  fit_model<- list()
  
  # loop through all data splits in trainset
  for (j in 1:length(subsplit)){
    
    # train model on current split
    print(paste("  Fitting Models on Split-Number: ",j,"/",length(subsplit),sep=""))
    
    pred_glm=fit_glm(fulltrain_upd2, fulltest, index=subsplit[[j]])
    pred_ridge=fit_ridge(fulltrain_upd2, fulltest, index=subsplit[[j]])
    pred_rf=fit_rf(fulltrain_upd2, fulltest, index=subsplit[[j]])
    pred_xgb=fit_xgb(fulltrain_upd2, fulltest_2, index=subsplit[[j]])
    
    pred_glm_ens=fit_glm_ens(fulltrain_upd2,fulltest,index=subsplit[[j]])
    pred_ridge_ens=fit_ridge_ens(fulltrain_upd2, fulltest,index=subsplit[[j]])
    pred_rf_ens=fit_rf_ens(fulltrain_upd2,fulltest,index=subsplit[[j]])
    pred_xgb_ens=fit_xgb_ens(fulltrain_upd2,fulltest_2,index=subsplit[[j]])
    
    te_rf<-pred_rf$test_pred
    te_rfens<-pred_rf_ens$test_pred	
    te_glm<-pred_glm$test_pred
    te_glmens<-pred_glm_ens$test_pred		
    te_ridge<-pred_ridge$test_pred
    te_ridgeens<-pred_ridge_ens$test_pred
    te_xgb<-pred_xgb$test_pred
    te_xgbens<-pred_xgb_ens$test_pred	
    
    write.csv(te_glm, file=paste("te_glm",i,"_",j,".csv",sep=""))
    write.csv(te_glmens, file=paste("te_glmens",i,"_",j,".csv",sep=""))
    write.csv(te_ridge, file=paste("te_ridge",i,"_",j,".csv",sep=""))
    write.csv(te_ridgeens, file=paste("te_ridgeens",i,"_",j,".csv",sep=""))
    write.csv(te_rf, file=paste("te_rf",i,"_",j,".csv",sep=""))
    write.csv(te_rfens, file=paste("te_rfens",i,"_",j,".csv",sep=""))
    write.csv(te_xgb, file=paste("te_xgb",i,"_",j,".csv",sep=""))
    write.csv(te_xgbens, file=paste("te_xgbens",i,"_",j,".csv",sep=""))
    
    
    tr_te_err[[j]]=list(glm=pred_glm$tr_te_err, ridge=pred_ridge$tr_te_err,rf=pred_rf$tr_te_err,
                        xgb=pred_xgb$tr_te_err,glm_ensumble=pred_glm_ens$tr_te_err,
                        ridge_ensumble=pred_ridge_ens$tr_te_err,
                        rf_ensumble=pred_rf_ens$tr_te_err,xgb_ensumble=pred_xgb_ens$tr_te_err)
    
    fit_model[[j]]=list(glm=pred_glm$fit, ridge=pred_ridge$fit,rf=pred_rf$fit,xgb=pred_xgb$fit,
                        glm_ensumble=pred_glm_ens$best_w,ridge_ensumble=pred_ridge_ens$best_w,
                        rf_ensumble=pred_rf_ens$best_w, xgb_ensumble=pred_xgb_ens$best_w)
    
  }
  
  err_model2[[i]]=tr_te_err
  fit_model2[[i]]=fit_model
  
}

save(file='err_model2.RData', list="err_model2")
save(file='fit_model2.RData', list="fit_model2")

################
#fulltrain set3

err_model3=list()
fit_model3=list()

for (i in 1: length(datasplit)){
  split_name <- names(datasplit)[i]
  subsplit <- datasplit[[i]]
  print(paste("Evaluatin on Split-Scheme ",i,"/",length(datasplit),sep=""))
  tr_te_err <- list()
  fit_model<- list()
  
  # loop through all data splits in trainset
  for (j in 1:length(subsplit)){
    
    # train model on current split
    print(paste("  Fitting Models on Split-Number: ",j,"/",length(subsplit),sep=""))
    
    pred_glm=fit_glm(fulltrain_upd3, fulltest, index=subsplit[[j]])
    pred_ridge=fit_ridge(fulltrain_upd3, fulltest, index=subsplit[[j]])
    pred_rf=fit_rf(fulltrain_upd3, fulltest, index=subsplit[[j]])
    pred_xgb=fit_xgb(fulltrain_upd3, fulltest_3, index=subsplit[[j]])
    
    pred_glm_ens=fit_glm_ens(fulltrain_upd3,fulltest,index=subsplit[[j]])
    pred_ridge_ens=fit_ridge_ens(fulltrain_upd3, fulltest,index=subsplit[[j]])
    pred_rf_ens=fit_rf_ens(fulltrain_upd3,fulltest,index=subsplit[[j]])
    pred_xgb_ens=fit_xgb_ens(fulltrain_upd3,fulltest_3,index=subsplit[[j]])
    
    te_rf<-pred_rf$test_pred
    te_rfens<-pred_rf_ens$test_pred	
    te_glm<-pred_glm$test_pred
    te_glmens<-pred_glm_ens$test_pred		
    te_ridge<-pred_ridge$test_pred
    te_ridgeens<-pred_ridge_ens$test_pred
    te_xgb<-pred_xgb$test_pred
    te_xgbens<-pred_xgb_ens$test_pred	
    
    write.csv(te_glm, file=paste("te_glm",i,"_",j,".csv",sep=""))
    write.csv(te_glmens, file=paste("te_glmens",i,"_",j,".csv",sep=""))
    write.csv(te_ridge, file=paste("te_ridge",i,"_",j,".csv",sep=""))
    write.csv(te_ridgeens, file=paste("te_ridgeens",i,"_",j,".csv",sep=""))
    write.csv(te_rf, file=paste("te_rf",i,"_",j,".csv",sep=""))
    write.csv(te_rfens, file=paste("te_rfens",i,"_",j,".csv",sep=""))
    write.csv(te_xgb, file=paste("te_xgb",i,"_",j,".csv",sep=""))
    write.csv(te_xgbens, file=paste("te_xgbens",i,"_",j,".csv",sep=""))
    
    
    tr_te_err[[j]]=list(glm=pred_glm$tr_te_err, ridge=pred_ridge$tr_te_err,rf=pred_rf$tr_te_err,
                        xgb=pred_xgb$tr_te_err,glm_ensumble=pred_glm_ens$tr_te_err,
                        ridge_ensumble=pred_ridge_ens$tr_te_err,
                        rf_ensumble=pred_rf_ens$tr_te_err,xgb_ensumble=pred_xgb_ens$tr_te_err)
    
    fit_model[[j]]=list(glm=pred_glm$fit, ridge=pred_ridge$fit,rf=pred_rf$fit,xgb=pred_xgb$fit,
                        glm_ensumble=pred_glm_ens$best_w,ridge_ensumble=pred_ridge_ens$best_w,
                        rf_ensumble=pred_rf_ens$best_w, xgb_ensumble=pred_xgb_ens$best_w)
    
  }
  
  err_model3[[i]]=tr_te_err
  fit_model3[[i]]=fit_model
  
}

save(file='err_model3.RData', list="err_model3")
save(file='fit_model3.RData', list="fit_model3")


stopCluster(cl)

