##################################################################################################
## Apply SMOTE to balance the dataset
## Modeling: SVM with radial kernel basis 
##################################################################################################
library(data.table)
library(caret)
library(DMwR)
library(leaps)
library(glmnet)
library(randomForest)
library(pROC)
library(ROCR)
library(e1071)
library(gbm)

f.cutoff.optimizer = function(pred, y) {
    output.auc <- vector()
    grid = seq(0.01, 0.99, by=0.01)
    for (cut.i in 1:length(grid)) {
        yhat = ifelse(pred >= grid[cut.i], 1, 0)
        result = ROCR::prediction(yhat, y)
        perf = performance(result,"tpr","fpr")
        auc = performance(result,"auc")
        auc = unlist(slot(auc, "y.values"))
        output.auc = rbind(output.auc, auc)
    }   
    output = cbind(grid, output.auc)
    return(output)
}

##############################################################################################
## read training data
##############################################################################################
data.wnv = fread("D:/Pinsker/Kaggle/kaggle/train_new.csv")
data.wnv = data.frame(data.wnv)
## data preparation
data.wnv = data.wnv[,c(-1,-2,-3,-5)]
data.wnv$WnvPresent = as.factor(data.wnv$WnvPresent)
data.wnv$Month = as.factor(data.wnv$Month)
data.wnv$Tavg.ordinal = as.factor(data.wnv$Tavg.ordinal)
data.wnv$HeavyRain = as.factor(data.wnv$HeavyRain)
data.wnv$LowWind.byMean = as.factor(data.wnv$LowWind.byMean)
data.wnv$LowWind.byLow = as.factor(data.wnv$LowWind.byLow)
data.wnv$Species = ifelse(data.wnv$Species=="CULEX PIPIENS" 
                          | data.wnv$Species=="CULEX PIPIENS/RESTUANS" 
                          | data.wnv$Species=="CULEX RESTUANS", 1, 0)
data.wnv$Species = as.factor(data.wnv$Species)

##############################################################################################
## read test.csv data
##############################################################################################
#test.wnv = fread("D:/Pinsker/Kaggle/kaggle/test_new.csv")
#test.wnv = data.frame(test.wnv)
## data preparation
#test.wnv$Month = as.factor(test.wnv$Month)
#test.wnv$Tavg.ordinal = as.factor(test.wnv$Tavg.ordinal)
#test.wnv$HeavyRain = as.factor(test.wnv$HeavyRain)
#test.wnv$LowWind.byMean = as.factor(test.wnv$LowWind.byMean)
#test.wnv$LowWind.byLow = as.factor(test.wnv$LowWind.byLow)
#test$Species = as.factor(test.wnv$Species)

##############################################################################################
## Modeling: SVM with radial kernel basis
##############################################################################################
set.seed(sample(1:100000,1))
repeats = 200
summary.table = vector()
for (i in 1:repeats) {
    ## split data into 70-30 train-test data sets
    data.wnv.yes = data.wnv[data.wnv$WnvPresent==1,]
    data.wnv.no = data.wnv[data.wnv$WnvPresent==0,]
    yes.train.index = createDataPartition(data.wnv.yes$WnvPresent,p=0.7, list=F)
    no.train.index = createDataPartition(data.wnv.no$WnvPresent,p=0.7, list=F)
    train.data = rbind(data.wnv.yes[yes.train.index,], data.wnv.no[no.train.index,])
    test.data = rbind(data.wnv.yes[-(yes.train.index),], data.wnv.no[-(no.train.index),])
    
    #==========================================================================================
    # Transform imbalanced data into balanced data using SMOTE
    #==========================================================================================
    count = table(train.data$WnvPresent)
    Over = ((0.45*count[1]) - count[2])/count[2]
    Under = (0.55*count[1])/(count[2]*Over)
    Over_Perc = round(Over, 1)*100
    Under_Perc = round(Under, 1)*100
    newData = SMOTE(WnvPresent ~., train.data, k=5, perc.over=Over_Perc, perc.under=Under_Perc)
    train.data = newData
    
    ## preProcess(): test data set
    x.test = test.data[,c("NumMosquitos","daytime","HotSpot","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1","Month","Species")]
    y.test = test.data[,c("WnvPresent")]
    preProc = preProcess(x.test[,c(-8,-9)], method=c("center","scale"))
    x.test.1 = cbind.data.frame(predict(preProc,x.test[,c(-8,-9)]), Month=test.data[,"Month"], Species=test.data[,"Species"])
    test  = cbind.data.frame(x.test.1, WnvPresent=y.test)
    
    K = 10
    block = sample(1:K, nrow(train.data), replace=TRUE)
    output2 = vector()
    for (j in 1:K) {
        #print(paste0("repeats=",i," K=",j))
        x.train.train = train.data[block!=j,c("NumMosquitos","daytime","HotSpot","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1","Month","Species")]
        y.train.train = train.data[block!=j,c("WnvPresent")]
        preProc = preProcess(x.train.train[,c(-8,-9)], method=c("center","scale"))
        x.train.train.1 = cbind.data.frame(predict(preProc,x.train.train[,c(-8,-9)]), Month=x.train.train[,"Month"], Species=x.train.train[,"Species"])
        train.train = cbind.data.frame(x.train.train.1, WnvPresent=y.train.train)
        
        x.train.test = train.data[block==j,c("NumMosquitos","daytime","HotSpot","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1","Month","Species")]
        y.train.test = train.data[block==j,c("WnvPresent")]
        preProc = preProcess(x.train.test[,c(-8,-9)], method=c("center","scale"))
        x.train.test.1 = cbind.data.frame(predict(preProc,x.train.test[,c(-8,-9)]), Month=x.train.test[,"Month"], Species=x.train.test[,"Species"])
        train.test = cbind.data.frame(x.train.test.1, WnvPresent=y.train.test)
        
        #################################################
        ## SVM
        #################################################
        m = model.matrix(~ WnvPresent + NumMosquitos + daytime + HotSpot + Tavg.ma2 
                         + RH.ma4 + PrecipTotal.ma2 + DewPoint.ma1 + Month + Species, 
                         data=train.train)[,-1]
        
        mod = svm(WnvPresent1 ~ ., data=m, kernel="radial", cost=5, gamma=0.1, probability=TRUE, type="C")
        
        ###############################
        ## Optimize probability cutoff
        ###############################
        m = model.matrix(~ WnvPresent+ NumMosquitos + daytime + HotSpot + Tavg.ma2 
                         + RH.ma4 + PrecipTotal.ma2 + DewPoint.ma1 + Month + Species, 
                         data=train.test)[,-1]
        pred.svm = predict(mod, m, probability=TRUE)
        pred = attr(pred.svm, "probabilities")[,2]
        table1 = f.cutoff.optimizer(pred=pred, y=train.test$WnvPresent)
        cutoff = table1[which.max(table1[,2]),1]
        auc = table1[which.max(table1[,2]),2]
        output2 = rbind(output2, c(cutoff=cutoff, auc=auc)) 
    }
    
    #################################################
    ## Find the optimal ntree and probability cutoff
    #################################################
    par.cutoff = output2[which.max(output2[,c("auc")]),1]
    
    ###########################################################
    ## Use the optimal ntree to construct new model  
    ###########################################################
    train = rbind.data.frame(train.train, train.test)
    m = model.matrix(~ WnvPresent + NumMosquitos + daytime + HotSpot + Tavg.ma2 
                     + RH.ma4 + PrecipTotal.ma2 + DewPoint.ma1 + Month + Species, 
                     data=train)[,-1]
    mod = svm(WnvPresent1 ~ ., data=m, kernel="radial", cost=5, gamma=0.1, probability=TRUE, type="C")
    
    ###########################################################
    ## Use the optimal cutoff to predict the test dataset 
    ###########################################################
    m = model.matrix(~ WnvPresent+ NumMosquitos + daytime + HotSpot + Tavg.ma2 
                     + RH.ma4 + PrecipTotal.ma2 + DewPoint.ma1 + Month + Species, 
                     data=test)
    pred.svm = predict(mod, m, probability=TRUE)
    test.pred = attr(pred.svm, "probabilities")[,2]
    pred = ifelse(test.pred>=par.cutoff, 1, 0)
    conf.table = confusionMatrix(table(pred=pred, actual=test$WnvPresent))
    acc = as.numeric(conf.table$overall[1])
    sens = as.numeric(conf.table$byClass[2])
    spec = as.numeric(conf.table$byClass[1])
    kappa = as.numeric(conf.table$overall[2])
    
    summary.table = rbind(summary.table, c(Accuracy=acc, Sensitivity=sens, Specificity=spec, Kappa=kappa))
    print(c(acc, sens, spec, kappa))
}

## write output
write.csv(summary.table, "/home/linpyl/ADM/SMOTE/predict_smote_svm_radial.csv")

apply(summary.table,2,mean)
#Accuracy Sensitivity Specificity       Kappa     
