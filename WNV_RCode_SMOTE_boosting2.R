##################################################################################################
## Apply SMOTE to balance the dataset
## Modeling: boosting
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
data.wnv = fread("/home/linpyl/ADM/SMOTE/train_new.csv")
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
## Modeling: boosting
##############################################################################################
set.seed(sample(1:100000,1))
repeats = 20
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
    
    train.data$WnvPresent = ifelse(train.data$WnvPresent==1,"YES","NO")
    test.data$WnvPresent = ifelse(test.data$WnvPresent==1,"YES","NO")
    
    ## preProcess(): test data set
    x.test = test.data[,c("NumMosquitos","daytime","HotSpot","Month","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1")]
    y.test = test.data[,c("WnvPresent")]
    preProc = preProcess(x.test[,-4], method=c("center","scale"))
    x.test.1 = cbind.data.frame(predict(preProc,x.test[,-4]),Month=x.test[,4])
    test  = cbind.data.frame(x.test.1, WnvPresent=y.test)
    
    K = 10
    block = sample(1:K, nrow(train.data), replace=TRUE)
    
    for (j in 1:K) {
        #print(paste0("repeats=",i," K=",j))
        x.train.train = train.data[block!=j,c("NumMosquitos","daytime","HotSpot","Month","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1")]
        y.train.train = train.data[block!=j,c("WnvPresent")]
        preProc = preProcess(x.train.train[,-4], method=c("center","scale"))
        x.train.train.1 = cbind.data.frame(predict(preProc,x.train.train[,-4]), Month=x.train.train[,4])
        train.train = cbind.data.frame(x.train.train.1, WnvPresent=y.train.train)
        
        x.train.test = train.data[block==j,c("NumMosquitos","daytime","HotSpot","Month","Tavg.ma2","RH.ma4","PrecipTotal.ma2","DewPoint.ma1")]
        y.train.test = train.data[block==j,c("WnvPresent")]
        preProc = preProcess(x.train.test[,-4], method=c("center","scale"))
        x.train.test.1 = cbind.data.frame(predict(preProc,x.train.test[,-4]), Month=x.train.test[,4])
        train.test = cbind.data.frame(x.train.test.1, WnvPresent=y.train.test)
        
        #################################################
        ## gbm
        #################################################
        depth.list = 10 #c(seq(25, 275, 25), seq(300, 1000, 100))
        ntree.list = c(seq(25, 275, 25), seq(300, 1000, 100))
        shrinkage.list = c(0.001, 0.005, 0.01, 0.05, 0.1)
        output2 = vector()
        for (Ntree in 1:length(ntree.list)) {
            for (Shrinkage in 1:length(shrinkage.list)) {
                objControl = trainControl(returnResamp="none", 
                                          summaryFunction=twoClassSummary, 
                                          classProbs=TRUE,
                                          #allowParallel=TRUE,
                                          method="none" #, number=10, 
                )
                mod = train(WnvPresent ~ .,
                            data=train.train,
                            method='gbm', 
                            trControl=objControl, 
                            verbose=FALSE,
                            tuneGrid=data.frame(interaction.depth=depth.list,
                                                n.trees=ntree.list[Ntree],
                                                shrinkage=shrinkage.list[Shrinkage]),
                            metric="ROC")
                
                ###############################
                ## Optimize probability cutoff
                ###############################
                pred = predict(mod, train.test, type="prob")[,2]
                actual = ifelse(train.test$WnvPresent=="YES",1,0)
                table1 = f.cutoff.optimizer(pred=pred, y=actual)
                temp = cbind(depth=depth.list,
                             ntree=ntree.list[Ntree],
                             shrinkage=shrinkage.list[Shrinkage],
                             cutoff=table1[which.max(table1[,2]),1], 
                             auc=table1[which.max(table1[,2]),2])
                output2 = rbind(output2, temp)
            }
        } 
        
        #################################################
        ## Find the optimal ntree and probability cutoff
        #################################################
        par.depth  = output2[which.max(output2[,c("auc")]),1]
        par.ntree  = output2[which.max(output2[,c("auc")]),2]
        par.shrinkage = output2[which.max(output2[,c("auc")]),3]
        par.cutoff = output2[which.max(output2[,c("auc")]),4]
        
        ###########################################################
        ## Use the optimal ntree to construct new model  
        ###########################################################
        objControl = trainControl(returnResamp="none", 
                                  summaryFunction=twoClassSummary, 
                                  classProbs=TRUE,
                                  #allowParallel=TRUE,
                                  method="none" #, number=10, 
        )
        mod = train(WnvPresent ~ .,
                    data=test,
                    method='gbm', 
                    trControl=objControl, 
                    verbose=FALSE,
                    tuneGrid=data.frame(interaction.depth=par.depth,
                                        n.trees=par.ntree,
                                        shrinkage=par.shrinkage),
                    metric="ROC")
        
        ###########################################################
        ## Use the optimal cutoff to predict the test dataset 
        ###########################################################
        test.pred = predict(mod, test, type="prob")[,2]
        pred = ifelse(test.pred>=par.cutoff, 1, 0)
        actual = ifelse(test$WnvPresent=="YES",1,0)
        conf.table = confusionMatrix(table(pred=pred, actual=actual))
        acc = as.numeric(conf.table$overall[1])
        sens = as.numeric(conf.table$byClass[2])
        spec = as.numeric(conf.table$byClass[1])
        kappa = as.numeric(conf.table$overall[2])
        
        summary.table = rbind(summary.table, c(Accuracy=acc, Sensitivity=sens, Specificity=spec, Kappa=kappa))
        print(c(acc, sens, spec, kappa))
    }
}

## write output
write.csv(summary.table, "/home/linpyl/ADM/SMOTE/predict_smote_boosting2.csv")

apply(summary.table,2,mean)
#Accuracy Sensitivity Specificity       Kappa 
