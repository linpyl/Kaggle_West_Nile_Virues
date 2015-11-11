##################################################################################################
## Apply SMOTE to balance the dataset
## Modeling: random forest
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

f.cutoff.optimizer = function(pred, y) {
    output.auc <- vector()
    grid = seq(0.1, 0.99, by=0.01)
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
## Modeling: random Forest
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
    
    #################################################
    ## Optimize probability cutoff using 10-fold CV
    #################################################
    K = 10
    block = sample(1:K, nrow(train.data), replace=TRUE)
    output = vector()
    for (j in 1:K) {
        #print(paste0("repeats=",i," K=",j))
        #======[Put your model here]==============================================================
        fit.formula = as.formula(WnvPresent ~ NumMosquitos + daytime + Month + log.HotSpot 
                                 + Tavg.ma2 + PrecipTotal.ma3 + DewPoint.ma1 + AvgSpeed.ma3
                                 + Species + RH)
        NumVariable = 10
        ntree.list = c(seq(25, 275, 25), seq(300, 1000, 100))
        
        for (p in 1:length(NumVariable)) {
            for (q in 1:length(ntree.list)) {
                mod = randomForest(fit.formula, train.data[block!=j,], 
                                   mtry=p, ntree=ntree.list[q])
                ###############################
                ## Optimize probability cutoff
                ###############################
                pred = predict(mod, train.data[block==j,], type="prob")
                table1 = f.cutoff.optimizer(pred=pred[,2], y=train.data[block==j,]$WnvPresent)
                temp = cbind(Nvar=p,
                             ntree=ntree.list[q],
                             cutoff=table1[which.max(table1[,2]),1], 
                             auc=table1[which.max(table1[,2]),2])
                output = rbind(output, temp)
                
            }
        }
    }
    #################################################
    ## Find the optimal ntree and probability cutoff
    #################################################
    par.Nvar   = output[which.max(output[,3]),1]
    par.ntree  = output[which.max(output[,3]),2]
    par.cutoff = output[which.max(output[,3]),3]
    
    ###########################################################
    ## Use the optimal ntree to construct new model  
    ###########################################################
    mod = randomForest(fit.formula, train.data, 
                       mtry=par.Nvar, ntree=par.ntree)
    
    ###########################################################
    ## Use the optimal cutoff to predict the test.data dataset 
    ###########################################################
    test.pred = predict(mod, test.data, type="prob")[,2]
    pred = ifelse(test.pred>=par.cutoff, 1, 0)
    conf.table = confusionMatrix(table(pred=pred, actual=test.data$WnvPresent))
    acc = as.numeric(conf.table$overall[1])
    sens = as.numeric(conf.table$byClass[2])
    spec = as.numeric(conf.table$byClass[1])
    kappa = as.numeric(conf.table$overall[2])
    
    summary.table = rbind(summary.table, c(Accuracy=acc, Sensitivity=sens, Specificity=spec, Kappa=kappa))
    print(c(acc, sens, spec, kappa))
}

## write output
write.csv(summary.table, "D:/Pinsker/Kaggle/kaggle/predict_smote_rf.csv")

apply(summary.table,2,mean)
#Accuracy Sensitivity Specificity       Kappa 

