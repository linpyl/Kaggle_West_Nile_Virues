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
        result = prediction(yhat, y)
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
data.wnv = fread("train_new.csv")
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
#test.wnv = fread("test_new.csv")
#test.wnv = data.frame(test.wnv)
## data preparation
#test.wnv$Month = as.factor(test.wnv$Month)
#test.wnv$Tavg.ordinal = as.factor(test.wnv$Tavg.ordinal)
#test.wnv$HeavyRain = as.factor(test.wnv$HeavyRain)
#test.wnv$LowWind.byMean = as.factor(test.wnv$LowWind.byMean)
#test.wnv$LowWind.byLow = as.factor(test.wnv$LowWind.byLow)
#test$Species = as.factor(test.wnv$Species)


##############################################################################################
## select model formula
##############################################################################################
fit.formula = as.formula(WnvPresent ~ Month + Species + Tavg.ordinal + RH.ma4 + 
                                      DewPoint.ma1 + LowWind.byMean + daytime +
                                      log.HotSpot + NumMosquitos + PrecipTotal.ma3)
mod1 = glm(fit.formula, data=data.wnv, family=binomial(link="logit"))
summary(mod1)
AIC(mod1)
pred = predict(mod1, data.wnv, type="response")
table1 = f.cutoff.optimizer(pred=pred, y=data.wnv$WnvPresent)
(cutoff = table1[which.max(table1[,2]),1])
(auc = table1[which.max(table1[,2]),2])
# cutoff=0.1, auc=0.7556581

mod2 = step(mod1, direction="both")
summary(mod2)
AIC(mod2)
pred = predict(mod2, data.wnv, type="response")
table1 = f.cutoff.optimizer(pred=pred, y=data.wnv$WnvPresent)
(cutoff = table1[which.max(table1[,2]),1])
(auc = table1[which.max(table1[,2]),2])
# cutoff=0.1, auc=0.7558452

mod2$call[2]


##############################################################################################
## Modeling: GLM
##############################################################################################
set.seed(1)
repeats = 20
summary.table = vector()
for (i in 1:repeats) {
    ## split data into 70-30 train-test data sets
    data.wnv.yes = data.wnv[data.wnv$WnvPresent==1,]
    data.wnv.no = data.wnv[data.wnv$WnvPresent==0,]
    yes.train.index = createDataPartition(data.wnv.yes$WnvPresent,p=0.7, list=F)
    no.train.index = sample(1:nrow(data.wnv.no), size=ceiling(length(yes.train.index)*0.55/0.45), replace=FALSE)
    temp = data.wnv.no[-(no.train.index),]
    no.test.index = sample(1:nrow(temp), size=ceiling((nrow(data.wnv.yes)-length(yes.train.index))*0.55/0.45), replace=FALSE)
    train.data = rbind(data.wnv.yes[yes.train.index,], data.wnv.no[no.train.index,])
    test.data = rbind(data.wnv.yes[-(yes.train.index),], temp[no.test.index,])
        
    #################################################
    ## Optimize probability cutoff using 10-fold CV
    #################################################
    K = 10
    block = sample(1:K, nrow(train.data), replace=TRUE)
    
    for (j in 1:K) {
        #======[Put your model here]==============================================================
        fit.formula = as.formula(WnvPresent ~ Month + Species + Tavg.ordinal + RH.ma4 + DewPoint.ma1 + 
                                 LowWind.byMean + daytime + log.HotSpot + NumMosquitos + PrecipTotal.ma3)
        
        mod1 = glm(fit.formula, data=train.data[block!=j,], family=binomial(link="logit"))
        mod2 = step(mod1, direction="both")
        
        ###############################
        ## Optimize probability cutoff
        ###############################
        pred = predict(mod2, train.data[block==j,], type="response")
        table1 = f.cutoff.optimizer(pred=pred, y=train.data[block==j,]$WnvPresent)
        cutoff = table1[which.max(table1[,2]),1]
                        
        ###########################################################
        ## Use the optimal cutoff to predict the test.data dataset 
        ###########################################################
        test.pred = predict(mod2, test.data, type="response")
        pred = ifelse(test.pred>=cutoff, 1, 0)
        conf.table = confusionMatrix(table(pred=pred, actual=test.data$WnvPresent))
        acc = as.numeric(conf.table$overall[1])
        sens = as.numeric(conf.table$byClass[2])
        spec = as.numeric(conf.table$byClass[1])
        kappa = as.numeric(conf.table$overall[2])
        
        summary.table = rbind(summary.table, c(Accuracy=acc, Sensitivity=sens, Specificity=spec, Kappa=kappa))
    }
}    

## write output
#write.csv(summary.table, "predict_glm.csv")

apply(summary.table,2,mean)
#Accuracy Sensitivity Specificity       Kappa 
#0.7679738   0.8041752   0.7384524   0.5366953 
