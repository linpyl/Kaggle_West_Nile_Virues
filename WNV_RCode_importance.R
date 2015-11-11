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

# function to handle the situation where function returning more than one value
':=' = function(lhs, rhs) {
    frame = parent.frame()
    lhs = as.list(substitute(lhs))
    if (length(lhs) > 1)
        lhs = lhs[-1]
    if (length(lhs) == 1) {
        do.call(`=`, list(lhs[[1]], rhs), envir=frame)
        return(invisible(NULL)) }
    if (is.function(rhs) || is(rhs, 'formula'))
        rhs = list(rhs)
    if (length(lhs) > length(rhs))
        rhs = c(rhs, rep(list(NULL), length(lhs) - length(rhs)))
    for (i in 1:length(lhs))
        do.call(`=`, list(lhs[[i]], rhs[[i]]), envir=frame)
    return(invisible(NULL)) 
}

# set up lm model
modeller = function(form, data) {
    lm(fit.formula, data=data) 
}

# define predict.regsubsets function to compute the predicted y values using models from regsubsets results
predict.regsubsets = function(object, newdata, id, ...) {
    form = as.formula(object$call[[2]])
    mat = model.matrix(form, newdata)
    coefi = coef(object, id=id)
    mat[, names(coefi)] %*% coefi
}

# define mse.fn function for the calculation of MSE 
mse.fn = function(y, y_hat) {
    mean((y-y_hat)^2)
}

# Split function to split data set into training and test sub sets:
split.fn = function(data, probtrain, probtest) {
    # label training and test data sets
    index = sample(x=c(0,1), size=nrow(data), prob=c(probtrain,probtest), replace=TRUE)
    # split data set into training and test
    data.train = data[index==0,]
    data.test = data[index==1,]  
    return(list(data.train, data.test))
}

# define predict.mse.fn function for calulation of MSE for model prediction
predict.mse.fn = function(data, Ycol=2, probtrain, probtest) {
    # call split.fn function to split data into training and test
    c(data.train, data.test) := split.fn(data, probtrain, probtest)
    # call lm.fit function to construct lm model using training data set
    lm.fit = modeller(form, data=data.train)
    # predict using test data set
    lm.pred = predict(lm.fit, newdata=data.test)
    # call mse.fn function to calculate MSE of prediction
    mse.fn(data.test[, Ycol], lm.pred)
}

# Repeat function
Repeat.mse.fn = function(data, Repeat, Ycol, probtrain, probtest) {
    mse.table = matrix(0, Repeat, dimnames=(list(1:Repeat,c("MSE"))))
    for (i in 1:Repeat) {
        mse.i = predict.mse.fn(data, Ycol, probtrain, probtest)
        mse.table[i] = mse.i
    }
    return(mse.table)
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
## Lasso: Importance of variables for the prediction of WnvPresent
##############################################################################################
## fit.formula: (WnvPresent ~.)
data.wnv = data.wnv[,c(-3,-4)]
n = names(data.wnv)
fit.formula = as.formula(paste("WnvPresent ~", paste(n[!n %in% c("WnvPresent")], collapse=" + ")))
x = model.matrix(fit.formula, data.wnv)[,-1]
y = data.wnv[,c("WnvPresent")]

mod.lasso = cv.glmnet(x, y, alpha=1, type.measure="auc", family="binomial")
bestlam = mod.lasso$lambda.min
predict(mod.lasso, x, lambda=bestlam, type="coefficient")


##############################################################################################
## Bagging: Importance of variables for the prediction of WnvPresent
##############################################################################################
n = names(data.wnv)
fit.formula = as.formula(paste("WnvPresent ~", paste(n[!n %in% c("WnvPresent")], collapse=" + ")))
mod.bag = randomForest(fit.formula, data=data.wnv, mtry=ncol(data.wnv)-1, ntree=1000, importance=TRUE)    
## plot importnace
varImpPlot(mod.bag, sort=TRUE, n.var=min(24, nrow(mod.bag$importance)),
           type=NULL, class=NULL, scale=TRUE, 
           main=deparse(substitute("variable importance ranking by bagging"))) 


##############################################################################################
## Boosting: Importance of variables for the prediction of WnvPresent
##############################################################################################
n = names(data.wnv)
fit.formula = as.formula(paste("WnvPresent ~", paste(n[!n %in% c("WnvPresent")], collapse=" + ")))

data.wnv$WnvPresent = ifelse(data.wnv$WnvPresent==1,"YES","NO")
data.wnv$WnvPresent = as.factor(data.wnv$WnvPresent)
data.wnv = data.wnv[,c(-3,-4)]
x = data.wnv[,1:23]
y = data.wnv[,c("WnvPresent")]
## scale numeric x
x$Tavg = scale(x$Tavg)
x$Tavg.ma1 = scale(x$Tavg.ma1)
x$Tavg.ma2 = scale(x$Tavg.ma2)
x$RH = scale(x$RH)
x$RH.ma4 = scale(x$RH.ma4)
x$DewPoint = scale(x$DewPoint)
x$DewPoint.ma1 = scale(x$DewPoint.ma1)
x$PrecipTotal = scale(x$PrecipTotal)
x$PrecipTotal.ma2 = scale(x$PrecipTotal.ma2)
x$PrecipTotal.ma3 = scale(x$PrecipTotal.ma3)
x$AvgSpeed = scale(x$AvgSpeed)
x$AvgSpeed.ma3 = scale(x$AvgSpeed.ma3)
x$daytime = scale(x$daytime)
x$daytime.ma4 = scale(x$daytime.ma4)
x$log.HotSpot = scale(x$log.HotSpot)
x$HotSpot = scale(x$HotSpot)
x$NumMosquitos = scale(x$NumMosquitos)

x = cbind(x, WnvPresent=y)

#getModelInfo(model = "gbm", regex = FALSE)

objControl = trainControl(returnResamp="none", 
                          summaryFunction=twoClassSummary, 
                          classProbs=TRUE,
                          #allowParallel=TRUE,
                          method="none" #, number=10, 
                          )

mod.boost = train(WnvPresent ~.,
                  data=x,
                  method='gbm', 
                  trControl=objControl, 
                  verbose=FALSE,
                  tuneGrid=data.frame(interaction.depth=4,
                                      n.trees=100,
                                      shrinkage=0.001),
                  metric="ROC")
                  #,preProc=c("center","scale"),

## plot importnace
summary(mod.boost)


# predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
# print(postResample(pred=predictions, obs=as.factor(testDF[,outcomeName])))
# predict(gbmFit4, newdata = head(testing), type = "prob")





##############################################################################################
## Boosting -> Best subset: Importance of variables for the prediction of NumMosquitos
##############################################################################################
## Boosting
x = data.wnv[,1:23]
objControl = trainControl(returnResamp="none", 
                          summaryFunction=defaultSummary, 
                          selectionFunction="best",
                          #allowParallel=TRUE,
                          method="none" #, number=10, 
)

mod.boost = train(NumMosquitos ~.,
                  data=x,
                  method='gbm', 
                  trControl=objControl, 
                  verbose=FALSE,
                  tuneGrid=data.frame(interaction.depth=4,
                                      n.trees=100,
                                      shrinkage=0.001),
                  metric="RMSE")

## plot importnace
summary(mod.boost)


## regsubset
Number.Variables = 8
regfit.full = regsubsets(NumMosquitos ~ log.HotSpot + AvgSpeed.ma3 + DewPoint.ma1 + RH + Tavg.ma1 
                         + Month + daytime + PrecipTotal.ma3, data=x, nvmax=Number.Variables)
reg.summary = summary(regfit.full)
set.seed(1)
K = 5
block = sample(1:K, nrow(x), replace=TRUE, prob=rep(1/K, K))
cv.errors = matrix(0, K, Number.Variables)
for (i in 1:K) {
    best.full = regsubsets(NumMosquitos ~ log.HotSpot + AvgSpeed.ma3 + DewPoint.ma1 + RH + Tavg.ma1 
                           + Month + daytime + PrecipTotal.ma3, data=x[block!=i,], nvmax=Number.Variables)
    
    # compute CV-MSE
    for (j in 1:Number.Variables) {
        pred = predict.regsubsets(best.full, x[block==i,], id=j)
        cv.errors[i, j] = mse.fn(y=x$NumMosquitos[block==i], y_hat=pred)
    }
}

CV.MSE = apply(cv.errors, 2, mean)
fitscores = cbind(1-reg.summary$rsq,
                  1-reg.summary$adjr2,
                  reg.summary$cp,
                  reg.summary$bic,
                  CV.MSE)

colnames(fitscores) = c("1-rsq","1-adjR2","Cp","BIC", "CV-MSE")
rownames(fitscores) = paste0(apply(summary(best.full)$which, 1, sum)-1,"-Variable")
fitscores

adjR2.max = which.max(reg.summary$adjr2)
cp.min = which.min(reg.summary$cp)
bic.min = which.min(reg.summary$bic)
cv.mse.min = which.min(CV.MSE)
Model.Selection = cbind(adj.R2=adjR2.max, Cp=cp.min, BIC=bic.min, CV.MSE=cv.mse.min)
rownames(Model.Selection) = c("Best Model Selected")
Model.Selection

# Model 1
coef(regfit.full, id=adjR2.max)
# NumMosquitos ~ log.HotSpot+RH+Tavg.ma1+Month8+PrecipTotal.ma3 

# Model 2
coef(regfit.full, id=cp.min)
# NumMosquitos ~ log.HotSpot+RH+Tavg.ma1+Month8+PrecipTotal.ma3

# Model 3
coef(regfit.full, id=bic.min)
# NumMosquitos ~ log.HotSpot+Tavg.ma1+Month8

# Model 4
coef(best.full, id=cv.mse.min)
# NumMosquitos ~ log.HotSpot+Tavg.ma1+Month8


## Model 1: NumMosquitos ~ log.HotSpot+RH+Tavg.ma1+Month+PrecipTotal.ma3 
fit.formula = as.formula(NumMosquitos ~ log.HotSpot+RH+Tavg.ma1+Month+PrecipTotal.ma3)
mod1 = lm(fit.formula, data=x)
(mod1.avg.test.MSE = mean(Repeat.mse.fn(x, 100, Ycol=7, 0.7, 0.3)))

## Model 2: NumMosquitos ~ log.HotSpot+Tavg.ma1+Month 
fit.formula = as.formula(NumMosquitos ~ log.HotSpot+Tavg.ma1+Month)
mod2 = lm(fit.formula, data=x)
(mod2.avg.test.MSE = mean(Repeat.mse.fn(x, 100, Ycol=7, 0.7, 0.3)))

## Model 3: NumMosquitos ~ log.HotSpot+RH+Tavg.ma1
fit.formula = as.formula(NumMosquitos ~ log.HotSpot+Tavg.ma1)
mod3 = lm(fit.formula, data=x)
(mod3.avg.test.MSE = mean(Repeat.mse.fn(x, 100, Ycol=7, 0.7, 0.3)))


##############################################################################################
## Predict NumMosquitos using the equation in:
## Lebl et al. Parasites & Vectors 2013, 6:129
##############################################################################################

