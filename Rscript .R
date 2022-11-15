### install package
install.packages("VIM")
install.packages("glmnet")
install.packages("pROC")
install.packages("ROCR")
install.packages("writexl")

### source functions
source("DataAnalyticsFunctions.R")
source("PerformanceCurves.R")

### some packages
library(tree)
library(partykit)
library(randomForest)

options(warn=-1)

############################
set.seed(1)
library(readxl)
rawdata = read_excel("marketing_campaign.xlsx")
summary(rawdata)
ncol(rawdata)
nrow(rawdata)
sum(complete.cases(rawdata))
paste("Missing values in ",nrow(rawdata)-sum(complete.cases(rawdata)), "observations out of ",nrow(rawdata))
###



#############################
library(VIM)
### visualization
aggr(rawdata, cex.axis=0.5)
aggr(rawdata,combined = TRUE, numbers=TRUE, cex.numbers=0.5)
sum(is.na(rawdata$Income))
# Having 23 missing value in income 
#remove 2 outliers in age 
rawdata=rawdata[rawdata$Year_Birth>1940,]

### since only a few missing we drop them
data<- rawdata[complete.cases(rawdata),]
### drop ID 
data <- data[,-1]

### make some variables factors
data$Education = factor(data$Education)
data$Marital_Status = factor(data$Marital_Status)

#drop outliers from income 
data= data[data$Income<100000,]

### check data type
str(data)

### change Dt_Customer to Date
data$Dt_Customer = as.Date(data$Dt_Customer,"%Y-%m-%d")
class(data$Dt_Customer)

### drop columns that are the same
unique(data$Z_CostContact)
unique(data$Z_Revenue)
library(dplyr)
data = data %>%
  select(-Z_CostContact,-Z_Revenue)

#calculate age
data$age= 2014- data$Year_Birth
data=data[,-1]

#calculate years of membership
data$memdate= as.numeric(format(data$Dt_Customer, format="%Y" ))
data$memdate= 2014-data$memdate

data= data %>% select(-Dt_Customer )

#write.csv(data,"data.csv")

##################################
data$completed= data$AcceptedCmp1 + data$AcceptedCmp2 + data$AcceptedCmp3 + data$AcceptedCmp4 + data$AcceptedCmp5
data$Monetary = data$MntWines+data$MntFruits+data$MntMeatProducts+data$MntFishProducts+data$MntSweetProducts+data$MntGoldProds
data$Monetary_Food=data$MntWines+data$MntFruits+data$MntMeatProducts+data$MntFishProducts+data$MntSweetProducts
data$Frequency = data$NumDealsPurchases+data$NumWebPurchases+data$NumCatalogPurchases+data$NumStorePurchases

################################
### Create a final holdout sample
RMF = data
set.seed(1)
holdout.indices <- sample(nrow(RMF), 400)
test <- RMF[holdout.indices,]
train <- RMF[-holdout.indices,]
nrow(test)
nrow(train)

### check balance
mean(train$Response==1) #15%
mean(test$Response==1) #12%


##################### EDA #############################################################
### see correlations
library(ggplot2)
library(GGally)
ggpairs(train[,c(1,2,3,4,5,6,26)])
ggpairs(test[,c(1,2,3,4,5,6,26)])
ggpairs(data[,c(20,21,22,23,24,25,26)])

library(corrplot)
varstolook <- c(  "age",  "Income",              "Kidhome",            
                 "Teenhome",              "Recency",             "MntWines",            "MntFruits" ,         
                 "MntMeatProducts",     "MntFishProducts",     "MntSweetProducts",    "MntGoldProds",        "NumDealsPurchases",  
                 "NumWebPurchases",     "NumCatalogPurchases", "NumStorePurchases",   "NumWebVisitsMonth",   "AcceptedCmp3" ,      
                 "AcceptedCmp4",        "AcceptedCmp5",        "AcceptedCmp1",        "AcceptedCmp2",        "Complain" ,          
                 "Response" )
CorMatrix <- cor(data[,varstolook])
corrplot(CorMatrix, method = "square")


###Age
counts<-table(data$age)
table(data$age)
barplot(counts, main="Age",xlab="age")

##Boxplot
age<- data$age
boxplot(age, main="Age")
summary(data$age)
sd(data$age)

##Education

Graduation <- sum(data$Education =="Graduation") 
PhD <- sum(data$Education =="PhD")
Master <- sum(data$Education =="Master")
Basic <- sum(data$Education =="Basic")
n_Cycle <- sum(data$Education =="2n Cycle") 
Education <- c(Graduation, PhD, Master, Basic, n_Cycle)
barplot(Education, main="Education", names.arg = c ("Graduation","PhD","Master","Basic","2n Cycle"),xlab="Education Level")

##Marital status
M <- sum(data$Marital_Status =="Married") 
Tog <- sum(data$Marital_Status =="Together")
Single <- sum(data$Marital_Status =="Single")
D <- sum(data$Marital_Status =="Divorced")
W <- sum(data$Marital_Status =="Widow")
A <- sum(data$Marital_Status =="Alone")
Ab <- sum(data$Marital_Status =="Absurd")
Y <- sum(data$Marital_Status =="YOLO")
Marital <- c(M, Tog, Single, D, W, A, Ab, Y)
barplot(Marital, main="Marital Status", names.arg= c("Married","Together","Single","Divorced","Widow","Alone","Absurd","YOLO"), xlab="Marital Status")

##Income
hist(data$Income, breaks = 10, ylim=c(0,400), xlim=c(0,100000), main='Income')
summary(data$Income)
sd(data$Income)


##Kids home pie
sum(data$Kidhome=='0')
sum(data$Kidhome=='1')
sum(data$Kidhome=='2')

table = data.frame(table(data$Kidhome))
x <- c(1270,883,46)
labels<- c("0 kid","1 kid", "2 kids")
piepercent<- round(100*x/sum(x),1)
pie(x, labels=piepercent, main ="Kidhome", col=rainbow(length(x)))
legend("topright",c("0 kid","1 kid","2 kids"),cex=0.8, fill=rainbow(length(x)))

##barplot
counts<-table(data$Kidhome)
table(data$Kidhome)
barplot(counts, main="#Kids in home",xlab="#kids")

##Teen homepie
sum(data$Teenhome=='0')
sum(data$Teenhome=='1')
sum(data$Teenhome=='2')

table = data.frame(table(data$Teenhome))
x <- c(1133,1015,51)
labels<- c("0 kid","1 kid", "2 kids")
piepercent<- round(100*x/sum(x),1)
pie(x, labels=piepercent, main ="Teenhome", col=rainbow(length(x)))
legend("topright",c("0 teen","1 teen","2 teens"),cex=0.8, fill=rainbow(length(x)))

##barplot
counts<-table(data$Teenhome)
table(data$Teenhome)
barplot(counts, main="#Teenagers in home",xlab="#teens")


########################### Customer Lifetime Value ########################################################
RMF0=sqldf("Select *
           FROM RMF
           WHERE Frequency >0 and Recency >0")
sum(RMF0$Frequency==0)
sum(RMF0$Recency==0)
retention_rate <- ((730-RMF0$Recency)/730)^RMF0$Frequency
CLV <- data.frame(retention_rate, RMF0)
##assume our gross margin is 40%
M <- (RMF0$Monetary/RMF0$Frequency)*0.4
CLV <- data.frame(M,CLV)
##discount rate 10%
clv<- (CLV$retention_rate/(1+0.1-CLV$retention_rate))*CLV$M
CLV <- data.frame(clv,CLV)
summary(CLV$clv)
##CLV>10(28.5% of population) takes 80% of the total CLV
sum(CLV$clv)
library(magrittr)
library(tidyr)
temp <- CLV %>% filter (CLV$clv >= 10)
temp
619/2169
sum(temp$clv)
23638.38/29458.39
##clv>1&<10
temp1 <- CLV %>% filter (CLV$clv > 1 &CLV$clv<10)
temp1
sum(temp1$clv)
1369/2169
sum(temp1$clv)
5719.759/29458.39


################################################### RFM analysis ####################################################

RMF = data
RMF$Monetary = data$MntWines+data$MntFruits+data$MntMeatProducts+data$MntFishProducts+
  data$MntSweetProducts+data$MntGoldProds
RMF$Frequency = data$NumDealsPurchases+data$NumWebPurchases+data$NumCatalogPurchases
+data$NumStorePurchases
set.seed(1)
# write.csv(RMF,'RMF.csv')

RMF_k= RMF
RMF_k$R = RMF_k$Recency

RMF_k = RMF_k %>%
  select(-Recency)
colnames(RMF_k)[28] = "Recency"


RMF_k$Monetary = as.numeric(RMF_k$Monetary)
RMF_k$Frequency = as.numeric(RMF_k$Frequency)
RMF_k$Recency = as.numeric(RMF_k$Recency)

## RMF
###############################################
RMF_1 = data.frame(cbind(RMF$Recency,RMF %>% select("Response","Monetary","Frequency")))
colnames(RMF_1)[1] = "Recency"
summary(RMF_1)
RMF_1$R = ifelse(RMF_1$Recency>24,ifelse(RMF_1$Recency>49,ifelse(RMF_1$Recency>74,4,3),2),1)
RMF_1$M = ifelse(RMF_1$Monetary>68,ifelse(RMF_1$Monetary>396.5,ifelse(RMF_1$Monetary>1033,4,3),2),1)
RMF_1$F = ifelse(RMF_1$Frequency>8,ifelse(RMF_1$Frequency>14.88,ifelse(RMF_1$Monetary>21,4,3),2),1)
data_RMF = sqldf("SELECT R,M,F,avg(Response)*100
                 FROM RMF_1
                 GROUP BY R,M,F")
colnames(data_RMF)[4] = "Response Rate (%)"
View(data_RMF)
# write.csv(data_RMF,"RMF2.csv")

data_RMF$RFMscore = data_RMF$R*100+data_RMF$F*10+data_RMF$M

ggplot(data=data_RMF,aes(x=reorder(RFMscore,-`Response Rate (%)`),y=`Response Rate (%)`,fill="RFMscore"))+geom_bar(stat="identity")


################################################# Supervised Learning- logistic Regression ############################3

####################################################################################
###################################################################################
### Model #########################################################################

err_metric=function(CM)
{
  TN =CM[1,1]
  TP =CM[2,2]
  FP =CM[1,2]
  FN =CM[2,1]
  precision =(TP)/(TP+FP)
  recall_score =(FP)/(FP+TN)
  f1_score=2*((precision*recall_score)/(precision+recall_score))
  accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
  False_positive_rate =(FP)/(FP+TN)
  False_negative_rate =(FN)/(FN+TP)
  True_positive_rate=(TP)/(TP+FN)
  print(paste("Precision value of the model: ",round(precision,2)))
  print(paste("Accuracy of the model: ",round(accuracy_model,2)))
  print(paste("Recall value of the model: ",round(recall_score,2)))
  print(paste("False Positive rate of the model: ",round(False_positive_rate,2)))
  print(paste("False Negative rate of the model: ",round(False_negative_rate,2)))
  print(paste("True Positive rate of the model: ",round(True_positive_rate,2)))
  print(paste("f1 score of the model: ",round(f1_score,2)))
}

PerformanceMeasure <- function(actual, prediction, threshold=.5) {
  #1-mean( abs( (prediction>threshold) - actual ) )  
  #R2(y=actual, pred=prediction, family="binomial")
  1-mean( abs( (prediction- actual) ) )  
}
PerformanceMeasure_RMSE <- function(actual, prediction, threshold=.5) {
  #1-mean( abs( (prediction>threshold) - actual ) )  
  sqrt(mean((actual-prediction)^2))
}


library(pROC)
library(ROCR)
library(lmtest)
library(tree)

###  model_1
colnames(train)
model_1 <-glm(Response~. -MntWines -MntFruits -MntMeatProducts -MntFishProducts -MntSweetProducts -MntGoldProds -NumDealsPurchases -NumWebPurchases -NumCatalogPurchases -NumStorePurchases- NumWebVisitsMonth -Monetary - Frequency - completed -Monetary_Food, data=train, family="binomial")
summary(model_1)
R2_1 <- 1-model_1$deviance/model_1$null.deviance
R2_1
lrtest(model_1)

pred_1 <- predict(model_1, newdata=train, type="response")
roccurve <-  roc(p=pred_1, y=train$Response, bty="n")


## TPR, FPR
logic_1 = ifelse(pred_1>0.5,1,0)
CM_1 = table(train$Response,logic_1)
print(CM_1)
err_metric(CM_1)

### ROC, AUC
pred_1 = prediction(predictions=pred_1,labels=train$Response)
roc_1 = performance(pred_1,"tpr","fpr")
auc_1 = performance(pred_1,measure="auc")@y.values[[1]]
plot(roc_1,colorize=F, main="Model 1 ROC")
lines(c(0,1),c(0,1), lty=2)

auc_1

### test on test data
pred_t1 = predict(model_1, newdata=test, type="response")
logic_t1 = ifelse(pred_t1>0.5,1,0)
CM_t1 = table(test$Response,logic_t1)
print(CM_t1)
err_metric(CM_t1)


### Model_2

colnames(train)
model_2 <-glm(Response~. -MntWines -MntFruits -MntMeatProducts -MntFishProducts -MntSweetProducts -MntGoldProds -NumDealsPurchases -NumWebPurchases -NumCatalogPurchases -NumStorePurchases- NumWebVisitsMonth - completed -Monetary_Food , data=train, family="binomial")
summary(model_2)
R2_2 <- 1-model_2$deviance/model_2$null.deviance
R2_2
lrtest(model_2)

pred_2 <- predict(model_2, newdata=train, type="response")
roccurve <-  roc(p=pred_2, y=train$Response, bty="n")

## TPR, FPR
logic_2 = ifelse(pred_2>0.5,1,0)
CM_2 = table(train$Response,logic_2)
print(CM_2)
err_metric(CM_2)

### ROC, AUC
pred_2 = prediction(predictions=pred_2,labels=train$Response)
roc_2 = performance(pred_2,"tpr","fpr")
auc_2 = performance(pred_2,measure="auc")@y.values[[1]]
plot(roc_2,colorize=F)
auc_2

### test on test data
pred_t2 = predict(model_2, newdata=test, type="response")
logic_t2 = ifelse(pred_t2>0.5,1,0)
CM_t2 = table(test$Response,logic_t2)
print(CM_t2)
err_metric(CM_t2)


#### model_3
colnames(train)
model_3 <-glm(Response~Income+MntWines+MntMeatProducts+MntFishProducts+NumCatalogPurchases+AcceptedCmp3+AcceptedCmp4+AcceptedCmp5+AcceptedCmp2+AcceptedCmp1, data=train, family="binomial")
summary(model_3)
R2_3 <- 1-model_3$deviance/model_3$null.deviance
R2_3
lrtest(model_3)

pred_3 <- predict(model_3, newdata=train, type="response")
roccurve <-  roc(p=pred_3, y=train$Response, bty="n")


## TPR, FPR
logic_3 = ifelse(pred_3>0.5,1,0)
CM_3 = table(train$Response,logic_3)
print(CM_3)
err_metric(CM_3)

### ROC, AUC
pred_3 = prediction(predictions=pred_3,labels=train$Response)
roc_3 = performance(pred_3,"tpr","fpr")
auc_3 = performance(pred_3,measure="auc")@y.values[[1]]
plot(roc_3,colorize=F)
auc_3

### test on test data
pred_t3 = predict(model_3, newdata=test, type="response")
logic_t3 = ifelse(pred_t3>0.5,1,0)
CM_t3 = table(test$Response,logic_t3)
print(CM_t3)
err_metric(CM_t3)

#### THRESHOLD DISCUSSION 

index <- c(50)
radius <- 0.009 *rep(1,length(index))
color <- c("red")
symbols(roccurve[index ,], circles=radius, inches = FALSE,ylim=c(0,1), xlim=c(0,1), ylab="True positive rate", xlab="False positive rate", bg=color)
Accuracy(pred_2>=0.5 , train$Response)


index <- c(40,50)
radius <- 0.009 *rep(1,length(index))
color <- c("blue","red")
symbols(roccurve[index ,], circles=radius, inches = FALSE,ylim=c(0,1), xlim=c(0,1), ylab="True positive rate", xlab="False positive rate", bg=color)
Accuracy(pred_2>=0.4, train$Response)

index <- c(75,40,50)
color <- c("black","blue","red")
radius <- 0.009 *rep(1,length(index))
symbols(roccurve[index ,], circles=radius, inches = FALSE,ylim=c(0,1), xlim=c(0,1), ylab="True positive rate", xlab="False positive rate", bg=color)
Accuracy(pred_2>=0.75 , train$Response)



