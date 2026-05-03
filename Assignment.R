setwd("/users/ankitsakuja/desktop/R")

#load necessary libraries
library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)   
library(pROC)
library(grDevices)
library(gridExtra)
library(rpart)
library(rpart.plot)
library(ROCR)
library(MLmetrics)
library(grid)
library(ROSE)
library(PRROC)

# 1 Dataset Preprocessing

# 1.1 IMPORTING DATA TO R
data=read.csv("ict583 s2.csv")
data
#TO GET THE RECORDS OF FIRST FEW LINES OF DATA
head(data)

#VIEWING THE DATASET
View(data)

#SUMMARY STATISTICS
summary(data)

## 1.2 Data cleaning

# 1.2.1 Handling missing values
#The function “is.na” finds whether there are any missing values. 
sum(is.na(data))
#The zero in the output denotes that there were no missing values in the respective columns.

#1.2.2 Removing S.no from dataset as it doesn't seem to provide any meaningful information for our classification
data <- data %>% select(-S.no)
print(data)

# 1.2.3 Scaling
scale_columns <- c("Age", "body_height", "body_weight", "MNAa_tot", "MNAb_tot", "waist", "bmi")

# Apply scaling to the selected columns
scaled_data <- data %>%
  mutate(across(all_of(scale_columns), scale))

head(scaled_data)


## 2 EXPLORATORY DATA ANALYSIS

# TO KNOW THE STRUCTURE OF THE DATA
str(data)

## 2.1 Summary statistics for continuous variables
summary(data[, c("Age", "body_height", "body_weight", "MNAa_tot", "MNAb_tot", "waist", "bmi")])

## 2.2 Histograms for continuous variables
## List of continuous variable names 
continuous_variables <- c("Age", "body_height", "body_weight", "MNAa_tot", "MNAb_tot", "waist", "bmi")
plot_list_Conti=list()
for (var in continuous_variables) {
  hist_plot <- ggplot(data, aes_string(x = var)) +
    geom_histogram(fill = "pink",alpha=0.7,bins = 30) 
    ggtitle(paste("Histogram of", var) )
  plot_list_Conti[[length(plot_list_Conti)+1]] = hist_plot
}
do.call(grid.arrange, c(plot_list_Conti,ncol=2))

## 2.3 Boxplots of continuous variables
par(mfrow = c(2, 4))  

for (var in continuous_variables) {
  boxplot(data[[var]], main = var)
}

## 2.4 Transforming and Tabulating the categorical variables
## Since in str(data) we found that categorical variables are listed as 'num' instead of 'factor',because they are representing a category, we can transform them into factor by following code

## EducationID
ed=data$Education_ID
educ=factor(ed,levels=c(1,2,3,4),labels=c("No formal education","Elementary school level","High school level","College level or above"))
View(table(educ))

## mobility
mob=data$Mobility
mobility=factor(mob,levels=c(1,2,3,4),labels=c("Wheelchair","Walker","Cane","Self-mobility"))
View(table(mobility))

## Hyperlipidaemia
hyp1=data$Endocrine.Disease_Hyperlipidaemia
hyp=factor(hyp1,levels=c(0,1),labels=c("NO","Yes"))
View(table(hyp))

## Risk of dementia(MSME_CLASS2)
mm=data$MMSE_class_2
dimentia=factor(mm,levels=c(0,1),labels=c("No risk","At risk"))
View(table(dimentia))

## MNAa_q3
n=data$MNAa_q3
activity=factor(n,levels=c(0,1,2),labels=c("Bedridden","can't go out","can go out"))
View(table(activity))

## barplot of Education levels
barplot(table(educ),xlab="Categories",ylab="No. of people",main="Education-levels",col=c("red","green","blue","yellow"))

## Barplot of Mobility Data
barplot(table(mobility),xlab="Mobility-level",ylab="No. of people",main="Mobility data",col=c("blue","yellow","orange","violet"))

## barplot of Hyperlipidamia
barplot(table(hyp),xlab="Categories",ylab="No. of people",main="Hyperlipidaemia",col=c("green","blue"))

## Barplot Risk of dementia
barplot(table(dimentia),xlab="Categories",ylab="No. of people",main="Risk of dementia",col=c("blue","yellow"))

## Activity ability
barplot(table(activity),xlab="Categories",ylab="No. of people",main="Activity ability ",col =c('green','red','blue'))

## 2.5 Correlation matrix of continous variables
pdf("CM.pdf", width = 8, height = 8)  
correlation_matrix <- cor(data[, c("Age", "body_height", "body_weight", "MNAa_tot", "MNAb_tot", "waist", "bmi")])
par(mar = c(5, 5, 4, 2))
corrplot(correlation_matrix, method = "circle", type = "full", tl.col = "darkblue", 
         title = "Correlation matrix plot", mar = c(1, 1, 1, 1))
dev.off()


## 2.6 Categorising and plotting the results of the MNAa_tot test

data$nutriton_level = ifelse(data$MNAa_tot < 8, "Malnourished",
                             ifelse(data$MNAa_tot >=12, "Normal", "At_risk_of_malnutrition"))

nutrition=factor(data$nutriton_level,levels=c( "Malnourished","Normal", "At_risk_of_malnutrition"),labels=c( "Malnourished","Normal", "At_risk_of_malnutrition"))

View(table(nutrition))

barplot(table(nutrition),xlab="categories",ylab="no. of prople",main="Nutrition-level assessment",col=c("red","lightgreen","yellow"))




### 3 PREDICTIVE MODELLING
### 3.1 LOGISTIC REGRESSION

### 3.1.1 Balance the data 
data_balanced <- ovun.sample(MMSE_class_2 ~ ., data = data, method = "both", p = 0.5)$data


### 3.1.2 Data Spilitting
# Split the balanced dataset into training and test sets
set.seed(123)  
split_indices <- createDataPartition(data_balanced$MMSE_class_2, p = 0.5, list = FALSE)
train_data <- data_balanced[split_indices, ]
test_data <- data_balanced[-split_indices, ]

### 3.1.3 Preprocess -scale the continuous variables 
scale_variables <- c("Age", "body_height", "body_weight", "MNAa_tot", "MNAb_tot", "waist", "bmi")

train_data[scale_variables] <- scale(train_data[scale_variables])
test_data[scale_variables] <- scale(test_data[scale_variables])

### 3.1.4 Logistic Regression Model
logistic_model <- glm(MMSE_class_2 ~ ., data = train_data, family = binomial)

###3.1.5 Coefficient 
coefficients_logistic <- coef(logistic_model)
print(coefficients_logistic)

### 3.1.6 predicting probabilities and classes
predictions_logistic <- predict(logistic_model, newdata = test_data, type = "response")
predictions_logi <- ifelse(predictions_logistic > 0.5, 1, 0)

### 3.1.7 Evaluate Logistic Regression Model
log_cm <- confusionMatrix(as.factor(predictions_logi),as.factor(test_data$MMSE_class_2))
cat("Confusion matrix for Logistic Regression:\n")
print(log_cm)

### 3.1.8 Calculate confusion matrix for Logistic Regression
log_cm <- confusionMatrix(as.factor(predictions_logi), as.factor(test_data$MMSE_class_2))

### 3.1.9 Extracting accuracy, precision, recall, and F1 score
accuracy_logistic <- log_cm$overall["Accuracy"]
precision_logistic <- log_cm$byClass["Pos Pred Value"]
recall_logistic <- log_cm$byClass["Sensitivity"]
f1_score_logistic <- log_cm$byClass["F1"]

### 3.1.10 Calculate AUC for Logistic Regression
roc_logistic <- roc(test_data$MMSE_class_2, as.numeric(predictions_logistic))
auc_logistic <- auc(roc_logistic)

### 3.1.11 Display the results
cat("Logistic Regression - Accuracy:", accuracy_logistic, "\n")
cat("Logistic Regression - Precision:", precision_logistic, "\n")
cat("Logistic Regression - Recall:", recall_logistic, "\n")
cat("Logistic Regression - F1 Score:", f1_score_logistic, "\n")
cat("Logistic Regression - AUC:", auc_logistic, "\n")


### 3.1.12 Plot ROC Curve
plot(roc_logistic, col = "blue", main = "ROC Curve")

### 3.1.13 Plot Precision-Recall Curve
plot(recall_values, precision_values, type = "l", col = "red", xlab = "Recall", ylab = "Precision", 
     main = "Precision-Recall Curve")







### 3.2 Decision Tree Modelling

### 3.2.1 Balance the data 
data_balanced <- ovun.sample(MMSE_class_2 ~ ., data = data, method = "both", p = 0.5)$data

# Split the balanced dataset into training and test sets
set.seed(123)  
split_indices <- createDataPartition(data_balanced$MMSE_class_2, p = 0.5, list = FALSE)
train_data <- data_balanced[split_indices, ]
test_data <- data_balanced[-split_indices, ]

### 3.2.2 Decision tree model
decision_tree_model <- rpart(MMSE_class_2 ~ ., data = train_data, method = "class")

### 3.2.3 Make predictions on the test data
predictions_decision_tree <- predict(decision_tree_model, newdata = test_data, type = "class")

### 3.2.4 Evaluate the Decision Tree Model
confusion_matrix_decision_tree <- table(Actual = test_data$MMSE_class_2, Predicted = predictions_decision_tree)
accuracy_decision_tree <- sum(diag(confusion_matrix_decision_tree)) / sum(confusion_matrix_decision_tree)

### 3.2.5 Calculate Precision and Recall
tp_dt <- sum(predictions_decision_tree == 1 & test_data$MMSE_class_2 == 1)
fp_dt <- sum(predictions_decision_tree == 1 & test_data$MMSE_class_2 == 0)
fn_dt <- sum(predictions_decision_tree == 0 & test_data$MMSE_class_2 == 1)

precision_decision_tree <- tp_dt / (tp_dt + fp_dt)
recall_decision_tree <- tp_dt / (tp_dt + fn_dt)

### 3.2.6 Calculate F1 score for Decision Tree using MLmetrics package
f1_score_decision_tree <- F1_Score(test_data$MMSE_class_2, predictions_decision_tree)

### 3.2.7 Calculate AUC for Decision Tree
roc_decision_tree <- prediction(as.numeric(predictions_decision_tree), as.numeric(test_data$MMSE_class_2))
roc_perf <- performance(roc_decision_tree, measure = "tpr", x.measure = "fpr")
plot(roc_perf, main="ROC Curve for Decision Tree")


auc_decision_tree <- as.numeric(performance(roc_decision_tree, "auc")@y.values)

### 3.2.8 Plot the Decision Tree
rpart.plot(decision_tree_model)

### 3.2.9 Display results for Decision Tree
cat("Decision Tree - Accuracy:", accuracy_decision_tree, "\n")
cat("Decision Tree - Precision:", precision_decision_tree, "\n")
cat("Decision Tree - Recall:", recall_decision_tree, "\n")
cat("Decision Tree - F1 Score:", f1_score_decision_tree, "\n")
cat("Decision Tree - AUC:", auc_decision_tree, "\n")




#### 4. Plotting both the models ROC curve together
roc_logistic <- roc(test_data$MMSE_class_2, predictions_logistic)
roc_decision_tree <- roc(test_data$MMSE_class_2, as.numeric(predictions_decision_tree))
#### 4.1 Plot the ROC curves
plot(roc_logistic, col = "blue", main = "ROC Curves")
plot(roc_decision_tree, col = "red", add = TRUE)  # Add the Decision Tree ROC curve
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lwd = 2)








































  