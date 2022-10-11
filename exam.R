library(geepack)
library(rpart)
library(randomForest)
library(dplyr)
library(ggplot2)
library(caret)
library(ROCR)
library(rpart.plot)

# 1 ----
is.prime <- function(num) {
  if (num %in% c(1,2)) {
    TRUE
  } else if (any(num %% 2:(num-1) == 0)) {
    FALSE
  } else { 
    TRUE
  }
}

twins <- function(num1, num2){
  if(abs(num1 - num2) == 2){
    if(is.prime(num1) & is.prime(num2)){
      return(TRUE)
    }
  }
  return(FALSE)
}

twins(199, 197)
twins(2, 5)
twins(4, 6)


# 2 ---- 
data("GermanCredit")

GermanCredit_age_stat <- GermanCredit %>% group_by(ResidenceDuration) %>%
                                          summarise(min = min(Age),
                                                    srednia = mean(Age),
                                                    mediana = median(Age),
                                                    max = max(Age)) %>%
                                          arrange(desc(ResidenceDuration))
GermanCredit_age_stat

ggplot(GermanCredit, aes(y = Age)) +
  geom_boxplot() +
  facet_wrap(~ResidenceDuration)


# 3 ----
data("seizure")

### a 
seizure$trt <- as.factor(seizure$trt)
set.seed(82591)
test_proportion <- 0.30
index <- (runif(nrow(seizure)) < test_proportion)
seizure_test <- seizure[index, ]
seizure_train <- seizure[-index, ]

### b
tree_model <- rpart(trt ~ .,
              data=seizure_train,
              method="class",
              control = list(maxdepth = 4))

rf_model <- randomForest(trt ~., 
                         data = seizure_train,
                         ntree = 300)

### c
varImpPlot(rf_model)

### d
confusion_matrices <- list()
confusion_matrices[["rtree"]] <- table(predict(tree_model, seizure_test, type = "class"), seizure_test$trt)
confusion_matrices[["rforest"]] <- table(predict(rf_model, seizure_test, type = "class"), seizure_test$trt)

evaluate_models <- function(cm) {
  TP <- cm[1,1]
  TN <- cm[2,2]
  Cond_P <- sum(cm[ ,1])
  Pred_P <- sum(cm[1, ])
  precision <- TP / Pred_P
  SE <- TP / Cond_P
  NPV <- cm[2, 2] / (cm[2, 1] + cm[2, 2])
  F1 <- (2 * precision * SE) / (precision + SE)
  return(list(F1 = F1,
              NPV = NPV))
}

evaluate_models(confusion_matrices[["rtree"]])
evaluate_models(confusion_matrices[["rforest"]])

### e
predicted <- predict(rf_model, seizure_test, type = "prob")[, 2]
plotting <- prediction(predicted, seizure_test$trt)

performance(plotting,"auc")@y.values[[1]]

plot(performance(plotting,"tpr","fpr"),lwd=2, colorize=TRUE)


# 4 ----
data("esoph")

### a
set.seed(82591)
test_pr <- 0.30
ind <- (runif(nrow(esoph)) < test_pr)
esoph_test <- esoph[ind, ]
esoph_train <- esoph[-ind, ]

### b
lm_model <- lm(ncases ~ ., data = esoph_train)
tree_regr_model <- rpart(ncases ~., data = esoph_train, cp = 0.08)
plot

### c
rpart.plot(tree_regr_model, under=FALSE, fallen.leaves = FALSE, cex = 0.9)

### d
models <- list("tree_regr_model" = tree_regr_model,
               "lm_model" = lm_model)

evaluate_models_2 <- function(models, data, predicted) {
  RMSE <- sapply(models, function(x) sqrt(sum((data[[predicted]] - predict(x, data))^2)/nrow(data)))
  RAE <- sapply(models, function(x) sum(abs((data[[predicted]] - predict(x, data))))/sum(abs(data[[predicted]] - mean(data[[predicted]]))))
  return (list(RMSE = RMSE,
               RAE = RAE))
}

evaluate_models_2(models, esoph_test, 'ncases')

### e
# linear model is better