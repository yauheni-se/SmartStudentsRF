library(dplyr)
library(caret)
library(tictoc)
library(pROC)
seed <- set.seed(123)

data_portugal <- data.table::fread("student-por.csv")

# EDA ----
glimpse(data_portugal)
any(unlist(lapply(data_portugal, anyNA), use.names = FALSE))
mean(data_portugal$G3 >= 15)

data_portugal <- data_portugal %>% 
  mutate(talented = factor(ifelse(G3 >= 15, "Talented", "Ordinary")))

cor(data_portugal[, 31:33])
data_portugal <- data_portugal %>%
  select(-G1, -G2, -G3)

index <- createDataPartition(data_portugal$talented, p = .75, list = FALSE)
train <- data_portugal[index,]
test <- data_portugal[-index]

# Trees, ROC, AUC, Lift ----
gridControl <- trainControl(method = "repeatedcv",
                            number = 5,
                            repeats = 3,
                            search = "grid",
                            summaryFunction = twoClassSummary,
                            classProbs = TRUE)

Grid <- expand.grid(mtry = seq(10,18, 2),
                    splitrule = c("gini", "extratrees", "hellinger"),
                    min.node.size = 1:6)

tic()
model_rf_grid <- train(talented ~.,
               data = train,
               method = "ranger",
               metric = "ROC",
               trControl = gridControl,
               tuneGrid = Grid,
               seed = seed)
toc()

model_rf_grid$bestTune

res_rf_grid <- model_rf_grid$results
res_rf_grid

plot(model_rf_grid)

predicted_grid <- predict(model_rf_grid, test)
confusionMatrix(predicted_grid, test$talented, positive = "Talented")


# ROC:
predicted_grid_probs <- predict(model_rf_grid, test, type = "prob")

rf_grid_roc <- roc(predictor = predicted_grid_probs$Talented,
                   response = test$talented,
                   levels = rev(levels(test$talented)))
plot(rf_grid_roc)

# AUC:
rf_grid_roc$auc

histogram(~predicted_grid_probs$Talented|test$talented, xlab = "Probability of Poor Segmentation RF Grid")


# Best Trees ----
Best_Grid <- expand.grid(mtry = model_rf_grid$finalModel$mtry,
                         splitrule = model_rf_grid$finalModel$splitrule,
                         min.node.size = model_rf_grid$finalModel$min.node.size)

ntrees <- seq(500,2500, 500)
models_rf_list_grid <- list()

tic()
for (ntree in ntrees) {
  model <- train(talented ~.,
                 data = train,
                 method = "ranger",
                 metric = "ROC",
                 trControl = gridControl,
                 tuneGrid = Best_Grid,
                 num.trees = ntree,
                 seed = seed)
  key <- toString(ntree)
  models_rf_list_grid[[key]] <- model
}
toc()

results_grid <- summary(resamples(models_rf_list_grid))
results_grid

Best_ntree_grid <- results_grid$statistics$ROC[, 4] %>% which.max %>% names %>% as.integer
Best_ntree_grid

tic()
model_rf_grid_best <- train(talented ~.,
                       data = train,
                       method = "ranger",
                       metric = "ROC",
                       trControl = gridControl,
                       tuneGrid = Best_Grid,
                       num.trees = Best_ntree_grid,
                       seed = seed)
toc()

test$predicted_grid <- predict(model_rf_grid_best, test)
confusionMatrix(test$predicted_grid, test$talented, positive = "Talented")

# ROC:
predicted_grid_probs_best <- predict(model_rf_grid_best, test, type = "prob")

rf_grid_roc_best <- roc(predictor = predicted_grid_probs_best$Talented,
                   response = test$talented,
                   levels = rev(levels(test$talented)))
plot(rf_grid_roc_best)

# AUC:
rf_grid_roc_best$auc

histogram(~predicted_grid_probs_best$Talented|test$talented, xlab = "Probability of Poor Segmentation RF Grid")


# Random Forests ----
randomControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  search = "random",
  summaryFunction = twoClassSummary,
  classProbs = TRUE)

model_rf_random <- train(talented ~.,
               data = train,
               method = "ranger",
               metric = "ROC",
               trControl = randomControl,
               tuneLength = 10,
               seed = seed)

random_Grid <- expand.grid(mtry = model_rf_random$finalModel$mtry,
                           splitrule = model_rf_random$finalModel$splitrule,
                           min.node.size = model_rf_random$finalModel$min.node.size)
model_rf_random$bestTune

res_rf_random <- model_rf_random$results
res_rf_random

plot(model_rf_random)
 
predicted_random <- predict(model_rf_random, test)
confusionMatrix(predicted_random, test$talented, positive = "Talented")

# ROC:
predicted_random_probs <- predict(model_rf_random, test, type = "prob")

rf_random_roc <- roc(predictor = predicted_random_probs$Talented,
                   response = test$talented,
                   levels = rev(levels(test$talented)))
plot(rf_random_roc)

# AUC:
rf_random_roc$auc

histogram(~predicted_random_probs$Talented|test$talented, xlab = "Probability of Poor Segmentation RF random")


# Best Random Forest ----
models_rf_list_random <- list()

tic()
for (ntree in ntrees) {
  model <- train(talented ~.,
                 data = train,
                 method = "ranger",
                 metric = "ROC",
                 trControl = gridControl,
                 tuneGrid = random_Grid,
                 num.trees = ntree,
                 seed = seed)
  key <- toString(ntree)
  models_rf_list_random[[key]] <- model
}
toc()

results_random <- summary(resamples(models_rf_list_random))
results_random

Best_ntree_random <- results_random$statistics$ROC[, 4] %>% which.max %>% names %>% as.integer
Best_ntree_random

tic()
model_rf_random_best <- train(talented ~.,
                            data = train,
                            method = "ranger",
                            metric = "ROC",
                            trControl = gridControl,
                            tuneGrid = random_Grid,
                            num.trees = Best_ntree_random,
                            seed = seed)
toc()

test$predicted_random <- predict(model_rf_random_best, test)
confusionMatrix(test$predicted_random, test$talented, positive = "Talented")

# ROC:
predicted_random_probs_best <- predict(model_rf_random_best, test, type = "prob")

rf_random_roc_best <- roc(predictor = predicted_random_probs_best$Talented,
                          response = test$talented,
                          levels = rev(levels(test$talented)))
plot(rf_random_roc_best)

# AUC:
rf_random_roc_best$auc

histogram(~predicted_random_probs_best$Talented|test$talented, xlab = "Probability of Poor Segmentation RF random")


# Test on new data ----
data_mat <- data.table::fread("student-mat.csv")
mean(data_mat$G3 >= 15)
data_mat <- data_mat %>% 
  mutate(talented = factor(ifelse(G3 >= 15, "Talented", "Ordinary")))

data_mat <- data_mat %>%
  select(-G1, -G2, -G3)

data_mat$predicted_grid <- predict(model_rf_grid_best, data_mat)
confusionMatrix(data_mat$predicted_grid, data_mat$talented, positive = "Talented")

data_mat$predicted_random <- predict(model_rf_random_best, data_mat)
confusionMatrix(data_mat$predicted_random, data_mat$talented, positive = "Talented")

rValues <- resamples(list(# rtree = model_rtree_best
                          rf_grid = model_rf_grid_best,
                          rf_random = model_rf_random_best))

rValues$values
summary(rValues)

bwplot(rValues, metric = "ROC", main = "Rtree vs RF grid vs RF Random")
dotplot(rValues, metric = "ROC", main = "Rtree vs RF grid vs RF Random")