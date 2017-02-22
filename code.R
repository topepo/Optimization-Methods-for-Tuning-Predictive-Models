###################################################################
## Code from RUG talk "Optimization Methods for Tuning Predictive 
## Models" by Max Kuhn


library(caret)
data(Sacramento)
str(Sacramento)

## Create split of data

set.seed(955)
in_train <- createDataPartition(log10(Sacramento$price), p = .8, list = FALSE)
head(in_train)
training <- Sacramento[ in_train,]
testing  <- Sacramento[-in_train,]

library(ggmap)

ca_map <- qmap("folsom ca",
               color = "bw",
               legend = "topleft",
               darken = 0, zoom = 10)

ca_map +
  geom_point(data = training, aes(x = longitude, y = latitude), col = "red",  alpha = .5, size = 1) +
  geom_point(data = testing,  aes(x = longitude, y = latitude), col = "blue", alpha = .5, size = 1)

## Conduct a simple grid search over sigma and cost

## These fold splits will be used in several functions so save them
set.seed(3313)
index <- createFolds(training$price, returnTrain = TRUE, list = TRUE)
ctrl <- trainControl(method = "cv", index = index)

## for parallelization: on linux and OS X I recommend doMC and
## on windows doParallel

# library(doMC)
# registerDoMC(cores=7)

set.seed(30218) 
grid_search <- train(log10(price) ~ ., data = training,
                     method = "svmRadialSigma",
                     ## Will create 48 parameter combinations
                     tuneLength = 8,
                     metric = "RMSE",
                     preProc = c("center", "scale", "zv"),
                     trControl = ctrl)
getTrainPerf(grid_search)

ggplot(grid_search) + scale_x_log10() + theme(legend.position = "top")


## Create the objective function for optimization calls. Takes the 2 parameters
## as inputs and `maximize` is used because different function have different
## objective function formats. 

svm_obj <- function(param, maximize = FALSE) {
  mod <- train(log10(price) ~ ., data = training,
               method = "svmRadial",
               preProc = c("center", "scale", "zv"),
               metric = "RMSE",
               trControl = ctrl,
               tuneGrid = data.frame(C = 10^(param[1]), sigma = 10^(param[2])))
  if(maximize)
    -getTrainPerf(mod)[, "TrainRMSE"] else
      getTrainPerf(mod)[, "TrainRMSE"]
}

## We will standardize to always evaluate about the same number of 
## models (this can get sliced base don generations, iterations etc. )
num_mods <- 100

## Simulated annealing from base R
set.seed(45642)
san_res <- optim(par = c(0, 0), fn = svm_obj, method = "SANN",
                 control = list(maxit = num_mods))
san_res

## Nelder-Mead with Base R
set.seed(45642)
nm_res <- optim(par = c(0, 0), fn = svm_obj, method = "Nelder-Mead",
                control = list(maxit = num_mods))
nm_res

## Partical Swarm from pso
library(pso)
set.seed(45642)
pso_res <- psoptim(par = c(0, 0), fn = svm_obj,
                   lower = c(-5, -5), upper = c(5, 5),
                   control = list(maxit = ceiling(num_mods/12)))
pso_res

## Genetic algorithm
library(GA)
set.seed(45642)
ga_res <- ga(type = "real-valued",
             fitness = svm_obj,
             min = c(-5, -5), max = c(5, 5),
             maxiter = ceiling(num_mods/50),
             maximize = TRUE)
ga_res@solution

## Bayesian optimization search

## Use this function to optimize the model. The two parameters are
## evaluated on the log scale given their range and scope. A different
## function is used here because of how the `BayesianOptimization`
## function wants the output

svm_fit_bayes <- function(logC, logSigma) {
  ## Use the same model code but for a single (C, sigma) pair.
  mod <- train(log10(price) ~ ., data = training,
               method = "svmRadial",
               preProc = c("center", "scale", "zv"),
               metric = "RMSE",
               trControl = ctrl,
               tuneGrid = data.frame(C = 10^(logC), sigma = 10^(logSigma)))

  ## The function wants to _maximize_ the outcome so we return
  ## the negative of the resampled RMSE value. `Pred` can be used
  ## to return predicted values but we'll avoid that and use NULL
  list(Score = -getTrainPerf(mod)[, "TrainRMSE"], Pred = 0)
}

library(rBayesianOptimization)

bounds <- list(logC = c(-2,  5), logSigma = c(-7, -2))

set.seed(8606)
bo_search <- BayesianOptimization(svm_fit_bayes,
                                  bounds = bounds,
                                  init_points = 10,
                                  n_iter = 100,
                                  acq = "ucb",
                                  kappa = 1,
                                  eps = 0.0)
bo_search


rs_res <- rbind(c(san_res$par, san_res$value),
                c(nm_res$par, nm_res$value), 
                c(pso_res$par, pso_res$value), 
                c(ga_res@solution, -ga_res@fitnessValue), 
                c(bo_search$Best_Par, -bo_search$Best_Value))
rownames(rs_res) <- c("SA", "NM", "PSO", "GA", "BO")
colnames(rs_res)[3] <- "RMSE"
rs_res[order(rs_res[,3]),]


## create models based on optimization results to use
## on the test set

set.seed(30218) 
nm_mod <- train(log10(price) ~ ., data = training,
                method = "svmRadialSigma",
                tuneGrid = data.frame(C = 10^nm_res$par[1], 
                                      sigma = 10^nm_res$par[2]),
                metric = "RMSE", 
                trControl = ctrl, 
                preProc = c("center", "scale", "zv"))
postResample(predict(nm_mod, testing), log10(testing$price))

set.seed(30218) 
bo_mod <- train(log10(price) ~ ., data = training,
                method = "svmRadialSigma",
                tuneGrid = data.frame(C = 10^bo_search$Best_Par[1], 
                                      sigma = 10^bo_search$Best_Par[2]),
                metric = "RMSE", 
                trControl = ctrl, 
                preProc = c("center", "scale", "zv"))
postResample(predict(bo_mod, testing), log10(testing$price))

