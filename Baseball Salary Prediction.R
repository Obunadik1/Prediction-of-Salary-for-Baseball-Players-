---
  title: "Using Machine Learnig Algorithm to Predict the Salary of Baseball Players "
author: "Obunadike Callistus"
date: "11/16/2021"
output:
  word_document: default
pdf_document: default
html_document: default
---
  
  ####INTRODUCTION#####
The data used for this project was generated from 1992 baseball salary (link: http://www.amstat.org/publications/jse/datasets/baseball.dat.txt). We choose salary as the response variable and the remaining features were analyzed to predict features and relationship that could influence the salary of baseball players.  The data sets were subjected to exploratory data analysis, stepwise elimination method, linear model and elastic net model. 


####READING OF DATA INTO R#####
The first step was to import the data set in R studio. We checked the head, structure and dimension of the data set. This will help us to know the sample size and the variables (features) been analyzed.
```{r}   
###############Q1a Exploratory Data Analysis############
library(caret)  # classification and regression training
library(glmnet) # fits generalized linear and similar models via penalized maximum likelihood.

baseball <- read.table(file="http://www.amstat.org/publications/jse/datasets/baseball.dat.txt",
                       header = F, col.names=c("salary", "batting.avg", "OBP", "runs", "hits",
                                               "doubles", "triples", "homeruns", "RBI", "walks", "strike.outs",
                                               "stolen.bases", "errors", "free.agency.elig", "free.agent.91",
                                               "arb.elig", "arb.91", "name"))
head(baseball)
str(baseball)
dim(baseball)
```


```{r}
####Q1a####
plot(baseball)
baseball$log_salary=log(baseball$salary)
hist(baseball$salary, main = 'History of Raw Salary', xlab = "Salary")
hist(baseball$log_salary,main="Histogram of Logarithmic Salary",xlab="Log-transformed salary")
```
Comment Q1a: Continuous variables plots showing histogram of salary and logarithm of salary. The histogram of original salary is rightly-skewed while the Log-transformed Salary variable is not skewed to either left or right (uni-modal).The distribution behaves like a uniform distribution (normality).

```{r}
####Q1b####
baseball=baseball[, -1]
data=baseball
vnames <- colnames(data)
n <- nrow(data)
out <- NULL
for (j in 1:ncol(data)){
  vname <- colnames(data)[j]
  x <- as.vector(data[,j])
  n1 <- sum(is.na(x), na.rm=TRUE)  # NA
  n2 <- sum(x=="NA", na.rm=TRUE) # "NA"
  n3 <- sum(x==" ", na.rm=TRUE)  # missing
  nmiss <- n1 + n2 + n3
  nmiss <- sum(is.na(x))
  ncomplete <- n-nmiss
  out <- rbind(out, c(col.num=j, v.name=vname, mode=mode(x),
                      n.level=length(unique(x)),
                      ncom=ncomplete, nmiss= nmiss, miss.prop=nmiss/n))
}
out <- as.data.frame(out)
row.names(out) <- NULL
out
for (j in 1:NCOL(data)){
  print(colnames(data)[j])
  print(table(data[,j], useNA="ifany"))
}
```
Comment Q1b: None of the variables in the data set is Missing. 

-Continuous predictors are: Salary,batting.avg,OBP logarithmic salary..
-Integer count predictors are: runs , hits,triples, homeruns, RBI, walks,  strike.outs stolen.bases errors.
-Categorical Predictors: "free.agency.elig", "free.agent.9,"arb.elig", "arb.91" and name.
 
```{r}
                                   ####Q2a#####
full.model <- lm(log_salary~.-name, data=baseball)
summary(full.model)
formula(full.model) #show all the predictors used

```
Comment Q2a: firstly, we removed the column called "name" from the dataframe. Furthermore, we assigned our predictor variable (other variables in baseball) and response variable (logarithmic salary). In addition, we've now prepared our data ready to be fitted using linear model lm(). Fitting a linear regression model helps us to find the line of best fit in other to better understand the association between the Salary of the baseball players along with the predictor variables.

Comment Q2b:The full Adjusted R-squared of 79%  means there is 79% variation in the log-transformed salary. The p-value of the full model F statistics implies that the fitted model is efficient for predicting the salary. The p-value of the individual Regression coefficient t-test indicates that variables batting.avg,OBP,runs,hits,doubles, triples,homeruns,walks,stolen.bases, errors,arb.91 are not significant in the model when using alpha level of 0.05.
 
```{r}   
                                          #####Q3a#####
                            #####Step Wise Backward Elimination######

baseball.backward <- step(full.model, direction="backward", trace=0)
summary(baseball.backward)
formula(baseball.backward) #show all the predictors used
```
Comment Q3a: In other to use Step Wise Backward Elimination, we checked to see if the p<<n. p stands for predictors (number of variables or columns) versus "n" which stands for number of rows.Setting 'trace' = 0 helps use to get the most important variables as predicted by the backward stepwise. Based on the results from backward stepwise, the important variables are: salary ~ OBP + hits + RBI + walks + strike.outs + stolen.bases + free.agency.elig + free.agent.91 + arb.elig

```{r}
                                            ####Q3b####
finalmodelFit <- lm(log_salary ~ OBP + hits + RBI + walks + strike.outs + stolen.bases + 
                      free.agency.elig + free.agent.91 + arb.elig, data = baseball)
summary(finalmodelFit)
```
Comment Q3b: We've to finally fit the model using the vital variables from backward stepwise. THe RSE = 0.5341 while multiple R-square and Adjusted R-squared is 0.79. The limitation of stepwise could be seen from  the generated values because it's R-square is highly biased, with low biased RSE and p-values that is too small < 2.2e-16.

```{r}
                                                    ####Q3c####
library(car)
(final.jack <- rstudent(finalmodelFit))

hist(final.jack, main="Studentized residuals", xlab="Value", prob=TRUE)
```


```{r}

qqPlot(finalmodelFit, main="Q-Q plot", col="mediumblue",col.lines="orange", pch="*", grid=FALSE,
       labels=baseball$batting.avg, id.n=3)

# homscedasticity check
spreadLevelPlot(finalmodelFit, main="Spread level plot", col="mediumblue", col.lines="orange",   
                col.smoother="red", pch="*", grid=FALSE)

# Independece check 
durbinWatsonTest(finalmodelFit)

#Leverage plots
leveragePlots(finalmodelFit, col="mediumblue", col.lines="orange", grid=FALSE, pch=".", main="Leverage plots")

influenceIndexPlot(finalmodelFit, col="orange", id.n=3,id.col="mediumblue", grid=FALSE, pch="*")
influencePlot(finalmodelFit, col="orange", id.col="mediumblue")

vif(finalmodelFit)
```
Comments on  Q3c: To Check for Normality, Homoscedasticity,Independence and Multicollinearity. 
- The histogram of the studentized residuals indicate the data is approximately normally distributed.
- QQ-plot: The plot also indicates Normality since most of all the t-quantiles Studentized residuals falls on the 
  qqline.
  
- SpreadLevelPlot: The Spread levels plot indicates constancy of error variance (equal variance) since the fitted values   against the Studentized Residuals hover around the constant studentized Residuals 1 with no regular patterns.

- DurbinWatsonTest: Above is the result of  test for autocorrelation on our final method fitted . Since the p-value  >  
  alpha level of 0.05, we fail to reject the null hypothesis of no autocorrelation and conclude that there exist no  
  autocorrelation between the values of the Predictor variables. Then we conclude thatthere exist Independence.

- VIF: To check for Multicollinearity, if the vif score > 10, it means that the variable is problematic 
  (multicillinearity is present). Therefore for simplicity, we choose a cut-off value 10. The vif scores for the 
  analyzed variables were all < 10. 

                                     #####Q3d#####
Outliers: 322, 205, 284, 80,303

Comment Q3d: From the diagnostic plots and leverage plots the outliers were identified to be:322, 205,284, 80 and 303. Using the data frame we could also see that values occurred in variables or columns of  StudRes, Hat, CookD.

From the influence plot, the index  80 and 303 has the high leverage with lower cook distance (CookD) of 0.03180569 and 0.01535851 respectively. Thus, it indicates that they are less influential in comparison with  322, 205 ,284. The dataset at index 322,205,284 has the larger residual (StudRes) with -4.6001695, -4.2098585 and -4.8567850 with Cook Distance(CookD) 0.07504008, 0.08345036 and 0.15157341. Lastly, the data at row index 322, 205 and 284 as most influential and it might have significant changes in our model.

```{r}
                                              ####Q4a####
                          #####Find the best lambda using cross-validation####
x <- model.matrix(log_salary~.-name, data = baseball)[,-1] # Predictor variables
y <- baseball$log_salary  # Response variable

set.seed(123)  # reproducibility
crossvalid <- cv.glmnet(x, y, alpha = 0.5) #cross-validation based on alpha
crossvalid
```
Comment Q4a: We cross validate our model by using Elastic net model and setting our x and y variables already defined on section #Q2a# Fitting the ELastic net model. When alpha=0, Ridge Model is fit and  if alpha=1, a lasso model is fit whereas if alpha = (0, 1), a ELastic Net is fit.The alpha value was set at 5%.

```{r}
                                   #####Q4b#####
plot(crossvalid)   # plot cross-validation error
```
Comment Q4b: The cross-validation error plot indicates that the first dotted line has a  min error of 0.01243 at lambda -4.3 with more variables (14). As we move further to 1se error of 0.13962 at lambda of -1.9, we found out that important variables decreased to (7). Therefore, we can conclude that the later is more efficient and effective compared to the former see Q4a plot for more visualization.
```{r}
                                  #####Q4c######
plot(crossvalid$glmnet.fit,"lambda", label=FALSE)
                    ###fit the penalized model with best lambda####
blambda = crossvalid$lambda.min #print the lambda that gives minimum error
clambda = crossvalid$lambda.1se #print the lambda that gives minimum error
blambda
clambda
lambda_elnet <- clambda

               ###Fit the final Penalized model based on your lambda###
elnet.model <- glmnet(x = x, y = y, alpha  = 0.5,lambda = clambda)
```
Comment Q4c: The above plot shows that as the log-transformed lambda increases, the number of the feature/predictor variables decreases. The only difference is that it uses different schema to indicate that the as the lambda increases with respect to error while the number of important variables reduces. The different colors indicates various important analyzed variables. The amount of penalty that can be used for the Elastic Net regression using an alpha value of 0.5 is 0.1603943. Thus, It gives us the most regularized model that is within 1sd of the lambda 0.02071572 minimum error.


```{r}
                                             #####Q4d#####
elnet.model$beta  # weights of the fitted model
predictions <- predict(elnet.model, x)  # Make predictions
net=data.frame(RMSE = RMSE(predictions, y),Rsquare = R2(predictions, y)) # Evaluation metrics
net
```
Comment on Q4: Only 7 predictor variables are important using the elastic net model coefficients. It simply indicates that unimportant variables have been eliminated.In addition, the weights of the important variable has also been penalized to a minimum value. Our elastic net,we have fewer important variables since unimportant variables has been eliminated from the model.The weights of the important variable were been penalized to a minimum.
```{r}
                      #####Q4f Tuning of our model using Ridge/Regularization##### 
#a
set.seed(123) 
crossvalid <- cv.glmnet(x, y, alpha = 0) 
crossvalid
#b
plot(crossvalid)   # plot cross-validation error
#c
plot(crossvalid$glmnet.fit,"lambda", label=FALSE) # plot the model(lasso) path 

# fit the penalized model with best lambda
blambda = crossvalid$lambda.min #print the lambda that gives minimum error
clambda = crossvalid$lambda.1se #print the lambda that gives minimum error
lamda_ridge <- clambda
# Fit the final Penalized model based on your lambda
ridge.model <- glmnet(x = x, y = y, alpha  = 0, lambda = clambda)

#d
ridge.model$beta  # weights of the fitted model

# Make predictions
predictions <- predict(ridge.model, x)

# Evaluation metrics
ridge = data.frame(RMSE = RMSE(predictions, y),Rsquare = R2(predictions, y))
ridge
```

```{r}
                       #####Q4g Tuning of our model using Lasso/Regularization##### 
#a
set.seed(123) 
crossvalid <- cv.glmnet(x, y, alpha = 1) 
crossvalid

#b
plot(crossvalid)   # plot cross-validation error

#c
plot(crossvalid$glmnet.fit,"lambda", label=FALSE)   # plot the model(lasso) path 


# fit the penalized model with best lambda
blambda = crossvalid$lambda.min #print the lambda that gives minimum error
clambda = crossvalid$lambda.1se #print the lambda that gives minimum error
lamda_lasso <- clambda

# Fit the final Penalized model based on your lambda
lasso.model <- glmnet(x = x, y = y,alpha  = 1, lambda = clambda)

#d
lasso.model$beta  # weights of the fitted model

# Make predictions
predictions <- predict(lasso.model, x)

# Evaluation metrics
lasso = data.frame(RMSE = RMSE(predictions, y),Rsquare = R2(predictions, y))
lasso

```
Comment: Applying Lasso Regression model, only 5 important features/ predictor variables are important while other were eliminated. In addition, the weights of the important variable has also been penalized to a minimum value.

```{r}
                                                ##### #Q4g At lambda=0 #####
                                                    ####No Penalty #####

                 ##Elastic net Coefficients at Lambda is 0##
elnet.model_Zerolambda <- glmnet(x=x, y=y,alpha  = 0.5, lambda = 0)
elnet.model_Zerolambda$beta 


                   ###Ridge Coefficients at Lambda is 0###
ridge.model_Zerolambda <- glmnet(x = x, y = y, alpha  = 0,lambda = 0)
ridge.model_Zerolambda$beta


                  ####Lasso Coefficients at Lambda is 0####

lasso.model_Zerolambda <- glmnet(x = x, y = y, alpha  = 1,lambda = 0)
lasso.model_Zerolambda$beta
```

Comment: Elastic net, Ridge and Lasso Regression results into a Ordinary Least Square method when the penalty is zero (When lambda is zero). Full model Coefficients After penalizing the full model using Elastic net , Ridge and Lasso Regression. It was discovered that Lasso Regression Method penalized the predictors variable the most with just 5 predictors variable after the Shrinkage process. Elastic net has 7 variables and Ridge still maintain the full predictors.

                                   ###  Q5 CONCLUSION  ###
After penalizing the full model using Elastic net , Ridge and Lasso Regression It was seen that Lasso Regression penalized the predictors variable the most with fewer predictors variable after the Shrinkage process. Elastic net has 8 variables and Ridge full full predictors.


