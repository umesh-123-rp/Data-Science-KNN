# Implement a KNN model to classify the animals in to categories
# Loading the required packages create training and test datasets
library(caret)
library(C50)
library(pROC)
library(mlbench)
library(lattice)
install.packages("ggvis")
library(ggvis)
install.packages("corrplot")
library(corrplot)
install.packages("psych")
library(psych)
# Read the dataset
zoo <- read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\KNN-R\\Zoo.csv")
View(zoo)
# dropping the animal name 
zoo <- zoo[,2:18]
str(zoo)
summary(zoo)
describe(zoo)
# All the features are categorical in nature with "0 - Absent" and "1-Present"
hist(zoo$hair)
hist(zoo$feathers)
hist(zoo$eggs)
hist(zoo$milk)
hist(zoo$airborne)
hist(zoo$aquatic)
hist(zoo$predator)
hist(zoo$toothed)
hist(zoo$backbone)
hist(zoo$breathes)
hist(zoo$venomous)
hist(zoo$fins)
hist(zoo$legs)
hist(zoo$tail)
hist(zoo$domestic)
hist(zoo$catsize)
hist(zoo$type)
# Histogram of types of animals indicate that the type 1 animals are highest in quantities
# Type 4 animals are lowest in quantities
# The following are the important features present available in many types of animals
# hairs, eggs,milk, predator,toothed,backbone,breathes,legs, tail and catsize

# Scatter plot
zoo %>% ggvis(~backbone, ~legs,fill = ~zoo$type) %>% layer_points()
zoo %>% ggvis(~milk, ~legs,fill = ~zoo$type) %>% layer_points()
zoo %>% ggvis(~hair, ~legs,fill = ~zoo$type) %>% layer_points()
zoo %>% ggvis(~eggs, ~legs,fill = ~zoo$type) %>% layer_points()
# Animals having backbones and legs are of type 1/ type 2 animals
# We can see that the animals with milk/no milk, but having 2 legs belong to
# same type of animals i.e type 2.
# But this is complicated to analyse one by one like this, we can check correlation
# Correlation among all variables including type of animal
corrplot(cor(zoo))
# There is a strong negative correlation between backbone and type of animals
# It indicates that the animals having backbones come under lower types of animals
# Similarly there is a strong negative correlation of type of animals with milk and tail
# There is a moderate positive correlation between eggs and type of animals
# There are relations among other features. Animals with milk do not possess eggs and vice-versa
# Toothed animals have less tendency to have features like eggs and milk
# Most of the animals having hairs will have milk , but no eggs.

# Since all the features are categorical in nature, let's convert them into factors
# converting the variables into factors 
zoo$hair <- as.factor(zoo$hair)
zoo$feathers <- as.factor(zoo$feathers)
zoo$eggs <- as.factor(zoo$eggs)
zoo$milk <- as.factor(zoo$milk)
zoo$airborne <- as.factor(zoo$airborne)
zoo$aquatic <- as.factor(zoo$aquatic)
zoo$predator <- as.factor(zoo$predator)
zoo$toothed <- as.factor(zoo$toothed)
zoo$backbone <- as.factor(zoo$backbone)
zoo$breathes <- as.factor(zoo$breathes)
zoo$venomous <- as.factor(zoo$venomous)
zoo$fins <- as.factor(zoo$fins)
zoo$legs <- as.factor(zoo$legs)
zoo$tail <- as.factor(zoo$tail)
zoo$domestic <- as.factor(zoo$domestic)
zoo$catsize <- as.factor(zoo$catsize)
zoo$type <- as.factor(zoo$type)
str(zoo)

# Checking proportion of types of animals in the dataset
round(prop.table(table(zoo$type))*100, digits = 1)

# Splitting dataset into  training and testing
set.seed(7)
InlocalTraining<-createDataPartition(zoo$type,p=0.70,list=F)

training<- zoo[InlocalTraining,]
testing <- zoo[-InlocalTraining,]

# create labels for training and test data
training_labels <- training[, 17]
testing_labels <- testing[, 17]

# selecting the optimal k-value
trcontrol <- trainControl(method = "repeatedcv", number = 10,repeats = 3)
set.seed(222)
fit <- train(type ~., data = training, method = 'knn', tuneLength = 20,
trControl = trcontrol, preProc = c("center","scale"))
fit
plot(fit)
# Optimum number of k - value is observed to be k=5
# Important features with contributions was checked as follows:
varImp(fit)
 # Prediction of type of animals 
pred <- predict(fit, newdata = testing[,-17])
confusionMatrix(pred, testing$type)

# Accuracy is found to be 0.93 at k=5
  
# Build a KNN model on taining dataset manually at different values of k
# Loading required packages
install.packages("class")
install.packages("gmodels")
library("class")
library("gmodels")

# Building the KNN model on training dataset with k = 7

zoo_pred <- knn(train = training[,-17], test = testing[,-17], cl = training_labels, k=7                                                       )
CrossTable(x = testing_labels, y = zoo_pred, prop.chisq = FALSE)
result <- data.frame(zoo_pred,testing_labels)
confusionMatrix(zoo_pred,testing_labels)
View(result)
# Accuracy is 0.89

# Building the KNN model on training dataset with k = 5

zoo_pred1 <- knn(train = training[,-17], test = testing[,-17], cl = training_labels, k=5)

CrossTable(x = testing_labels, y = zoo_pred, prop.chisq = FALSE)

result <- data.frame(zoo_pred1,testing_labels)
confusionMatrix(zoo_pred1,testing_labels)
View(result)
# Accuracy is 0.93

# Building the KNN model on training dataset with k = 2

zoo_pred2 <- knn(train = training[,-17], test = testing[,-17], cl =training_labels, k=2)
CrossTable(x = testing_labels, y = zoo_pred, prop.chisq = FALSE)
result <- data.frame(zoo_pred2,testing_labels)
confusionMatrix(zoo_pred2,testing_labels)
View(result)
# Accuracy is found to be 0.93

# CONCLUSION :
# KNN is a useful technique for such classification where the model is trained 
# with different features for different types of animals and then predict 
# type of animal based on given features.
# The test accuracy is 0.93 with k=2 which is very good.



