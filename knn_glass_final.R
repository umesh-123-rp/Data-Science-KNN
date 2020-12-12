# Prepare a model for glass classification using KNN

# Data Description:
  
# RI : refractive index

# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)

# Mg: Magnesium

# Al: Aluminum

# Si: Silicon

# K:Potassium

# Ca: Calcium

# Ba: Barium

# Fe: Iron

# Type: Type of glass: (class attribute)
# 1 -- building_windows_float_processed
# 2 --building_windows_non_float_processed
# 3 --vehicle_windows_float_processed
# 4 --vehicle_windows_non_float_processed (none in this database)
# 5 --containers
# 6 --tableware
# 7 --headlamps
# Solution :
# Load required packages
install.packages("class")
install.packages("gmodels")
install.packages("corrplot")
install.packages("psych")
library (caret)
library(C50)
library(class)
library(gmodels)
library(corrplot)
library(psych)
#load the data
glass <- read.csv("C:\\Users\\IN102385\\OneDrive - Super-Max Personal Care Pvt. Ltd\\Assignment - Data Science -UL\\KNN-R\\glass.csv")
View(glass)
summary(glass)
describe(glass)
hist(glass$RI)
hist(glass$Na)
hist(glass$Mg)
hist(glass$Al)
hist(glass$Si)
hist(glass$K)
hist(glass$Ca)
hist(glass$Ba)
hist(glass$Fe)
hist(glass$Type)
# Out of 7 types of glasses, maximum types are of type 1 & 2 and less number is of types 5 & 6.
# Among all the constituents, Fe, Ba and K contents are there in all types of glasses
boxplot(glass$RI)
boxplot(glass$Na)
boxplot(glass$Mg)
boxplot(glass$Al)
boxplot(glass$Si)
boxplot(glass$K)
boxplot(glass$Ca)
boxplot(glass$Ba)
boxplot(glass$Fe)
# There are many outliers in Refractive Index,Na,Al,Si,K,Ca,Ba and Fe.
# The constituents differ with high variation for different types of glasses

# checking proportion  of class variable
prop.table(table(glass$Type))*100

# Correlation among all the variables and types of glasses
pairs(glass)
corrplot(cor(glass))
# It seems that there is a strong correlation between Mg and Glass Type.
# Type of glass is either 5 or 6 if it contains magnesium.
# There is a strong positive correlation between Ca and RI
# There is a moderate negative correlation between Si and RI
cor(glass$RI,glass$Ca)
cor(glass$RI, glass$Si)
# Correlation Coefficient between RI and Ca is 0.81 and between RI and Si is -0.54

#creating training and test data
colnames(glass)
str(glass)
# Declaring type of glasses as a factor
glass$Type <- as.factor(glass$Type)

#Create a function to normalize the data
norm <- function(x){ 
  return((x-min(x))/(max(x)-min(x)))
}
#Apply the normalization function to glass dataset
norm_glass<- as.data.frame(lapply(glass[1:9], norm))
View(norm_glass)
norm_glass <- cbind(norm_glass,glass[10])
View(norm_glass)

# Splitting the dataset into training and testing
set.seed(7)
inlocalTraining<-createDataPartition(norm_glass$Type,p=0.70,list=F)
training<-norm_glass[inlocalTraining,]
testing<-norm_glass[-inlocalTraining,]

# create labels for training and test data
training_label <- training[,10]
testing_label <- testing[,10]
#TRaining a model
# Building and testing the model
glass_test_pred <-knn(train = training[,-10],test = testing[,-10],cl = training_label,k=2)
confusionMatrix(glass_test_pred, testing$Type)
# Accuracy is 0.75

# creating another model using grid search method
ctrl<-trainControl(method="repeatedcv",repeats=10)
my_knn_model <-train(Type ~.,method ="knn",data = training,trControl=ctrl,tuneGrid = expand.grid(k=c(2,3,5,7,9,10,13,15,17)))
my_knn_model
plot(my_knn_model)
pred <- predict(my_knn_model, newdata = testing[,-10])
confusionMatrix(pred, testing$Type)
# The optimum value of k= 3
# With k=3, We get Accuracy is 0.75

# CONCLUSION :
# KNN model has been prepared with k=3 and accuracy = 0.75
# Same data set can be analysed through many other classification and regression 
# techniques to optimise accuracy
# With given combination of glass compositions, we can identify the type of glass through this model
