setwd("/Users/xucc/Documents/GMU/CS657/assigns/project")
library(e1071)
library(ggplot2)
library(dplyr)

data1 = read.csv("/Users/xucc/Documents/GMU/CS657/assigns/project/new_sample2.csv", header = TRUE)
rows = nrow(data1)
index1 = sample(rows, 0.8*rows, replace = FALSE)
train = data1[index1,]
test = data1[-index1,]

trainX = train[,c(1:9)]
trainY = train[,10]
testX = test[,c(1:9)]
testY = test[,10]
ctrl <- trainControl(method = "cv", number = 10)

ggplot(train, aes(x, y )) +
  geom_point(aes(color = place_id), size = 0.1) + 
  theme_minimal() +
  theme(legend.position = "none") +
  ggtitle("Check-ins colored by place_id")

model <- svm(placeId ~ ., data = train) 
pred <- predict(model, test)
table(pred, test$placeId)

