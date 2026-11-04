train <- read.csv("train.csv")
test <- read.csv("test.csv")

# TODO: replace with your model
predictions <- runif(nrow(test), min(train$price), max(train$price))

write.csv(data.frame(id = test$id, price = predictions), "test-predictions.csv", row.names = FALSE)
