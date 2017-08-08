# Load Iris
iris <- as.data.table(iris) 

# Make it a binary classification problem
# iris <- iris[Species != 'virginica'] 

# Normalize data
iris[, x1 := (iris$Sepal.Length - min(iris$Sepal.Length)) / (max(iris$Sepal.Length)-min(iris$Sepal.Length))]
iris[, x2 := (iris$Sepal.Width - min(iris$Sepal.Width)) / (max(iris$Sepal.Width)-min(iris$Sepal.Width))]
iris[, x3 := (iris$Petal.Length - min(iris$Petal.Length)) / (max(iris$Petal.Length)-min(iris$Petal.Length))]
iris[, x4 := (iris$Petal.Width - min(iris$Petal.Width)) / (max(iris$Petal.Width)-min(iris$Petal.Width))]

# Categorical variable encoding
# iris[Species == 'setosa', y := 1]
# iris[Species == 'versicolor', y := 0]

# Prepare data
a1 <- as.matrix(cbind(rep(1, nrow(iris)), iris[, 6:9]))
y <- data.table(spec=iris[, Species], V1=0, V2=0, V3=0)

# One-Hot Encoding
y[spec == "setosa", V1 := 1]
y[spec == "versicolor", V2 := 1]
y[spec == "virginica", V3 := 1]
y[, spec := NULL]
y <- as.matrix(y)

# Split train/test
test_x <- rbind(a1[41:50,], a1[91:100,], a1[141:150,])
a1 <- rbind(a1[1:40,], a1[51:90,], a1[101:140,])

test_y <- rbind(y[41:50,], y[91:100,], y[141:150,])
y <- rbind(y[1:40,], y[51:90,], y[101:140,])

# Parameters
HIDDEN_UNITS <- 4
INPUT_UNITS <- dim(a1)[2]
OUTPUT_UNITS <- dim(y)[2]
N <- nrow(a1)

# Initialize Network
w1 <- matrix(rnorm(INPUT_UNITS * HIDDEN_UNITS), INPUT_UNITS, HIDDEN_UNITS)
w2 <- matrix(rnorm(HIDDEN_UNITS * OUTPUT_UNITS), HIDDEN_UNITS, OUTPUT_UNITS)

sigm <- function(x) { 1 / (1 + exp(-x)) }

# Train network
for (i in 1:100000) {
  a2 <- sigm(a1 %*% w1); 
  a3 <- sigm(a2 %*% w2); 
  d2 <- (a3 - y) * a3 * (1 - a3);
  d1 <- d2 %*% t(w2) * a2 * (1 - a2);
  w2 <- w2 - 0.5 * (t(a2) %*% d2);
  w1 <- w1 - 0.5 * (t(a1) %*% d1); 
  e <- (y - a3)**2 / (2 * N)
}

# Test network
pred <- sigm(sigm(test_x %*% w1) %*% w2)
test_error <- mean((test_y - pred)**2 / nrow(test_y)) 

compare <- data.table(pred_prob=pred, y=test_y)
compare[, pred1 := round(pred_prob.V1)]
compare[, pred2 := round(pred_prob.V2)]
compare[, pred3 := round(pred_prob.V3)]
