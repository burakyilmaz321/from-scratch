# Load Iris
iris <- as.data.table(iris) 

# Make it a binary classification problem
iris <- iris[Species != 'virginica'] 

# Normalize data
iris[, x1 := (iris$Sepal.Length - min(iris$Sepal.Length)) / (max(iris$Sepal.Length)-min(iris$Sepal.Length))]
iris[, x2 := (iris$Sepal.Width - min(iris$Sepal.Width)) / (max(iris$Sepal.Width)-min(iris$Sepal.Width))]
iris[, x3 := (iris$Petal.Length - min(iris$Petal.Length)) / (max(iris$Petal.Length)-min(iris$Petal.Length))]
iris[, x4 := (iris$Petal.Width - min(iris$Petal.Width)) / (max(iris$Petal.Width)-min(iris$Petal.Width))]

# Categorical variable encoding
iris[Species == 'setosa', y := 1]
iris[Species == 'versicolor', y := 0]

# Initialize Network
n <- nrow(iris)

a1 <- as.matrix(cbind(rep(1, n), iris[, 7:10]))
y <- as.matrix(iris[, y])

w1 <- matrix(rnorm(20), 5, 4)
w2 <- matrix(rnorm(4))

sigm <- function(x) { 1 / (1 + exp(-x)) }

# Train network
for (i in 1:100000) {
  a2 <- sigm(a1 %*% w1); 
  a3 <- sigm(a2 %*% w2); 
  d2 <- (a3 - y) * a3 * (1 - a3);
  d1 <- d2 %*% t(w2) * a2 * (1 - a2);
  w2 <- w2 - 0.5 * (t(a2) %*% d2);
  w1 <- w1 - 0.5 * (t(a1) %*% d1); 
  e <- (y - a3)**2 / (2 * n)
}

