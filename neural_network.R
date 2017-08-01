a1 <- matrix(c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0), 4,3)
y <- matrix(c(1, 0, 0, 1))
w1 <- matrix(rnorm(9), 3, 3)
w2 <- matrix(rnorm(3))
sigm <- function(x) { 1 / (1 + exp(-x)) }
for (i in 1:1000) {
  a2 <- sigm(a1 %*% w1); 
  a3 <- sigm(a2 %*% w2); 
  d2 <- (a3 - y) * a3 * (1 - a3); d2
  d1 <- d2 %*% t(w2) * a2 * (1 - a2); d1
  w2 <- w2 - t(a2) %*% d2; w2
  w1 <- w1 - t(a1) %*% d1; w1 
}