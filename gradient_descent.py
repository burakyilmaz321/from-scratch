f = lambda x: x**2 - 4*x - 4 # Function:   f(x)  = x^2 - 4x + 4
f_d = lambda x: 2*x - 4      # Derivative: f'(x) = 2x - 4
x = 4                        # Initialize: x = 4
alpha = 0.1                  # Learning rate
epsilon = 0.00000000000001   # Epsilon: Is x' small enough?
i = 0                        # Loop counter
while f_d(x) > epsilon:
    i += 1
    x = x - alpha * f_d(x)
    print("Step " + str(i) + ": x = " + str(x))
print("Local Minima is at x = " + str(x))