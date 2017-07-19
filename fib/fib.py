'''
Iterative fibonacci algorithm.
Accepts a command line argument as the parameter
of fib function
'''

import sys

def fib(n):
    '''
    Prints first n fibonacci numbers iteratively
    '''
    x = 0
    y = 1
    while x < n:
        print(x)
        z = x + y
        x = y
        y = z

if __name__ == "__main__":
    try: 
        fib(int(sys.argv[1]))
    except IndexError:
        print("No argument given: n = 255")
        fib(255)
    except ValueError:
        print("Argument is not an integer!")