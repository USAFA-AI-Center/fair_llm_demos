# Basic calculator functions
# Student submission

def add(a, b):
    # adds two numbers
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    result = a * b
    return result

def divide(a, b):
    # division function
    if b != 0:
        return a / b
    else:
        return None  # Should raise an error instead

def power(base, exp):
    result = 1
    for i in range(int(exp)):
        result = result * base
    return result  # Doesn't handle negative or float exponents correctly
