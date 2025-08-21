"""
Assignment 1: Basic Calculator Functions
Student: Alice Johnson
Date: 2024-01-15
"""

def add(a, b):
    """
    Add two numbers together.
    
    Args:
        a (float): First number
        b (float): Second number
    
    Returns:
        float: Sum of a and b
    """
    return a + b

def subtract(a, b):
    """
    Subtract b from a.
    
    Args:
        a (float): First number
        b (float): Second number to subtract
    
    Returns:
        float: Difference of a and b
    """
    return a - b

def multiply(a, b):
    """
    Multiply two numbers.
    
    Args:
        a (float): First number
        b (float): Second number
    
    Returns:
        float: Product of a and b
    """
    return a * b

def divide(a, b):
    """
    Divide a by b with error handling for division by zero.
    
    Args:
        a (float): Numerator
        b (float): Denominator
    
    Returns:
        float: Quotient of a and b
    
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power(base, exponent):
    """
    Calculate base raised to the power of exponent.
    
    Args:
        base (float): Base number
        exponent (float): Exponent
    
    Returns:
        float: Result of base^exponent
    """
    return base ** exponent
