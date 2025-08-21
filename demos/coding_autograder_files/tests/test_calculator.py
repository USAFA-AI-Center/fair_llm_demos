import pytest
import sys
import os

# Import the student's code (will be dynamically loaded)
from temp_student_code import add, subtract, multiply, divide, power

class TestCalculatorFunctions:
    """Test suite for calculator assignment"""
    
    def test_add_positive(self):
        """Test addition with positive numbers"""
        assert add(2, 3) == 5
        assert add(10, 20) == 30
    
    def test_add_negative(self):
        """Test addition with negative numbers"""
        assert add(-1, -1) == -2
        assert add(-5, 3) == -2
    
    def test_add_zero(self):
        """Test addition with zero"""
        assert add(0, 0) == 0
        assert add(5, 0) == 5
    
    def test_subtract_positive(self):
        """Test subtraction with positive numbers"""
        assert subtract(10, 5) == 5
        assert subtract(3, 3) == 0
    
    def test_subtract_negative(self):
        """Test subtraction with negative numbers"""
        assert subtract(-5, -3) == -2
        assert subtract(5, -3) == 8
    
    def test_multiply_basic(self):
        """Test basic multiplication"""
        assert multiply(3, 4) == 12
        assert multiply(0, 100) == 0
    
    def test_multiply_negative(self):
        """Test multiplication with negative numbers"""
        assert multiply(-2, 3) == -6
        assert multiply(-2, -3) == 6
    
    def test_divide_basic(self):
        """Test basic division"""
        assert divide(10, 2) == 5
        assert divide(9, 3) == 3
    
    def test_divide_float(self):
        """Test division resulting in float"""
        assert divide(7, 2) == 3.5
        assert divide(1, 3) == pytest.approx(0.3333, rel=1e-3)
    
    def test_divide_by_zero(self):
        """Test division by zero raises appropriate error"""
        with pytest.raises((ValueError, ZeroDivisionError)):
            divide(10, 0)
    
    def test_power_positive(self):
        """Test power function with positive exponents"""
        assert power(2, 3) == 8
        assert power(5, 2) == 25
    
    def test_power_zero(self):
        """Test power function with zero exponent"""
        assert power(10, 0) == 1
        assert power(5, 0) == 1
    
    def test_power_negative(self):
        """Test power function with negative exponents"""
        assert power(2, -1) == 0.5
        assert power(4, -2) == 0.0625
