# src/tools.py
"""
Арифметические инструменты для LangGraph.
Все функции должны иметь docstring.
"""

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a by b"""
    return a / b

TOOLS = [add, multiply, divide]