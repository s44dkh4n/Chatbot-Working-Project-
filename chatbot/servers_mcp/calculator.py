import random
from typing import Any
from fastmcp import FastMCP

mcp = FastMCP("Calculator")

@mcp.tool
def add(a: float, b: float) -> dict[str, Any]:
    """
    Add two numbers together.
    """
    return {"a": a, "b": b, "result": a + b}

@mcp.tool
def subtract(a: float, b: float) -> dict[str, Any]:
    """
    Subtract 'b' from 'a'.
    """
    return {"a": a, "b": b, "result": a - b}

@mcp.tool
def multiply(a: float, b: float) -> dict[str, Any]:
    """
    Multiply two numbers together.
    """
    return {"a": a, "b": b, "result": a * b}

@mcp.tool
def divide(a: float, b: float) -> dict[str, Any]:
    """
    Divide 'a' by 'b'. Returns an error if 'b' is zero.
    """
    if b == 0:
        return {"error": "Division by zero is not allowed"}
    return {"a": a, "b": b, "result": a / b}

@mcp.tool
def remainder(a: float, b: float) -> dict[str, Any]:
    """
    Calculate the remainder (modulo) when 'a' is divided by 'b'.
    Returns an error if 'b' is zero.
    """
    if b == 0:
        return {"error": "Division by zero is not allowed"}
    return {"a": a, "b": b, "result": a % b}

@mcp.tool
def random_number(min_val: int, max_val: int) -> dict[str, Any]:
    """
    Generate a random integer between min_val and max_val (inclusive).
    """
    if min_val > max_val:
        return {"error": "min_val must be less than or equal to max_val"}
    result = random.randint(min_val, max_val)
    return {"min_val": min_val, "max_val": max_val, "result": result}

if __name__ == "__main__":
    mcp.run(transport="stdio")