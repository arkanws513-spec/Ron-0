"""Safe deterministic arithmetic for Ron-0.

Only a deliberately small mathematical grammar is accepted. No eval/exec,
attribute access, names, function calls, or arbitrary Python expressions.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_EXPR_CANDIDATE = re.compile(r"(?<![A-Za-z_])([+-]?(?:[0-9]|\()[0-9\s.+\-*/×÷()%]*[0-9)])")
_OPERATORS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow}
_UNARY = {ast.UAdd, ast.USub}
_PREVIOUS_PATTERNS = (
    "احسب العملية السابقة",
    "احسب الحساب السابق",
    "احسب المسألة السابقة",
    "نتيجة العملية السابقة",
    "احسب السابق",
    "calculate the previous calculation",
    "calculate previous calculation",
    "calculate the previous one",
    "previous calculation",
)


class ArithmeticError(ValueError):
    """Raised when an expression is outside Ron's safe arithmetic grammar."""


@dataclass(frozen=True)
class Calculation:
    expression: str
    value: int | float

    @property
    def answer(self) -> str:
        if isinstance(self.value, float) and self.value.is_integer():
            return str(int(self.value))
        return format(self.value, ".12g")


def is_previous_calculation_request(text: str) -> bool:
    normalized = " ".join(str(text).translate(_DIGITS).lower().split())
    return any(phrase in normalized for phrase in _PREVIOUS_PATTERNS)


def extract_expression(text: str) -> str | None:
    """Extract a numeric expression from a short natural-language request."""
    normalized = str(text).translate(_DIGITS).replace("×", "*").replace("÷", "/")
    for match in _EXPR_CANDIDATE.finditer(normalized):
        candidate = match.group(1).strip()
        if not any(operator in candidate for operator in "+-*/%"):
            continue
        candidate = re.sub(r"\s+", "", candidate)
        # Reject fragments cut from words or unsupported comparison/equality syntax.
        if not candidate or "=" in candidate or not re.fullmatch(r"[0-9.+\-*/()%]+", candidate):
            continue
        try:
            evaluate_expression(candidate)
        except (ArithmeticError, SyntaxError, ValueError, OverflowError, ZeroDivisionError):
            continue
        return candidate
    return None


def evaluate_expression(expression: str) -> int | float:
    """Evaluate arithmetic using a strict AST allow-list and bounded operations."""
    source = str(expression).translate(_DIGITS).replace("×", "*").replace("÷", "/").strip()
    if not source or len(source) > 120:
        raise ArithmeticError("expression is empty or too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticError("invalid arithmetic expression") from exc

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
            if not math.isfinite(float(value)) or abs(value) > 10**100:
                raise ArithmeticError("number outside supported range")
            return value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            value = visit(node.operand)
            result = value if isinstance(node.op, ast.UAdd) else -value
            return bounded(result)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ArithmeticError("exponent outside supported range")
            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    result = left // right
                elif isinstance(node.op, ast.Mod):
                    result = left % right
                else:
                    result = left ** right
            except (ZeroDivisionError, OverflowError) as exc:
                raise ArithmeticError("invalid arithmetic operation") from exc
            return bounded(result)
        raise ArithmeticError("unsupported syntax")

    def bounded(value: int | float) -> int | float:
        if not math.isfinite(float(value)) or abs(value) > 10**100:
            raise ArithmeticError("result outside supported range")
        return value

    return visit(tree)


def looks_like_arithmetic_expression(text: str) -> bool:
    """Identify numeric operator patterns even when they fail safe evaluation."""
    normalized = str(text).translate(_DIGITS).replace("×", "*").replace("÷", "/")
    return bool(re.search(r"\\d\\s*(?:\\*\\*|//|[+*/%\\-])\\s*\\d", normalized))


def calculate_from_text(text: str) -> Calculation | None:
    expression = extract_expression(text)
    if expression is None:
        return None
    return Calculation(expression=expression, value=evaluate_expression(expression))
