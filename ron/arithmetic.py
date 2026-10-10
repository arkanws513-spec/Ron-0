"""Safe deterministic arithmetic for simple user questions; never evaluates arbitrary code."""
from __future__ import annotations

import ast
import math
import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_PREFIXES = (
    r"^(?:كم\s+يساوي|ما\s+ناتج|ناتج|احسب|احسبي|حساب|calculate|what\s+is|what's|solve)\s*",
    r"^(?:من\s+فضلك\s+)?",
)
_OPERATORS = (
    (r"\b(?:multiplied\s+by|times|multiply|x)\b", "*"),
    (r"\b(?:divided\s+by|divide|over)\b", "/"),
    (r"\b(?:plus|add)\b", "+"),
    (r"\b(?:minus|subtract)\b", "-"),
    (r"(?:ضرب|مضروب\s+في|في)", "*"),
    (r"(?:قسمة|مقسوم\s+على|على)", "/"),
    (r"(?:زائد|جمع)", "+"),
    (r"(?:ناقص|طرح)", "-"),
)
_MAX_INPUT_LENGTH = 120
_MAX_ABS_VALUE = 1e15


class _InvalidExpression(ValueError):
    pass


def _evaluate_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
        if abs(value) > _MAX_ABS_VALUE:
            raise _InvalidExpression("number too large")
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _evaluate_node(node.left), _evaluate_node(node.right)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        else:
            if right == 0:
                raise _InvalidExpression("division by zero")
            result = left / right
        if not math.isfinite(float(result)) or abs(result) > _MAX_ABS_VALUE:
            raise _InvalidExpression("result out of range")
        return result
    raise _InvalidExpression("unsupported expression")


def answer_arithmetic(text: str) -> str | None:
    """Answer a plain arithmetic expression or an explicit arithmetic question.

    Only numeric literals, parentheses, unary signs, and + - * / are accepted.
    Unsupported syntax is passed to the normal Ron conversation path.
    """
    raw = str(text).strip()
    if not raw or len(raw) > _MAX_INPUT_LENGTH:
        return None
    value = raw.translate(_DIGITS).replace("×", "*").replace("÷", "/")
    value = value.replace("−", "-").replace("٫", ".")
    for pattern, replacement in _OPERATORS:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)
    for pattern in _PREFIXES:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"[؟?!。]+$", "", value).strip()
    # A calculation must contain an operator and no leftover natural-language text.
    if not re.search(r"[+*/-]", value) or not re.fullmatch(r"[\d\s().+*/-]+", value):
        return None
    try:
        tree = ast.parse(value, mode="eval")
        result = _evaluate_node(tree)
    except (SyntaxError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if isinstance(result, float) and result.is_integer():
        rendered = str(int(result))
    else:
        rendered = format(result, ".10g")
    return f"الناتج: {rendered}"
