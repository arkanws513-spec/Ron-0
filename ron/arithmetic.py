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


def _linear_form(node: ast.AST) -> tuple[float, float]:
    """Return (coefficient_of_x, constant) for a safe linear expression."""
    if isinstance(node, ast.Expression):
        return _linear_form(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
        if not math.isfinite(value) or abs(value) > _MAX_ABS_VALUE:
            raise _InvalidExpression("number too large")
        return 0.0, value
    if isinstance(node, ast.Name) and node.id in {"x", "س"}:
        return 1.0, 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a, b = _linear_form(node.operand)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return sign * a, sign * b
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left_a, left_b = _linear_form(node.left)
        right_a, right_b = _linear_form(node.right)
        if isinstance(node.op, ast.Add):
            return left_a + right_a, left_b + right_b
        if isinstance(node.op, ast.Sub):
            return left_a - right_a, left_b - right_b
        if isinstance(node.op, ast.Mult):
            if left_a != 0 and right_a != 0:
                raise _InvalidExpression("non-linear equation")
            return left_a * right_b + right_a * left_b, left_b * right_b
        if right_a != 0 or right_b == 0:
            raise _InvalidExpression("division by a variable or zero")
        return left_a / right_b, left_b / right_b
    raise _InvalidExpression("unsupported linear expression")


def answer_linear_equation(text: str) -> str | None:
    """Solve a single-variable linear equation using a restricted AST, never eval()."""
    raw = str(text or "").strip()
    if not raw or len(raw) > _MAX_INPUT_LENGTH or raw.count("=") != 1:
        return None
    value = raw.translate(_DIGITS).replace("×", "*").replace("÷", "/").replace("−", "-")
    value = re.sub(r"^(?:حل\s+(?:المعادلة|معادله)|solve)\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<=\d)\s*([xس])\b", r"*\1", value)
    left, right = (part.strip() for part in value.split("=", 1))
    if not left or not right or not re.search(r"[xس]", value):
        return None
    if not re.fullmatch(r"[\dxس\s().+*/-]+", left) or not re.fullmatch(r"[\\dxس\\s().+*/-]+", right):
        return None
    try:
        left_a, left_b = _linear_form(ast.parse(left, mode="eval"))
        right_a, right_b = _linear_form(ast.parse(right, mode="eval"))
        coefficient = left_a - right_a
        constant = right_b - left_b
        if math.isclose(coefficient, 0.0, abs_tol=1e-12):
            return None
        solution = constant / coefficient
        if not math.isfinite(solution) or abs(solution) > _MAX_ABS_VALUE:
            return None
    except (SyntaxError, ValueError, OverflowError, ZeroDivisionError):
        return None
    rendered = str(int(solution)) if float(solution).is_integer() else format(solution, ".10g")
    return f"الحل: x = {rendered}"
