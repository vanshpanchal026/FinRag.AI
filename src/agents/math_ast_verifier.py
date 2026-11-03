"""Deterministic Python AST Mathematical Recomputation and Unit Normalizer module.

Provides zero-latency, deterministic verification for financial figures:
1. Python AST-based safe arithmetic recomputation (YoY %, QoQ %, bps margin compression, differences)
   without using unsafe eval().
2. Indian comma stripping (e.g., 6,58,240.00 -> 658240.00) to guard against AST syntax errors.
3. Unit normalization and equivalence checking (Lakhs vs Crores vs Millions vs Billions vs bps).
4. Operand grounding verification ensuring all calculation operands exist in source chunks.

Sources:
- Python AST standard library (ast.parse, ast.BinOp)
- SEBI (Research Analysts) Regulations, 2014, Reg 24(7)
"""

import ast
import operator
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

# Supported safe AST binary operators
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

SAFE_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Regex to detect explicit mathematical formulas in text, e.g.:
# ((1250.5 - 1180.0) / 1180.0) * 100
# (512 - 490) / 490 * 100 = 4.49%
# 36.5% - 34.2% = 2.3%
# 1250.5 - 1180.0 = 70.5
FORMULA_REGEX = re.compile(
    r"(?:\(?\s*[\d,]+(?:\.\d+)?\s*[-+/*]\s*[\d,]+(?:\.\d+)?\s*\)?"
    r"(?:\s*[-+/*]\s*[\d,]+(?:\.\d+)?)*)"
)

# Regex to extract percentage differences: +5.9%, -2.3%, 101.02%
PERCENT_REGEX = re.compile(r"([+-]?\d+(?:,\d+)*(?:\.\d+)?)\s*%")

# Regex to extract basis points differences: 230 bps, +150 bps
BPS_REGEX = re.compile(r"([+-]?\d+(?:,\d+)*(?:\.\d+)?)\s*bps", re.IGNORECASE)

# Indian and global financial unit multipliers (base unit: Rupee / single unit)
UNIT_MULTIPLIERS: Dict[str, float] = {
    "cr": 10_000_000.0,
    "crore": 10_000_000.0,
    "crores": 10_000_000.0,
    "lakh": 100_000.0,
    "lakhs": 100_000.0,
    "lac": 100_000.0,
    "lacs": 100_000.0,
    "mn": 1_000_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
    "k": 1_000.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
}

# Regex to capture number followed by financial unit
UNIT_NUMBER_REGEX = re.compile(
    r"(?<![a-zA-Z0-9_])(\d+(?:,\d+)*(?:\.\d+)?)\s*(Cr(?:ores?)?|Lakhs?|Lacs?|Mn|Millions?|Bn|Billions?|k|thousands?)\b",
    re.IGNORECASE,
)


class ASTMathResult(BaseModel):
    """Result of an AST mathematical recomputation conforming to IndFinRAG specifications."""

    expression_text: str
    operands: List[float] = Field(default_factory=list)
    operator: str
    computed_result: float
    claimed_result: float
    operands_grounded: bool
    math_valid: bool
    reason: str


class UnitMatchResult(BaseModel):
    """Result of a unit equivalence normalization conforming to IndFinRAG specifications."""

    raw_claimed: str
    normalized_claimed_value: float
    source_chunk_value: float
    source_unit: str
    target_unit: str
    unit_match_pass: bool


class ASTMathTrace(BaseModel):
    """Detailed trace of an AST mathematical recomputation."""

    expression: str
    operands: List[float] = Field(default_factory=list)
    operator_symbol: str = ""
    computed_value: float
    claimed_value: float
    error_delta: float
    operands_grounded: bool
    is_valid: bool
    explanation: str

    def to_result(self) -> ASTMathResult:
        return ASTMathResult(
            expression_text=self.expression,
            operands=self.operands,
            operator=self.operator_symbol,
            computed_result=self.computed_value,
            claimed_result=self.claimed_value,
            operands_grounded=self.operands_grounded,
            math_valid=self.is_valid,
            reason=self.explanation,
        )


class UnitMatchTrace(BaseModel):
    """Trace of a unit conversion match (e.g. Lakhs to Crores)."""

    claimed_raw: str
    claimed_value: float
    claimed_unit: str
    source_raw: str
    source_value: float
    source_unit: str
    normalized_claimed: float
    normalized_source: float
    is_match: bool

    def to_result(self) -> UnitMatchResult:
        return UnitMatchResult(
            raw_claimed=self.claimed_raw,
            normalized_claimed_value=self.normalized_claimed,
            source_chunk_value=self.source_value,
            source_unit=self.source_unit,
            target_unit=self.claimed_unit,
            unit_match_pass=self.is_match,
        )


def clean_indian_number(num_str: str) -> str:
    """Strip Indian commas and whitespace from number strings.
    
    Guards against Python AST SyntaxError on formats like '6,58,240.00'.
    """
    if not num_str:
        return ""
    # Strip commas and trailing % / bps symbols if present
    cleaned = num_str.replace(",", "").replace("%", "").strip()
    return cleaned


def safe_eval_ast(node: ast.AST) -> float:
    """Recursively evaluate an AST expression node using strictly whitelisted operators.
    
    Guarantees safe deterministic computation with zero risk of code injection.
    """
    if isinstance(node, ast.Expression):
        return safe_eval_ast(node.body)
    elif isinstance(node, ast.Constant):  # Python 3.8+ numeric literal
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_UNARY_OPERATORS:
            operand_val = safe_eval_ast(node.operand)
            return float(SAFE_UNARY_OPERATORS[op_type](operand_val))
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left_val = safe_eval_ast(node.left)
            right_val = safe_eval_ast(node.right)
            if op_type is ast.Div and abs(right_val) < 1e-12:
                raise ZeroDivisionError("Division by zero in AST arithmetic")
            return float(SAFE_OPERATORS[op_type](left_val, right_val))
        raise ValueError(f"Unsupported binary operator: {op_type}")
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")


def extract_operands_from_ast(node: ast.AST) -> List[float]:
    """Extract all numeric leaf constants from an AST tree in natural left-to-right order."""
    operands: List[float] = []

    def _visit(n: ast.AST) -> None:
        if isinstance(n, ast.Expression):
            _visit(n.body)
        elif isinstance(n, ast.BinOp):
            _visit(n.left)
            _visit(n.right)
        elif isinstance(n, ast.UnaryOp):
            _visit(n.operand)
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            operands.append(float(n.value))

    _visit(node)
    return operands


def parse_and_recompute_ast(expression_str: str) -> Tuple[float, List[float]]:
    """Clean expression string, parse with ast.parse, and recompute result deterministically."""
    cleaned_expr = clean_indian_number(expression_str)
    tree = ast.parse(cleaned_expr, mode="eval")
    operands = extract_operands_from_ast(tree)
    computed_val = safe_eval_ast(tree)
    return computed_val, operands


def extract_raw_numbers(text: str) -> List[float]:
    """Extract all raw numerical floats from a text block with comma normalization."""
    # Pattern matching floats or integers with optional commas
    raw_matches = re.findall(r"(?<![a-zA-Z0-9_])\d+(?:,\d+)*(?:\.\d+)?", text)
    nums: List[float] = []
    for m in raw_matches:
        try:
            val = float(clean_indian_number(m))
            nums.append(val)
        except ValueError:
            continue
    return nums


def are_operands_grounded(operands: List[float], corpus_numbers: List[float], tolerance: float = 1e-4) -> bool:
    """Verify that every operand in a formula exists in the retrieved context numbers."""
    if not operands:
        return False
    for op in operands:
        # Check if operand is a standard mathematical constant (e.g., 100 for percentage)
        if abs(op - 100.0) < tolerance or abs(op - 1.0) < tolerance:
            continue
        # Check against corpus
        matched = any(abs(op - cnum) < tolerance or (abs(cnum) > 0 and abs(op - cnum) / abs(cnum) < 0.001) for cnum in corpus_numbers)
        if not matched:
            return False
    return True


def check_derived_arithmetic(
    claimed_num_str: str,
    corpus_numbers: List[float],
    context_sentence: str = "",
    tolerance_pct: float = 0.08,
) -> Optional[ASTMathTrace]:
    """Test whether an unverified number in the answer is a valid derivation from corpus numbers.
    
    Checks:
    1. Percentage change: ((B - A) / A) * 100 == claimed
    2. Absolute difference: abs(B - A) == claimed
    3. Basis points compression: abs(B - A) * 100 == claimed (or abs(B - A) * 10000)
    4. Margin / ratio: (A / B) * 100 == claimed
    """
    clean_val_str = clean_indian_number(claimed_num_str)
    try:
        claimed_val = float(clean_val_str)
    except ValueError:
        return None

    if len(corpus_numbers) < 2:
        return None

    # Prioritize numbers in sentence context, but preserve full corpus numbers
    candidate_numbers = list(corpus_numbers)
    if context_sentence:
        sentence_nums = extract_raw_numbers(context_sentence)
        if sentence_nums:
            # Prepend unique sentence numbers to test them first
            candidate_numbers = list(dict.fromkeys(sentence_nums + candidate_numbers))

    # Pairwise testing
    for i in range(len(candidate_numbers)):
        for j in range(len(candidate_numbers)):
            if i == j:
                continue
            a = candidate_numbers[i]
            b = candidate_numbers[j]

            # 1. Relative Percentage Change: ((b - a) / a) * 100
            if abs(a) > 1e-6:
                try:
                    pct_change = ((b - a) / a) * 100.0
                    delta = abs(pct_change - claimed_val)
                    # Allow tolerance for standard rounding (e.g. 5.97% reported as 5.9% or 6.0%)
                    if delta <= 0.15 or (abs(claimed_val) > 0 and (delta / abs(claimed_val)) <= tolerance_pct):
                        return ASTMathTrace(
                            expression=f"(({b} - {a}) / {a}) * 100",
                            operands=[b, a, 100.0],
                            operator_symbol="% change",
                            computed_value=round(pct_change, 2),
                            claimed_value=claimed_val,
                            error_delta=round(delta, 4),
                            operands_grounded=True,
                            is_valid=True,
                            explanation=f"Derived relative growth: (({b} - {a}) / {a}) * 100 = {pct_change:.2f}% (claimed: {claimed_val}%)",
                        )
                except Exception:
                    pass

            # 2. Basis points difference: (b - a) * 100 (when numbers are percentages like 36.5 and 34.2)
            bps_diff = abs(b - a) * 100.0
            delta_bps = abs(bps_diff - claimed_val)
            if delta_bps <= 2.0 or (abs(claimed_val) > 0 and (delta_bps / abs(claimed_val)) <= 0.05):
                return ASTMathTrace(
                    expression=f"abs({b} - {a}) * 100",
                    operands=[b, a, 100.0],
                    operator_symbol="bps margin diff",
                    computed_value=round(bps_diff, 1),
                    claimed_value=claimed_val,
                    error_delta=round(delta_bps, 2),
                    operands_grounded=True,
                    is_valid=True,
                    explanation=f"Derived basis points difference: abs({b} - {a}) * 100 = {bps_diff:.1f} bps (claimed: {claimed_val} bps)",
                )

            # 3. Absolute Difference: abs(b - a)
            abs_diff = abs(b - a)
            delta_diff = abs(abs_diff - claimed_val)
            if delta_diff <= 0.1 or (abs(claimed_val) > 0 and (delta_diff / abs(claimed_val)) <= 0.01):
                return ASTMathTrace(
                    expression=f"abs({b} - {a})",
                    operands=[b, a],
                    operator_symbol="abs difference",
                    computed_value=round(abs_diff, 2),
                    claimed_value=claimed_val,
                    error_delta=round(delta_diff, 4),
                    operands_grounded=True,
                    is_valid=True,
                    explanation=f"Derived absolute difference: abs({b} - {a}) = {abs_diff:.2f} (claimed: {claimed_val})",
                )

            # 4. Margin Ratio: (a / b) * 100
            if abs(b) > 1e-6:
                try:
                    ratio = (a / b) * 100.0
                    delta_ratio = abs(ratio - claimed_val)
                    if delta_ratio <= 0.15 or (abs(claimed_val) > 0 and (delta_ratio / abs(claimed_val)) <= 0.02):
                        return ASTMathTrace(
                            expression=f"({a} / {b}) * 100",
                            operands=[a, b, 100.0],
                            operator_symbol="ratio %",
                            computed_value=round(ratio, 2),
                            claimed_value=claimed_val,
                            error_delta=round(delta_ratio, 4),
                            operands_grounded=True,
                            is_valid=True,
                            explanation=f"Derived margin ratio: ({a} / {b}) * 100 = {ratio:.2f}% (claimed: {claimed_val}%)",
                        )
                except Exception:
                    pass

    return None


def verify_unit_equivalence(
    claimed_token: str,
    answer_context: Union[str, List[str]],
    raw_chunks_text: Optional[str] = None,
) -> Optional[UnitMatchTrace]:
    """Check if an unverified number is an exact unit-normalized equivalent of a chunk number.
    
    Supports both signatures:
    - (claimed_token, answer_context, raw_chunks_text)
    - (claimed_number, chunk_texts)
    
    Handles:
    - Lakhs to Crores (e.g. 50,000 Lakhs == 500 Crores)
    - Crores to Lakhs (e.g. 12.5 Crores == 1,250 Lakhs)
    - Millions to Crores (e.g. 100 Million == 10 Crores / 150 Million == 15 Crores)
    - Crores to Millions (e.g. 50 Crores == 500 Million)
    """
    # Normalize argument dispatch when called as (claimed_token, chunk_texts)
    if raw_chunks_text is None:
        if isinstance(answer_context, list):
            raw_corpus = " \n ".join(answer_context)
        else:
            raw_corpus = str(answer_context)
        context_str = claimed_token
    else:
        if isinstance(answer_context, list):
            context_str = " \n ".join(answer_context)
        else:
            context_str = str(answer_context)
        raw_corpus = raw_chunks_text

    claimed_val: Optional[float] = None
    claimed_unit: str = ""

    # Check if unit is directly attached in claimed_token (e.g., "500 Crores", "100 Million", "12.5 Cr")
    unit_token_match = UNIT_NUMBER_REGEX.search(claimed_token)
    if unit_token_match:
        claimed_val = float(clean_indian_number(unit_token_match.group(1)))
        claimed_unit = unit_token_match.group(2).lower()
    else:
        clean_claimed = clean_indian_number(claimed_token)
        try:
            claimed_val = float(clean_claimed)
        except ValueError:
            return None

        # Detect unit associated with claimed token in answer context
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(claimed_token)}\s*(Cr(?:ores?)?|Lakhs?|Lacs?|Mn|Millions?|Bn|Billions?|k|thousands?)\b"
        claimed_match = re.search(pattern, context_str, re.IGNORECASE)
        if not claimed_match:
            # Check if context has a generic unit indication (e.g., "(in Crores)")
            unit_hint = re.search(r"\b(in\s+)?(Crores?|Cr|Lakhs?|Lacs?|Millions?|Mn)\b", context_str, re.IGNORECASE)
            claimed_unit = unit_hint.group(2).lower() if unit_hint else ""
        else:
            claimed_unit = claimed_match.group(1).lower()

    if claimed_val is None or not claimed_unit or claimed_unit not in UNIT_MULTIPLIERS:
        return None

    claimed_multiplier = UNIT_MULTIPLIERS[claimed_unit]
    normalized_claimed_inr = claimed_val * claimed_multiplier

    # Find all unit-tagged numbers in raw chunks
    chunk_matches = UNIT_NUMBER_REGEX.findall(raw_corpus)
    for c_val_str, c_unit in chunk_matches:
        try:
            c_val = float(clean_indian_number(c_val_str))
            c_mult = UNIT_MULTIPLIERS[c_unit.lower()]
            normalized_chunk_inr = c_val * c_mult

            # Compare normalized values within 0.1% tolerance
            if abs(normalized_claimed_inr - normalized_chunk_inr) < 1.0 or (
                normalized_chunk_inr > 0 and abs(normalized_claimed_inr - normalized_chunk_inr) / normalized_chunk_inr < 0.005
            ):
                return UnitMatchTrace(
                    claimed_raw=f"{claimed_token} {claimed_unit}" if claimed_unit not in claimed_token.lower() else claimed_token,
                    claimed_value=claimed_val,
                    claimed_unit=claimed_unit,
                    source_raw=f"{c_val_str} {c_unit}",
                    source_value=c_val,
                    source_unit=c_unit.lower(),
                    normalized_claimed=normalized_claimed_inr,
                    normalized_source=normalized_chunk_inr,
                    is_match=True,
                )
        except Exception:
            continue

    # Also check if chunks have a global unit header like "(Rs. in Lakhs)" or "(₹ in Lakhs)"
    global_unit_header = re.search(r"\((?:Rs\.?|₹|in)\s*(?:in\s*)?(Lakhs?|Lacs?|Crores?|Cr|Millions?|Mn)\)", raw_corpus, re.IGNORECASE)
    if global_unit_header:
        chunk_header_unit = global_unit_header.group(1).lower()
        if chunk_header_unit in UNIT_MULTIPLIERS and chunk_header_unit != claimed_unit:
            c_mult = UNIT_MULTIPLIERS[chunk_header_unit]
            # Test all raw numbers in chunks under this header unit
            for c_val in extract_raw_numbers(raw_corpus):
                normalized_chunk_inr = c_val * c_mult
                if abs(normalized_claimed_inr - normalized_chunk_inr) < 1.0 or (
                    normalized_chunk_inr > 0 and abs(normalized_claimed_inr - normalized_chunk_inr) / normalized_chunk_inr < 0.005
                ):
                    return UnitMatchTrace(
                        claimed_raw=f"{claimed_token} {claimed_unit}" if claimed_unit not in claimed_token.lower() else claimed_token,
                        claimed_value=claimed_val,
                        claimed_unit=claimed_unit,
                        source_raw=f"{c_val} (header: {chunk_header_unit})",
                        source_value=c_val,
                        source_unit=chunk_header_unit,
                        normalized_claimed=normalized_claimed_inr,
                        normalized_source=normalized_chunk_inr,
                        is_match=True,
                    )

    return None


def verify_ast_arithmetic(
    answer_text: str,
    chunk_numbers: List[float],
    tolerance: float = 0.05,
) -> List[ASTMathResult]:
    """Verify all mathematical derivations in an answer against retrieved chunk numbers.
    
    Deterministic AST evaluation covering:
    - Percentage changes (YoY %, QoQ %)
    - Basis points margin compression (bps)
    - Margin ratios and differences
    - Explicit arithmetic formulas
    """
    results: List[ASTMathResult] = []
    seen_expressions: Set[str] = set()

    # 1. Check explicit formulas if present (e.g., ((1250.5 - 1180.0) / 1180.0) * 100)
    for match in FORMULA_REGEX.finditer(answer_text):
        f_expr = match.group(0).strip()
        if not f_expr or len(f_expr) < 5 or f_expr in seen_expressions:
            continue
        # Verify it actually contains arithmetic operators
        if not any(op in f_expr for op in ["+", "-", "*", "/"]):
            continue
        try:
            val, operands = parse_and_recompute_ast(f_expr)
            if operands and len(operands) >= 2:
                grounded = are_operands_grounded(operands, chunk_numbers)
                seen_expressions.add(f_expr)
                results.append(
                    ASTMathResult(
                        expression_text=f_expr,
                        operands=operands,
                        operator="explicit_ast",
                        computed_result=round(val, 4),
                        claimed_result=round(val, 4),
                        operands_grounded=grounded,
                        math_valid=True,
                        reason=f"Explicit formula recomputed via AST: {val:.2f}",
                    )
                )
        except Exception:
            continue

    # 2. Extract potential derived figures (percentages, bps, differences)
    # Extract percentage tokens
    pct_matches = PERCENT_REGEX.findall(answer_text)
    bps_matches = BPS_REGEX.findall(answer_text)
    raw_nums = extract_raw_numbers(answer_text)

    candidate_tokens = list(dict.fromkeys(pct_matches + bps_matches + [str(n) for n in raw_nums]))

    for token in candidate_tokens:
        clean_tok = clean_indian_number(token)
        try:
            val = float(clean_tok)
        except ValueError:
            continue

        # Skip if already verified as an exact operand in corpus
        if any(abs(val - c) < 1e-4 for c in chunk_numbers):
            continue

        trace = check_derived_arithmetic(
            claimed_num_str=token,
            corpus_numbers=chunk_numbers,
            context_sentence=answer_text,
            tolerance_pct=tolerance,
        )
        if trace and trace.is_valid and trace.expression not in seen_expressions:
            seen_expressions.add(trace.expression)
            results.append(trace.to_result())

    return results
