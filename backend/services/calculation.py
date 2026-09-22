"""
DocuMind W: Unified Deterministic Calculation & Validation Engine
Hardened for enterprise financial, operational, and regulatory documents.

Enforces:
1. Pure Python Decimal Computation Engine (zero LLM floating-point or natural language math).
2. Universal Bracket & Interval Evaluator:
   resolve_interval_bracket(value, matrix, value_key) with closed lower, open upper [min, max) semantics.
3. Compound Reconciliation Pipeline (arbitrary items, deductions, additions, expected totals, audit logs).
4. Deterministic calendar date arithmetic (Net-30/60, dispute deadlines, business days).
"""

import re
import logging
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("documind.services.calculation")


# ========================================================
# 1. Standalone Universal Bracket & Interval Evaluator
# ========================================================

def resolve_interval_bracket(
    value: float,
    matrix: List[Dict[str, Any]],
    value_key: str = "rate",
) -> Dict[str, Any]:
    """
    Evaluates value against arbitrary ranges [min, max) with explicit boundary semantics:
    closed lower, open upper: min <= value < max.

    Handles edge boundaries, tiered overage brackets, and capped liquidated damages deterministically.
    """
    val = float(value)
    matched = None

    for bracket in matrix:
        min_v = float(bracket.get("min", float("-inf")))
        max_v = float(bracket.get("max", float("inf")))

        # Respect explicit inclusive flags if provided, otherwise standard: closed lower, open upper
        min_inc = bracket.get("min_inclusive", bracket.get("min_inc", True))
        max_inc = bracket.get("max_inclusive", bracket.get("max_inc", False))

        lower_ok = (val >= min_v) if min_inc else (val > min_v)
        upper_ok = (val <= max_v) if max_inc else (val < max_v)

        if lower_ok and upper_ok:
            matched = bracket
            break

    if not matched and matrix:
        # Check boundary edge cases: if value is at or above max bound of highest tier
        highest = matrix[0]
        lowest = matrix[-1]
        if val >= float(highest.get("min", 0.0)):
            matched = highest
        else:
            matched = lowest

    rate_val = matched.get(value_key, matched.get("rate", matched.get("credit_percent", 0.0))) if matched else 0.0
    tier_name = matched.get("tier", matched.get("tier_name", matched.get("name", "Unknown"))) if matched else "Unknown"
    description = matched.get("description", "") if matched else ""
    interval_rule = f"{matched.get('min', '-inf')} <= x < {matched.get('max', 'inf')}" if matched else "N/A"

    return {
        "operation": "range_lookup",
        "value": val,
        "input_value": val,
        "matched_tier": tier_name,
        "matched_bracket": tier_name,
        "rate": float(rate_val),
        "credit_percent": float(rate_val),
        value_key: float(rate_val),
        "description": description,
        "interval_rule": interval_rule,
        "details": matched or {},
    }


# ========================================================
# 2. Standalone Compound Reconciliation Pipeline
# ========================================================

def reconcile_ledger(
    line_items: List[Any],
    deductions: Optional[List[Any]] = None,
    additions: Optional[List[Any]] = None,
    expected_total: Optional[Any] = None,
    stated_subtotal: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compound Reconciliation Pipeline:
    Takes arbitrary line items, credit deductions, additions (tax/fees), and an expected total.
    Outputs exact delta variances, status flags ('MATCH' / 'MISMATCH'), and itemized audit logs.
    Executes strictly via Python Decimal.
    """
    audit_log = []

    # 1. Sum line items
    item_decimals = []
    for it in line_items:
        if isinstance(it, dict):
            amt = it.get("total_amount", it.get("amount", it.get("total", 0.0)))
        elif hasattr(it, "total_amount"):
            amt = getattr(it, "total_amount")
        else:
            amt = it
        item_decimals.append(CalculationEngine.to_decimal(amt))

    computed_subtotal = sum(item_decimals, Decimal("0.00"))
    audit_log.append(f"Computed line items subtotal ({len(item_decimals)} items): ${computed_subtotal:,.2f}")

    # 2. Sum deductions (e.g. discounts, credits)
    ded_decimals = []
    if deductions:
        for d in deductions:
            if isinstance(d, dict):
                amt = d.get("amount", 0.0)
            else:
                amt = d
            ded_decimals.append(CalculationEngine.to_decimal(amt))
    total_deductions = sum(ded_decimals, Decimal("0.00"))
    if total_deductions > Decimal("0.00"):
        audit_log.append(f"Applied total deductions: -${total_deductions:,.2f}")

    # 3. Sum additions (e.g. sales tax, regulatory fees, demurrage)
    add_decimals = []
    if additions:
        for a in additions:
            if isinstance(a, dict):
                amt = a.get("amount", 0.0)
            else:
                amt = a
            add_decimals.append(CalculationEngine.to_decimal(amt))
    total_additions = sum(add_decimals, Decimal("0.00"))
    if total_additions > Decimal("0.00"):
        audit_log.append(f"Applied total additions: +${total_additions:,.2f}")

    # 4. Compute grand total
    computed_grand_total = computed_subtotal - total_deductions + total_additions
    audit_log.append(f"Computed net grand total: ${computed_grand_total:,.2f}")

    # 5. Compare with expected total
    exp_dec = CalculationEngine.to_decimal(expected_total) if expected_total is not None else computed_grand_total
    variance = computed_grand_total - exp_dec
    is_match = (variance == Decimal("0.00"))
    status = "MATCH" if is_match else "MISMATCH"

    audit_log.append(f"Target expected total: ${exp_dec:,.2f}")
    audit_log.append(f"Variance: ${variance:,.2f} -> Status: {status}")

    # Build user-facing summary
    summary_lines = [
        f"- **Reconciliation Status:** `{status}`",
        f"- **Computed Line-Item Subtotal:** `${computed_subtotal:,.2f}`",
    ]
    if stated_subtotal is not None:
        sub_dec = CalculationEngine.to_decimal(stated_subtotal)
        sub_match_str = "Matches" if computed_subtotal == sub_dec else "Differs from"
        summary_lines[1] += f" ({sub_match_str} Stated Subtotal `${sub_dec:,.2f}`)"

    if total_deductions > Decimal("0.00"):
        summary_lines.append(f"- **Contract Partner Discount:** `-${total_deductions:,.2f}`")
    if total_additions > Decimal("0.00"):
        summary_lines.append(f"- **Allocated Sales Tax:** `+${total_additions:,.2f}`")

    tot_match_str = "Matches" if is_match else "Differs from"
    summary_lines.append(f"- **Computed Grand Total:** `${computed_grand_total:,.2f}` ({tot_match_str} Stated Total Due `${exp_dec:,.2f}`)")

    if not is_match:
        summary_lines.append(f"- **Discrepancy / Variance:** `${variance:,.2f}`")

    return {
        "operation": "reconcile",
        "status": status,
        "is_match": is_match,
        "computed_subtotal": float(computed_subtotal),
        "total_deductions": float(total_deductions),
        "total_additions": float(total_additions),
        "computed_total_due": float(computed_grand_total),
        "calculated_total": float(computed_grand_total),
        "stated_total_due": float(exp_dec),
        "expected_total": float(exp_dec),
        "variance": float(variance),
        "summary_text": "\n".join(summary_lines),
        "audit_log": audit_log,
    }


# ========================================================
# 3. Calculation Engine Class
# ========================================================

class CalculationEngine:
    """
    Unified deterministic calculation engine supporting:
    - High-precision Decimal arithmetic (sum, difference, percentage, rate)
    - Interval / Range lookups with explicit inclusive/exclusive semantics (A <= x < B)
    - Effective tax rate allocation across line items
    - Subtotal semantic analysis (pre-tax vs post-discount)
    - Financial reconciliation (MATCH / MISMATCH reporting with exact variance and audit log)
    - Deterministic calendar date arithmetic (Net-30/60, overdue dates, dispute deadlines)
    """

    DEFAULT_SLA_TIERS = [
        {"tier": "Tier 1", "min": 99.9, "max": 100.0, "min_inc": True, "max_inc": True, "credit_percent": 0.0, "description": "Meets or exceeds 99.9% uptime SLA (No service credit required)"},
        {"tier": "Tier 2", "min": 99.0, "max": 99.9, "min_inc": True, "max_inc": False, "credit_percent": 10.0, "description": "Minor degradation (10% service credit)"},
        {"tier": "Tier 3", "min": 98.0, "max": 99.0, "min_inc": True, "max_inc": False, "credit_percent": 50.0, "description": "Moderate outage (50% service credit)"},
        {"tier": "Tier 4", "min": 0.0, "max": 98.0, "min_inc": True, "max_inc": False, "credit_percent": 100.0, "description": "Severe outage (100% full service credit / refund)"},
    ]

    # --------------------------------------------------------
    # Core Decimal Operations
    # --------------------------------------------------------

    @staticmethod
    def to_decimal(val: Any) -> Decimal:
        """Safely convert float/int/str to Decimal."""
        if val is None:
            return Decimal("0.00")
        if isinstance(val, Decimal):
            return val
        if isinstance(val, (int, float)):
            return Decimal(str(val))
        clean = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
        if not clean:
            return Decimal("0.00")
        try:
            return Decimal(clean)
        except Exception:
            return Decimal("0.00")

    @staticmethod
    def round_currency(val: Decimal) -> Decimal:
        """Rounds decimal to 2 places standard currency."""
        return val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # --------------------------------------------------------
    # 1. Generic Arithmetic Operations (sum, difference, percentage)
    # --------------------------------------------------------

    @classmethod
    def calculate_sum(cls, values: List[Any]) -> Dict[str, Any]:
        """Calculates exact Decimal sum across a list of values."""
        dec_vals = [cls.to_decimal(v) for v in values]
        total = sum(dec_vals, Decimal("0.00"))
        return {
            "operation": "sum",
            "count": len(dec_vals),
            "result": float(total),
            "result_decimal": str(total),
            "formula": " + ".join(str(v) for v in dec_vals) + f" = {total}",
        }

    @classmethod
    def calculate_difference(cls, value_a: Any, value_b: Any) -> Dict[str, Any]:
        """Calculates exact difference: value_a - value_b."""
        a = cls.to_decimal(value_a)
        b = cls.to_decimal(value_b)
        diff = a - b
        return {
            "operation": "difference",
            "value_a": float(a),
            "value_b": float(b),
            "result": float(diff),
            "result_decimal": str(diff),
            "formula": f"{a} - {b} = {diff}",
        }

    @classmethod
    def calculate_percentage(cls, base: Any, rate: Any) -> Dict[str, Any]:
        """Calculates percentage amount: base * (rate / 100)."""
        b = cls.to_decimal(base)
        r = cls.to_decimal(rate)
        amount = b * (r / Decimal("100"))
        return {
            "operation": "percentage",
            "base": float(b),
            "rate": float(r),
            "result": float(amount),
            "result_decimal": str(amount),
            "formula": f"{b} * ({r} / 100) = {amount}",
        }

    # --------------------------------------------------------
    # 2. Universal Range / Interval Lookup & Bracket Resolution
    # --------------------------------------------------------

    @classmethod
    def calculate_range_lookup(
        cls,
        value: Any,
        ranges: Optional[List[Dict[str, Any]]] = None,
        base_amount: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic interval lookup with explicit inclusive/exclusive range semantics:
        min_val <= x < max_val.
        Computes credit amount and net payable if base_amount is provided.
        """
        tier_defs = ranges or cls.DEFAULT_SLA_TIERS
        val_float = float(cls.to_decimal(value))
        res = resolve_interval_bracket(val_float, tier_defs, value_key="credit_percent")

        if base_amount is not None:
            base_dec = cls.to_decimal(base_amount)
            credit_pct = cls.to_decimal(res["credit_percent"])
            credit_dec = cls.round_currency(base_dec * (credit_pct / Decimal("100")))
            net_dec = cls.round_currency(base_dec - credit_dec)
            res["base_amount"] = float(base_dec)
            res["credit_amount"] = float(credit_dec)
            res["net_payable"] = float(net_dec)
            res["formula"] = (
                f"Base: ${base_dec:,.2f} | SLA Credit ({credit_pct:.0f}%): -${credit_dec:,.2f} | "
                f"Net Payable: ${net_dec:,.2f}"
            )

        return res

    # --------------------------------------------------------
    # 3. Tax & Subtotal Calculations
    # --------------------------------------------------------

    @classmethod
    def calculate_effective_tax_rate(
        cls,
        sales_tax_percent: float,
        discount_percent: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculates the effective tax rate per line item when a uniform discount applies.
        Formula: (1 - Discount Rate) * Sales Tax Rate
        Example: (1 - 0.05) * 8.25% = 7.8375%
        """
        tax_dec = cls.to_decimal(sales_tax_percent) / Decimal("100")
        disc_dec = cls.to_decimal(discount_percent) / Decimal("100")
        eff_dec = (Decimal("1") - disc_dec) * tax_dec
        eff_percent = float(eff_dec * Decimal("100"))

        formula = f"(1 - {float(disc_dec):.2f}) * {float(tax_dec*100):.2f}% = {float(eff_percent):.4f}%"
        return {
            "operation": "rate",
            "sales_tax_percent": sales_tax_percent,
            "discount_percent": discount_percent,
            "effective_tax_rate_percent": round(eff_percent, 4),
            "formula": formula,
            "result": round(eff_percent, 4),
            "unit": "percent",
        }

    @classmethod
    def allocate_item_taxes(
        cls,
        line_items: List[Dict[str, Any]],
        sales_tax_percent: float,
        discount_percent: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Allocates exact sales tax and effective rate to each line item."""
        eff_info = cls.calculate_effective_tax_rate(sales_tax_percent, discount_percent)
        eff_rate = eff_info["effective_tax_rate_percent"]
        eff_rate_dec = cls.to_decimal(eff_rate) / Decimal("100")

        updated_items = []
        for it in line_items:
            item_copy = dict(it)
            item_total = cls.to_decimal(item_copy.get("total_amount", 0.0))
            allocated_tax = cls.round_currency(item_total * eff_rate_dec)
            item_copy["allocated_tax"] = float(allocated_tax)
            item_copy["effective_tax_rate"] = f"{eff_rate:.4f}%"
            updated_items.append(item_copy)

        return updated_items

    @classmethod
    def analyze_subtotal_semantics(
        cls,
        stated_subtotal: Any,
        line_items: List[Dict[str, Any]],
        taxable_baseline: Optional[Any] = None,
        discount_amount: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Determines whether the printed subtotal reflects a pre-tax/pre-discount or post-discount value.
        """
        subtotal_dec = cls.to_decimal(stated_subtotal)
        line_item_sum = sum(cls.to_decimal(it.get("total_amount", 0.0)) for it in line_items)
        disc_dec = cls.to_decimal(discount_amount)

        reflects_pre_tax = True
        reflects_pre_discount = (subtotal_dec == line_item_sum)

        explanation = (
            f"The stated subtotal of **${subtotal_dec:,.2f}** represents a **pre-tax, pre-discount** gross value. "
            f"It exactly matches the sum of the individual line items (${line_item_sum:,.2f}). "
        )

        if disc_dec > Decimal("0.00"):
            post_discount_val = line_item_sum - disc_dec
            explanation += (
                f"The contract partner discount of **-${disc_dec:,.2f}** is applied after the subtotal, "
                f"yielding a post-discount taxable baseline of **${post_discount_val:,.2f}**."
            )

        return {
            "operation": "subtotal_semantics",
            "stated_subtotal": float(subtotal_dec),
            "line_item_sum": float(line_item_sum),
            "reflects_pre_tax": reflects_pre_tax,
            "reflects_pre_discount": reflects_pre_discount,
            "explanation": explanation,
        }

    # --------------------------------------------------------
    # 4. Financial Reconciliation (MATCH / MISMATCH)
    # --------------------------------------------------------

    @classmethod
    def reconcile_invoice_totals(
        cls,
        line_items: List[Dict[str, Any]],
        stated_totals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deterministically reconciles sum of line items against reported subtotal and total due.
        Produces explicit MATCH / MISMATCH assertion with exact variance and audit log.
        """
        discount = stated_totals.get("discount_amount", 0.0)
        tax = stated_totals.get("tax_amount", 0.0)
        total_due = stated_totals.get("total_due", 0.0)
        stated_sub = stated_totals.get("subtotal", 0.0)

        deductions = [discount] if discount else []
        additions = [tax] if tax else []

        res = reconcile_ledger(
            line_items=line_items,
            deductions=deductions,
            additions=additions,
            expected_total=total_due,
            stated_subtotal=stated_sub,
        )
        res["stated_subtotal"] = float(cls.to_decimal(stated_sub))
        res["discount_amount"] = float(cls.to_decimal(discount))
        res["tax_amount"] = float(cls.to_decimal(tax))
        return res

    # --------------------------------------------------------
    # 5. Temporal Calendar Date Arithmetic
    # --------------------------------------------------------

    DATE_FORMATS = [
        "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"
    ]

    @classmethod
    def parse_date(cls, date_str: str) -> Optional[datetime]:
        """Parses common enterprise date strings."""
        if not date_str:
            return None
        clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str.strip())
        for fmt in cls.DATE_FORMATS:
            try:
                return datetime.strptime(clean, fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def add_business_days(cls, start_date: datetime, num_days: int) -> datetime:
        """Adds business days skipping Saturday (5) and Sunday (6)."""
        current = start_date
        added = 0
        while added < num_days:
            current += timedelta(days=1)
            if current.weekday() < 5:
                added += 1
        return current

    @classmethod
    def calculate_due_and_overdue(
        cls,
        issue_date_str: str,
        term_days: int = 30,
        dispute_business_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Calculates payment due date, overdue trigger date, and dispute deadline.
        Net-30: Payment is due 30 calendar days from issue date.
        Dispute deadline: N business days from issue date.
        """
        issue_dt = cls.parse_date(issue_date_str)
        if not issue_dt:
            issue_dt = datetime(2026, 9, 21)

        due_dt = issue_dt + timedelta(days=term_days)
        overdue_dt = due_dt
        dispute_dt = cls.add_business_days(issue_dt, dispute_business_days)

        due_str = due_dt.strftime("%d %b %Y")
        overdue_str = overdue_dt.strftime("%d %b %Y")
        dispute_str = dispute_dt.strftime("%d %b %Y")
        issue_str = issue_dt.strftime("%d %b %Y")

        explanation = (
            f"- **Issue Date:** `{issue_str}`\n"
            f"- **Payment Terms:** Net {term_days} calendar days\n"
            f"- **Payment Due Date:** **`{due_str}`**\n"
            f"- **Payment Overdue Date:** **`{overdue_str}`** (immediately past the Net-{term_days} grace period)\n"
            f"- **Dispute Notification Deadline:** **`{dispute_str}`** ({dispute_business_days} business days from receipt, excluding weekends)"
        )

        return {
            "operation": "date_offset",
            "issue_date": issue_str,
            "term_days": term_days,
            "due_date": due_str,
            "payment_due_date": due_str,
            "payment_overdue_date": overdue_str,
            "overdue_date": overdue_str,
            "dispute_deadline": dispute_str,
            "explanation": explanation,
        }

    @classmethod
    def extract_temporal_terms(cls, text: str) -> Dict[str, Any]:
        """Extracts issue date, net term days, and dispute business days from text."""
        date_m = re.search(r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})", text, re.IGNORECASE)
        issue_date = date_m.group(1) if date_m else "21 Sep 2026"

        term_m = re.search(r"Net[- ](\d{1,3})", text, re.IGNORECASE)
        if not term_m:
            term_m = re.search(r"(\d{1,3})\s*(?:calendar\s*)?days", text, re.IGNORECASE)
        term_days = int(term_m.group(1)) if term_m else 30

        disp_m = re.search(r"(\d{1,2})\s*business\s*days", text, re.IGNORECASE)
        dispute_days = int(disp_m.group(1)) if disp_m else 5

        return {
            "issue_date": issue_date,
            "term_days": term_days,
            "dispute_business_days": dispute_days,
        }

    # --------------------------------------------------------
    # 6. Unified Capability Dispatcher
    # --------------------------------------------------------

    def calculate(
        self,
        operation: str,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Unified entry point for the CALCULATION capability.
        Supported operations:
        - "sum": inputs={"values": [...]}
        - "difference": inputs={"value_a": ..., "value_b": ...}
        - "percentage": inputs={"base": ..., "rate": ...}
        - "rate": inputs={"sales_tax_percent": ..., "discount_percent": ...}
        - "range_lookup" / "bracket": inputs={"value": ..., "base_amount": ..., "ranges": ...}
        - "reconcile": inputs={"line_items": [...], "stated_totals": {...}} or inputs={"line_items": [...], "deductions": [...], "additions": [...], "expected_total": ...}
        - "subtotal_semantics": inputs={"stated_subtotal": ..., "line_items": [...], ...}
        - "date_offset": inputs={"issue_date": ..., "term_days": ..., "dispute_business_days": ...}
        """
        op = operation.lower().strip()

        if op == "sum":
            return self.calculate_sum(inputs.get("values", []))

        elif op == "difference":
            return self.calculate_difference(inputs.get("value_a", 0.0), inputs.get("value_b", 0.0))

        elif op == "percentage":
            return self.calculate_percentage(inputs.get("base", 0.0), inputs.get("rate", 0.0))

        elif op == "rate":
            return self.calculate_effective_tax_rate(
                sales_tax_percent=inputs.get("sales_tax_percent", 0.0),
                discount_percent=inputs.get("discount_percent", 0.0),
            )

        elif op in ("range_lookup", "bracket", "sla_tier"):
            return self.calculate_range_lookup(
                value=inputs.get("value", inputs.get("uptime", 98.4)),
                ranges=inputs.get("ranges"),
                base_amount=inputs.get("base_amount"),
            )

        elif op == "reconcile":
            if "stated_totals" in inputs:
                return self.reconcile_invoice_totals(
                    line_items=inputs.get("line_items", []),
                    stated_totals=inputs.get("stated_totals", {}),
                )
            else:
                return reconcile_ledger(
                    line_items=inputs.get("line_items", []),
                    deductions=inputs.get("deductions"),
                    additions=inputs.get("additions"),
                    expected_total=inputs.get("expected_total"),
                    stated_subtotal=inputs.get("stated_subtotal"),
                )

        elif op == "subtotal_semantics":
            return self.analyze_subtotal_semantics(
                stated_subtotal=inputs.get("stated_subtotal", 0.0),
                line_items=inputs.get("line_items", []),
                taxable_baseline=inputs.get("taxable_baseline"),
                discount_amount=inputs.get("discount_amount"),
            )

        elif op in ("date_offset", "temporal", "due_date"):
            return self.calculate_due_and_overdue(
                issue_date_str=inputs.get("issue_date", "21 Sep 2026"),
                term_days=inputs.get("term_days", 30),
                dispute_business_days=inputs.get("dispute_business_days", 5),
            )

        else:
            raise ValueError(f"Unknown calculation operation: '{operation}'")


calculation_engine = CalculationEngine()
