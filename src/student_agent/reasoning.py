"""Deterministic evidence interpretation for the L3B workflow.

The competition constrains model size.  This module deliberately uses no model at
all: every conclusion is derived from MCP fields and explicit, reviewable rules.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class Evidence:
    ref: str
    domain: str
    tool: str
    data: Any
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    primary_issue: str
    secondary_issues: tuple[str, ...]
    case_status: str
    shipment_verdict: str
    payment_verdict: str
    captured_total: float | None
    refunded_total: float | None
    refundable_total: float | None
    recommended_refund: float
    responsible_party: str
    cause_code: str
    actions: tuple[str, ...]


def normalize_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def walk(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            yield child_path, child
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, (*path, str(index)))


def values_for(value: Any, aliases: Iterable[str]) -> list[Any]:
    wanted = {normalize_key(alias) for alias in aliases}
    return [child for path, child in walk(value) if path and normalize_key(path[-1]) in wanted]


def scalar_strings(value: Any, aliases: Iterable[str]) -> list[str]:
    result: list[str] = []
    for child in values_for(value, aliases):
        candidates = child if isinstance(child, list) else [child]
        for candidate in candidates:
            if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
                text = str(candidate).strip()
                if text and text not in result:
                    result.append(text)
    return result


def first_scalar(value: Any, aliases: Iterable[str]) -> Any | None:
    for candidate in values_for(value, aliases):
        if candidate is not None and not isinstance(candidate, (dict, list)):
            return candidate
    return None


def decimal_value(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value).replace(",", "."))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def money(value: Any, aliases: Iterable[str]) -> Decimal | None:
    for candidate in values_for(value, aliases):
        parsed = decimal_value(candidate)
        if parsed is not None:
            return parsed
    return None


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _records(value: Any, collection_aliases: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {normalize_key(alias) for alias in collection_aliases}
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    for path, child in walk(value):
        if path and normalize_key(path[-1]) in wanted and isinstance(child, list):
            rows = [item for item in child if isinstance(item, dict)]
            if rows:
                return rows
    return [value]


def _domain_data(evidence: Iterable[Evidence], *domains: str) -> list[Any]:
    allowed = set(domains)
    return [item.data for item in evidence if item.domain in allowed]


def extract_entities(
    evidence: Iterable[Evidence], resolved_orders: list[str]
) -> dict[str, list[str]]:
    all_data = [item.data for item in evidence]
    return {
        "order_ids": _unique(resolved_orders)[:20],
        "item_ids": _unique(
            _strings_from_many(all_data, ("item_id", "order_item_id", "item_ids"))
        )[:20],
        "seller_ids": _unique(
            _strings_from_many(all_data, ("seller_id", "seller_ids"))
        )[:20],
        "payment_references": _unique(
            _strings_from_many(
                all_data,
                (
                    "payment_reference",
                    "payment_id",
                    "transaction_id",
                    "charge_id",
                ),
            )
        )[:20],
        "shipment_ids": _unique(
            _strings_from_many(all_data, ("shipment_id", "tracking_id", "tracking_code"))
        )[:20],
    }


def customer_context(evidence: Iterable[Evidence]) -> tuple[str | None, list[str]]:
    all_data = [item.data for item in evidence]
    customer_ids = _strings_from_many(
        all_data, ("customer_unique_id", "customer_id", "customer_unique_ids")
    )
    related = _strings_from_many(
        _domain_data(evidence, "customer"),
        ("related_order_ids", "order_ids", "order_id"),
    )
    return (customer_ids[0] if customer_ids else None), _unique(related)[:20]


def _shipment_verdict(evidence: list[Evidence]) -> tuple[str, list[str], bool]:
    data = [
        item.data
        for domain in ("shipment", "order", "item")
        for item in evidence
        if item.domain == domain
    ]
    if not data:
        return "insufficient_evidence", [], False

    status_text = " ".join(
        str(value).lower()
        for item in data
        for value in values_for(item, ("status", "shipment_status", "delivery_status"))
        if not isinstance(value, (dict, list))
    )
    if "lost" in status_text:
        return "lost", [], False
    if "return" in status_text:
        return "returned", [], True

    delivered = _first_time(
        data,
        (
            "delivered_at",
            "delivery_date",
            "order_delivered_customer_date",
            "actual_delivery_date",
        ),
    )
    estimated = _first_time(
        data,
        (
            "estimated_delivery_at",
            "estimated_delivery_date",
            "order_estimated_delivery_date",
        ),
    )
    handoff = _first_time(
        data,
        (
            "carrier_handoff_at",
            "shipped_at",
            "order_delivered_carrier_date",
            "handoff_at",
        ),
    )
    shipping_limit = _first_time(data, ("shipping_limit_date", "seller_deadline"))
    timeline_complete = delivered is not None and estimated is not None
    seller_late = (
        handoff is not None
        and shipping_limit is not None
        and _is_after(handoff, shipping_limit)
    )
    late_sellers = (
        _unique(_strings_from_many(data, ("seller_id", "late_seller_ids")))[:20]
        if seller_late
        else []
    )
    explicit_seller_late = _truthy(
        data,
        ("seller_late", "late_by_seller", "seller_handoff_late", "is_seller_late"),
    )
    explicit_logistics_late = _truthy(
        data,
        ("logistics_late", "carrier_late", "delivery_late", "is_delivery_late"),
    )
    if seller_late or explicit_seller_late:
        if not late_sellers:
            late_sellers = _unique(_strings_from_many(data, ("seller_id",)))[:20]
        return "seller_delay", late_sellers, timeline_complete
    if explicit_logistics_late or (
        delivered and estimated and _is_after(delivered, estimated)
    ):
        return "logistics_delay", [], timeline_complete
    if delivered and estimated and not _is_after(delivered, estimated):
        return "on_time", [], True
    if any(token in status_text for token in ("delivered", "complete")):
        return "on_time", [], timeline_complete
    return "insufficient_evidence", [], timeline_complete


def _payment_analysis(
    evidence: list[Evidence], order_total: Decimal | None
) -> tuple[str, Decimal | None, Decimal | None, Decimal | None, int]:
    payment_evidence = [item for item in evidence if item.domain == "payment"]
    payment_data = [item.data for item in payment_evidence]
    payment_row_data = [
        item.data for item in payment_evidence if "timeline" not in item.tool
    ]
    refund_data = _domain_data(evidence, "refund")
    if not payment_data:
        return "insufficient_evidence", None, None, None, 0

    rows = [
        row for data in payment_row_data for row in _records(data, ("payments", "charges"))
    ]
    captured = _explicit_or_sum(
        payment_data,
        ("captured_total_brl", "captured_total", "total_paid", "payment_total"),
        rows,
        ("payment_value", "captured_amount", "amount_brl", "amount"),
    )
    refunded = _explicit_or_sum(
        refund_data or payment_data,
        ("refunded_total_brl", "refunded_total", "total_refunded"),
        [row for data in refund_data for row in _records(data, ("refunds",))],
        ("refund_amount", "amount_brl", "amount"),
    )
    if refunded is None:
        refunded = Decimal("0")

    status_text = " ".join(
        str(value).lower()
        for item in [*payment_data, *refund_data]
        for value in values_for(
            item,
            (
                "status",
                "payment_status",
                "refund_status",
                "event_type",
                "reason_code",
            ),
        )
        if not isinstance(value, (dict, list))
    )
    references = [
        reference
        for row in rows
        for reference in scalar_strings(
            row, ("payment_reference", "transaction_id", "charge_id", "payment_id")
        )
    ]
    duplicate = len(references) != len(set(references)) or _truthy(
        payment_data, ("duplicate", "duplicate_capture")
    )
    duplicate = duplicate or "duplicate" in status_text
    if (
        captured is not None
        and order_total is not None
        and captured > order_total + Decimal("0.01")
        and len(rows) > 1
    ):
        duplicate = True
    refundable = None if captured is None else max(captured - refunded, Decimal("0"))
    if duplicate:
        verdict = "duplicate_capture"
    elif "refund" in status_text and "fail" in status_text:
        verdict = "refund_failed"
    elif "refund" in status_text and any(token in status_text for token in ("pending", "process")):
        verdict = "refund_pending"
    elif captured is not None and refunded >= captured and captured > 0:
        verdict = "refunded"
        refundable = Decimal("0")
    elif (
        captured is not None
        and order_total is not None
        and abs(captured - order_total) > Decimal("0.01")
    ):
        verdict = "capture_mismatch"
    else:
        verdict = "reconciled"
    return verdict, captured, refunded, refundable, len(rows)


def decide(evidence: list[Evidence], entity_status: str) -> tuple[Decision, list[str], bool]:
    order_data = _domain_data(evidence, "order")
    item_data = _domain_data(evidence, "item")
    status_text = " ".join(
        str(value).lower()
        for item in order_data
        for value in values_for(item, ("status", "order_status"))
        if not isinstance(value, (dict, list))
    )
    item_rows = [row for data in item_data for row in _records(data, ("items", "order_items"))]
    order_total = _explicit_or_sum(
        order_data,
        ("order_total", "total_brl", "total_amount", "expected_total_brl"),
        item_rows,
        ("price", "item_price", "amount_brl"),
        extra_aliases=("freight_value", "freight_brl", "shipping_cost"),
    )
    shipment, late_sellers, timeline_complete = _shipment_verdict(evidence)
    payment, captured, refunded, refundable, payment_rows = _payment_analysis(
        evidence, order_total
    )
    paid_balance = (captured or Decimal("0")) - (refunded or Decimal("0"))

    issues: list[str] = []
    if "cancel" in status_text and paid_balance > 0:
        issues.append("canceled_order_paid")
    if any(token in status_text for token in ("unavailable", "out_of_stock")) and paid_balance > 0:
        issues.append("unavailable_order_paid")
    issue_by_payment = {
        "duplicate_capture": "duplicate_charge",
        "capture_mismatch": "payment_mismatch",
        "refund_pending": "refund_pending",
        "refund_failed": "refund_failed",
    }
    if payment in issue_by_payment:
        issues.append(issue_by_payment[payment])
    if shipment == "seller_delay":
        issues.append("late_delivery_seller")
    elif shipment in {"logistics_delay", "lost"}:
        issues.append("late_delivery_logistics")
    if payment_rows > 1 and payment == "reconciled":
        issues.append("valid_split_payment")

    if not issues:
        if entity_status != "resolved" or not evidence:
            issues.append("insufficient_evidence")
        elif shipment in {"on_time", "returned"} and payment in {"reconciled", "refunded"}:
            issues.append("unsupported_claim")
        else:
            issues.append("insufficient_evidence")

    primary = issues[0]
    action_required = primary not in {
        "unsupported_claim",
        "valid_split_payment",
        "insufficient_evidence",
    }
    status = (
        "action_required"
        if action_required
        else "needs_investigation"
        if primary == "insufficient_evidence"
        else "no_action"
    )
    responsible = _responsible_party(primary)
    recommended = _refund_amount(primary, captured, refunded, order_total)
    actions = _default_actions(primary)
    decision = Decision(
        primary_issue=primary,
        secondary_issues=tuple(_unique(issues[1:])[:10]),
        case_status=status,
        shipment_verdict=shipment,
        payment_verdict=payment,
        captured_total=_float_or_none(captured),
        refunded_total=_float_or_none(refunded),
        refundable_total=_float_or_none(refundable),
        recommended_refund=float(recommended),
        responsible_party=responsible,
        cause_code=primary.upper(),
        actions=actions,
    )
    return decision, late_sellers, timeline_complete


def apply_policy(decision: Decision, policy_data: Any) -> Decision:
    """Use explicit authoritative policy values when the gateway supplies them."""
    branch = _policy_branch(policy_data, decision.primary_issue)
    if branch is None:
        return decision
    refund = money(branch, ("recommended_refund_brl", "refund_amount_brl"))
    action_values = values_for(branch, ("resolution_actions", "actions"))
    actions: tuple[str, ...] = decision.actions
    for value in action_values:
        if isinstance(value, list):
            clean = tuple(str(item)[:80] for item in value if isinstance(item, str) and item)
            if clean:
                actions = tuple(_unique(clean)[:8])
                break
    party = first_scalar(branch, ("responsible_party", "party_type"))
    allowed_parties = {
        "seller",
        "platform",
        "logistics_provider",
        "payment_provider",
        "customer",
        "unknown",
    }
    return Decision(
        **{
            **decision.__dict__,
            "recommended_refund": (
                float(refund) if refund is not None else decision.recommended_refund
            ),
            "responsible_party": str(party)
            if isinstance(party, str) and party in allowed_parties
            else decision.responsible_party,
            "actions": actions,
        }
    )


def detect_conflicts(evidence: list[Evidence]) -> list[dict[str, Any]]:
    checks = {
        "order_status": (("order_status", "status"), {"order"}),
        "delivered_at": (
            ("delivered_at", "order_delivered_customer_date"),
            {"order", "shipment"},
        ),
        "captured_total_brl": (
            ("captured_total_brl", "captured_total", "total_paid"),
            {"payment"},
        ),
        "refunded_total_brl": (
            ("refunded_total_brl", "refunded_total"),
            {"payment", "refund"},
        ),
    }
    results: list[dict[str, Any]] = []
    precedence = {"refund": 5, "payment": 4, "shipment": 3, "order": 2, "item": 1}
    for field, (aliases, allowed_domains) in checks.items():
        observations: list[tuple[str, str, Any]] = []
        for item in evidence:
            if item.domain not in allowed_domains:
                continue
            value = first_scalar(item.data, aliases)
            if value is not None:
                observations.append((item.domain, item.tool, value))
        normalized = {str(value) for _, _, value in observations}
        if len(normalized) < 2:
            continue
        selected = max(observations, key=lambda row: precedence.get(row[0], 0))
        sources = _unique(f"{domain}:{tool}"[:80] for domain, tool, _ in observations)
        if len(sources) < 2:
            continue
        results.append(
            {
                "field": field,
                "sources": sources[:5],
                "selected_source": f"{selected[0]}:{selected[1]}"[:80],
                "resolution_code": "AUTHORITATIVE_DOMAIN_PRECEDENCE",
            }
        )
    return results[:5]


def confidence(
    evidence: list[Evidence], entity_status: str, conflict_count: int, decision: Decision
) -> float:
    domains = {item.domain for item in evidence}
    score = Decimal("0.35")
    for domain in ("order", "item", "payment", "shipment", "customer", "policy"):
        if domain in domains:
            score += Decimal("0.08")
    if entity_status == "resolved":
        score += Decimal("0.10")
    elif entity_status == "ambiguous":
        score -= Decimal("0.15")
    score -= Decimal("0.08") * conflict_count
    if any(item.warnings for item in evidence):
        score -= Decimal("0.05")
    if decision.primary_issue == "insufficient_evidence":
        score = min(score, Decimal("0.48"))
    return float(max(Decimal("0.15"), min(score, Decimal("0.95"))).quantize(Decimal("0.01")))


def _policy_branch(value: Any, issue: str) -> Any | None:
    if isinstance(value, dict):
        if first_scalar(value, ("primary_issue", "issue_code", "issue")) == issue:
            return value
        for key, child in value.items():
            if normalize_key(str(key)) == normalize_key(issue):
                return child
            if isinstance(child, dict) and first_scalar(
                child, ("primary_issue", "issue_code", "issue")
            ) == issue:
                return child
        for child in value.values():
            found = _policy_branch(child, issue)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _policy_branch(child, issue)
            if found is not None:
                return found
    return None


def _responsible_party(issue: str) -> str:
    if issue == "late_delivery_seller":
        return "seller"
    if issue == "late_delivery_logistics":
        return "logistics_provider"
    if issue in {"payment_mismatch", "duplicate_charge"}:
        return "payment_provider"
    if issue in {"refund_pending", "refund_failed"}:
        return "platform"
    if issue in {"canceled_order_paid", "unavailable_order_paid"}:
        return "platform"
    return "unknown"


def _refund_amount(
    issue: str,
    captured: Decimal | None,
    refunded: Decimal | None,
    order_total: Decimal | None,
) -> Decimal:
    captured = captured or Decimal("0")
    refunded = refunded or Decimal("0")
    refundable_issues = {
        "canceled_order_paid",
        "unavailable_order_paid",
        "refund_pending",
        "refund_failed",
    }
    if issue in refundable_issues:
        amount = max(captured - refunded, Decimal("0"))
    elif issue in {"duplicate_charge", "payment_mismatch"} and order_total is not None:
        amount = max(captured - order_total - refunded, Decimal("0"))
    else:
        amount = Decimal("0")
    return amount.quantize(Decimal("0.01"))


def _default_actions(issue: str) -> tuple[str, ...]:
    return {
        "canceled_order_paid": ("issue_refund", "close_canceled_order"),
        "unavailable_order_paid": ("issue_refund", "review_inventory"),
        "late_delivery_seller": ("contact_seller", "monitor_delivery"),
        "late_delivery_logistics": ("contact_logistics_provider", "monitor_delivery"),
        "payment_mismatch": ("investigate_payment", "refund_overcharge"),
        "duplicate_charge": ("reverse_duplicate_charge",),
        "refund_pending": ("monitor_refund",),
        "refund_failed": ("retry_refund", "escalate_payment_provider"),
        "valid_split_payment": ("close_case_no_action",),
        "unsupported_claim": ("close_case_no_action",),
        "insufficient_evidence": ("manual_investigation",),
    }[issue]


def _truthy(data: Iterable[Any], aliases: Iterable[str]) -> bool:
    for item in data:
        for value in values_for(item, aliases):
            if value is True or (isinstance(value, str) and value.lower() in {"true", "yes", "1"}):
                return True
    return False


def _first_time(data: Iterable[Any], aliases: Iterable[str]) -> datetime | None:
    for item in data:
        for value in values_for(item, aliases):
            parsed = parse_time(value)
            if parsed is not None:
                return parsed
    return None


def _is_after(left: datetime, right: datetime) -> bool:
    if (left.tzinfo is None) != (right.tzinfo is None):
        left = left.replace(tzinfo=None)
        right = right.replace(tzinfo=None)
    return left > right


def _explicit_or_sum(
    data: Iterable[Any],
    total_aliases: Iterable[str],
    rows: list[dict[str, Any]],
    row_aliases: Iterable[str],
    *,
    extra_aliases: Iterable[str] = (),
) -> Decimal | None:
    for item in data:
        explicit = money(item, total_aliases)
        if explicit is not None:
            return explicit
    amounts: list[Decimal] = []
    for row in rows:
        amount = money(row, row_aliases)
        if amount is None:
            continue
        for alias in extra_aliases:
            amount += money(row, (alias,)) or Decimal("0")
        amounts.append(amount)
    return sum(amounts, Decimal("0")) if amounts else None


def _strings_from_many(data: Iterable[Any], aliases: Iterable[str]) -> list[str]:
    return [value for item in data for value in scalar_strings(item, aliases)]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _float_or_none(value: Decimal | None) -> float | None:
    return float(value.quantize(Decimal("0.01"))) if value is not None else None
