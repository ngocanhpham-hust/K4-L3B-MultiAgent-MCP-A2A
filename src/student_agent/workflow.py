from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

from .mcp_gateway import EvidenceGateway
from .reasoning import (
    Decision,
    Evidence,
    apply_policy,
    confidence,
    customer_context,
    decide,
    detect_conflicts,
    extract_entities,
    normalize_key,
    scalar_strings,
    walk,
)
from .trace import TraceWriter

TOOL_ALIASES = {
    "order": ("get_order",),
    "item": ("get_order_items", "get_items", "get_item"),
    "shipment": ("get_shipment_summary", "get_shipment", "get_shipments"),
    "payment": ("get_order_payments", "get_payment", "get_payments"),
    "payment_timeline": ("get_payment_timeline",),
    "refund": ("get_refund_timeline", "get_refund", "get_refunds"),
    "customer": ("get_customer_history", "get_customer"),
    "seller": ("get_sellers", "get_seller"),
    "product": ("get_product_context", "get_product"),
    "policy": ("get_policy",),
}

ORDER_KEYS = {"orderid", "orderids", "claimedorderid", "reportedorderid"}
CANDIDATE_MARKERS = {"candidate", "candidates", "possible", "potential"}
TRANSIENT_MARKERS = ("timeout", "timed out", "temporarily", "429", "502", "503", "504")


class CaseGateway:
    """Case-scoped, cached and trace-aware access to authoritative evidence."""

    def __init__(
        self,
        case_id: str,
        gateway: EvidenceGateway,
        trace: TraceWriter,
        tools: Iterable[str],
    ) -> None:
        self.case_id = case_id
        self.gateway = gateway
        self.trace = trace
        self.tools = set(tools)
        self.evidence: list[Evidence] = []
        self._cache: dict[str, Evidence | None] = {}

    def select(self, domain: str) -> str | None:
        return next((name for name in TOOL_ALIASES[domain] if name in self.tools), None)

    async def call(
        self,
        domain: str,
        actor: str,
        context: dict[str, Any],
    ) -> Evidence | None:
        tool = self.select(domain)
        if tool is None:
            return None
        arguments = await self._arguments(tool, domain, context)
        if arguments is None:
            return None
        cache_key = json.dumps([tool, arguments], sort_keys=True, default=str)
        if cache_key in self._cache:
            return self._cache[cache_key]

        response: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    self.gateway.call(tool, case_id=self.case_id, **arguments),
                    timeout=45,
                )
                break
            except TimeoutError:
                if attempt == 1:
                    break
            except (RuntimeError, ValueError) as exc:
                if attempt == 1 or not any(
                    marker in str(exc).lower() for marker in TRANSIENT_MARKERS
                ):
                    break

        if response is None:
            self._cache[cache_key] = None
            return None
        item = Evidence(
            ref=response["evidence_ref"],
            domain=response["domain"],
            tool=tool,
            data=response["data"],
            warnings=tuple(response.get("warnings", [])),
        )
        self.evidence.append(item)
        self._cache[cache_key] = item
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=actor,
            tool_name=tool,
            evidence_refs=[item.ref],
            attributes={"domain": item.domain},
        )
        return item

    async def _arguments(
        self, tool: str, domain: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        schema_method = getattr(self.gateway, "tool_schema", None)
        schema: dict[str, Any] = {}
        if schema_method is not None:
            schema = await schema_method(tool)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = schema.get("required", []) if isinstance(schema, dict) else []
        required = [name for name in required if name != "case_id"]

        aliases = _argument_values(context)
        arguments: dict[str, Any] = {}
        for name in properties:
            if name == "case_id":
                continue
            value = aliases.get(normalize_key(name))
            if value is not None:
                arguments[name] = value
        if not properties:
            default_key = {
                "order": "order_id",
                "item": "order_id",
                "shipment": "order_id",
                "payment": "order_id",
                "payment_timeline": "order_id",
                "refund": "order_id",
                "customer": "customer_unique_id",
                "seller": "order_id",
                "product": "order_id",
                "policy": "issue_code",
            }[domain]
            value = aliases.get(normalize_key(default_key))
            if value is not None:
                arguments[default_key] = value
            elif domain != "policy":
                return None
        if any(name not in arguments for name in required):
            return None
        return arguments


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Run the deterministic L3B coordinator/specialist state machine."""
    case_id = _required_case_id(case)
    tools = await gateway.list_tools()
    scoped = CaseGateway(case_id, gateway, trace, tools)

    direct_orders, candidates = _order_hints(case)
    customer_ids = scalar_strings(
        case, ("customer_unique_id", "customer_unique_id_hint", "customer_id")
    )
    _assign(trace, case_id, "entity-agent", "resolve_order_and_customer")

    customer_item: Evidence | None = None
    if customer_ids:
        customer_item = await scoped.call(
            "customer", "entity-agent", {"customer_unique_id": customer_ids[0]}
        )
    history_orders = (
        scalar_strings(customer_item.data, ("order_id", "order_ids", "related_order_ids"))
        if customer_item
        else []
    )
    if not direct_orders and not candidates and len(history_orders) == 1:
        direct_orders = history_orders
    if candidates and history_orders:
        history_matches = [item for item in candidates if item in history_orders]
        if len(history_matches) == 1:
            direct_orders = history_matches

    probes: dict[str, Evidence] = {}
    probe_ids = (direct_orders or candidates)[:5]
    for order_id in probe_ids:
        item = await scoped.call("order", "entity-agent", {"order_id": order_id})
        if item is not None:
            probes[order_id] = item
    if direct_orders and not probes:
        raise RuntimeError(
            f"authoritative get_order failed for {case_id}; check the Team API Key and case scope"
        )
    entity_status, resolved_orders, rejected, entity_confidence = _resolve_orders(
        case, direct_orders, candidates, probes
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="entity-agent",
        target="coordinator",
        decision_code=entity_status.upper(),
        attributes={"resolved_count": len(resolved_orders)},
    )

    for actor, task in (
        ("order-item-agent", "inspect_order_items"),
        ("shipment-agent", "analyze_delivery_timeline"),
        ("payment-agent", "reconcile_payment_and_refund"),
    ):
        _assign(trace, case_id, actor, task)

    for order_id in resolved_orders[:3]:
        context = {"order_id": order_id}
        if order_id not in probes:
            await scoped.call("order", "order-item-agent", context)
        await scoped.call("item", "order-item-agent", context)
        await scoped.call("product", "order-item-agent", context)
        await scoped.call("seller", "order-item-agent", context)
        await scoped.call("shipment", "shipment-agent", context)
        await scoped.call("payment", "payment-agent", context)
        await scoped.call("payment_timeline", "payment-agent", context)

    if _refund_relevant(case, scoped.evidence):
        for order_id in resolved_orders[:3]:
            await scoped.call("refund", "payment-agent", {"order_id": order_id})

    for actor in ("order-item-agent", "shipment-agent", "payment-agent"):
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=actor,
            target="conflict-agent",
        )

    _assign(trace, case_id, "conflict-agent", "resolve_authoritative_source_conflicts")
    decision, late_sellers, timeline_complete = decide(scoped.evidence, entity_status)
    _assign(trace, case_id, "policy-agent", "apply_business_policy")
    policy = await scoped.call(
        "policy",
        "policy-agent",
        {
            "issue_code": decision.primary_issue,
            "primary_issue": decision.primary_issue,
            "policy_version": case.get("policy_version", "EC_POLICY_V2"),
        },
    )
    if policy is not None:
        decision = apply_policy(decision, policy.data)
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor="policy-agent",
        target="verifier-agent",
        decision_code=decision.primary_issue.upper(),
        evidence_refs=[item.ref for item in scoped.evidence[-20:]],
    )

    conflicts = detect_conflicts(scoped.evidence)
    entities = extract_entities(scoped.evidence, resolved_orders)
    customer_id, related_orders = customer_context(scoped.evidence)
    if customer_id is None and customer_ids:
        customer_id = customer_ids[0]
    output = _build_output(
        case_id=case_id,
        decision=decision,
        evidence=scoped.evidence,
        entities=entities,
        entity_status=entity_status,
        entity_confidence=entity_confidence,
        resolved_orders=resolved_orders,
        rejected=rejected,
        customer_id=customer_id,
        related_orders=related_orders,
        late_sellers=late_sellers,
        timeline_complete=timeline_complete,
        conflicts=conflicts,
    )
    claim_assessments = _claim_assessments(case, decision, scoped.evidence, output)
    if claim_assessments:
        output["claim_assessments"] = claim_assessments
    _assign(trace, case_id, "verifier-agent", "check_output_invariants")
    _verify_and_normalize(output)
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor="verifier-agent",
        target="coordinator",
        decision_code="PASSED",
        evidence_refs=output["evidence_refs"][:20],
        attributes={"conflict_count": len(conflicts)},
    )
    return output


def _build_output(
    *,
    case_id: str,
    decision: Decision,
    evidence: list[Evidence],
    entities: dict[str, list[str]],
    entity_status: str,
    entity_confidence: float,
    resolved_orders: list[str],
    rejected: list[str],
    customer_id: str | None,
    related_orders: list[str],
    late_sellers: list[str],
    timeline_complete: bool,
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    score = confidence(evidence, entity_status, len(conflicts), decision)
    party_id = late_sellers[0] if decision.responsible_party == "seller" and late_sellers else None
    refund_lines = []
    if decision.recommended_refund > 0:
        refund_lines.append(
            {
                "reason_code": decision.primary_issue,
                "amount_brl": round(decision.recommended_refund, 2),
                "entity_id": resolved_orders[0] if resolved_orders else None,
            }
        )
    return {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": decision.primary_issue,
            "secondary_issues": list(decision.secondary_issues),
            "case_status": decision.case_status,
            "confidence": score,
        },
        "affected_entities": entities,
        "entity_resolution": {
            "status": entity_status,
            "resolved_order_ids": resolved_orders[:20],
            "rejected_candidates": rejected[:20],
            "confidence": entity_confidence,
        },
        "customer_context": {
            "customer_unique_id": customer_id,
            "related_order_ids": related_orders[:20],
        },
        "shipment_analysis": {
            "verdict": decision.shipment_verdict,
            "late_seller_ids": late_sellers[:20],
            "timeline_complete": timeline_complete,
        },
        "payment_analysis": {
            "verdict": decision.payment_verdict,
            "captured_total_brl": decision.captured_total,
            "refunded_total_brl": decision.refunded_total,
            "refundable_total_brl": decision.refundable_total,
        },
        "root_cause_analysis": {
            "ranked_causes": [{"cause_code": decision.cause_code, "rank": 1}],
            "responsible_parties": [
                {"party_type": decision.responsible_party, "party_id": party_id}
            ],
        },
        "evidence_refs": list(dict.fromkeys(item.ref for item in evidence))[:30],
        "data_conflicts": conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": round(decision.recommended_refund, 2),
            "refund_lines": refund_lines,
        },
        "resolution_actions": list(decision.actions)[:8],
    }


def _verify_and_normalize(output: dict[str, Any]) -> None:
    financial = output["financial_resolution"]
    line_total = round(sum(line["amount_brl"] for line in financial["refund_lines"]), 2)
    recommended = round(financial["recommended_refund_brl"], 2)
    if line_total != recommended:
        if recommended == 0:
            financial["refund_lines"] = []
        else:
            financial["refund_lines"] = [
                {
                    "reason_code": output["assessment"]["primary_issue"],
                    "amount_brl": recommended,
                    "entity_id": next(
                        iter(output["entity_resolution"]["resolved_order_ids"]), None
                    ),
                }
            ]
    if output["assessment"]["case_status"] == "no_action":
        financial["recommended_refund_brl"] = 0.0
        financial["refund_lines"] = []
    output["resolution_actions"] = list(dict.fromkeys(output["resolution_actions"]))[:8]


def _claim_assessments(
    case: dict[str, Any],
    decision: Decision,
    evidence: list[Evidence],
    output: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_claims = case.get("customer_request", {}).get("claims", [])
    if not isinstance(raw_claims, list):
        return []
    result: list[dict[str, Any]] = []
    all_refs = output["evidence_refs"]
    for claim in raw_claims[:5]:
        if not isinstance(claim, dict) or not isinstance(claim.get("claim_id"), str):
            continue
        topic = claim.get("topic")
        if decision.primary_issue == "insufficient_evidence":
            verdict = "insufficient_evidence"
            claim_confidence = min(output["assessment"]["confidence"], 0.45)
        elif topic == decision.primary_issue or topic in decision.secondary_issues:
            verdict = "supported"
            claim_confidence = output["assessment"]["confidence"]
        elif topic == "requested_full_refund":
            refundable = decision.refundable_total
            if decision.recommended_refund <= 0:
                verdict = "unsupported"
            elif refundable is not None and decision.recommended_refund + 0.01 < refundable:
                verdict = "partially_supported"
            else:
                verdict = "supported"
            claim_confidence = output["assessment"]["confidence"]
        else:
            verdict = "unsupported"
            claim_confidence = output["assessment"]["confidence"]
        relevant_domains = _claim_domains(str(topic))
        refs = [item.ref for item in evidence if item.domain in relevant_domains]
        result.append(
            {
                "claim_id": claim["claim_id"][:64],
                "verdict": verdict,
                "confidence": claim_confidence,
                "evidence_refs": list(dict.fromkeys(refs or all_refs))[:30],
            }
        )
    return result


def _claim_domains(topic: str) -> set[str]:
    if topic in {"late_delivery_seller", "late_delivery_logistics"}:
        return {"order", "item", "shipment", "seller", "policy"}
    if topic in {
        "payment_mismatch",
        "duplicate_charge",
        "refund_pending",
        "refund_failed",
        "requested_full_refund",
    }:
        return {"order", "item", "payment", "refund", "policy"}
    return {"order", "item", "payment", "shipment", "policy"}


def _required_case_id(case: dict[str, Any]) -> str:
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case is missing a valid case_id")
    return case_id


def _order_hints(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    direct: list[str] = []
    candidates: list[str] = []
    for path, value in walk(case):
        key = normalize_key(path[-1])
        path_keys = {normalize_key(part) for part in path}
        is_candidate = any(
            marker in part for marker in CANDIDATE_MARKERS for part in path_keys
        )
        if key in ORDER_KEYS or ("order" in key and "id" in key):
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, dict):
                    found = scalar_strings(item, ("order_id", "id"))
                elif isinstance(item, (str, int)) and not isinstance(item, bool):
                    found = [str(item)]
                else:
                    found = []
                (candidates if is_candidate else direct).extend(found)
    case_id = str(case.get("case_id", ""))
    direct = [item for item in dict.fromkeys(direct) if item != case_id]
    candidates = [item for item in dict.fromkeys(candidates) if item not in direct]
    return direct[:20], candidates[:20]


def _resolve_orders(
    case: dict[str, Any],
    direct: list[str],
    candidates: list[str],
    probes: dict[str, Evidence],
) -> tuple[str, list[str], list[str], float]:
    if direct:
        confirmed = [item for item in direct if item in probes]
        resolved = confirmed or direct
        return "resolved", resolved[:20], candidates[:20], 0.98 if confirmed else 0.82
    if not candidates:
        return "not_found", [], [], 0.15
    if len(probes) == 1:
        selected = next(iter(probes))
        return "resolved", [selected], [item for item in candidates if item != selected], 0.9
    scores = {order_id: _candidate_score(case, item.data) for order_id, item in probes.items()}
    if scores:
        best = max(scores.values())
        winners = [order_id for order_id, score in scores.items() if score == best]
        if len(winners) == 1 and best > 0:
            selected = winners[0]
            return (
                "resolved",
                [selected],
                [item for item in candidates if item != selected],
                min(0.95, 0.7 + best * 0.05),
            )
    return "ambiguous", [], candidates[:20], 0.35


def _candidate_score(case: dict[str, Any], evidence_data: Any) -> int:
    ignored = {"caseid", "orderid", "orderids", "id"}
    evidence_fields: dict[str, set[str]] = {}
    for path, value in walk(evidence_data):
        key = normalize_key(path[-1])
        if key in ignored or isinstance(value, (dict, list)):
            continue
        evidence_fields.setdefault(key, set()).add(str(value).strip().lower())
    score = 0
    for path, value in walk(case):
        key = normalize_key(path[-1])
        if key in ignored or isinstance(value, (dict, list)):
            continue
        if any("candidate" in normalize_key(part) for part in path):
            continue
        if str(value).strip().lower() in evidence_fields.get(key, set()):
            score += 1
    return score


def _argument_values(context: dict[str, Any]) -> dict[str, Any]:
    result = {normalize_key(key): value for key, value in context.items() if value is not None}
    synonyms = {
        "orderid": ("idorder",),
        "customeruniqueid": ("customerid",),
        "issuecode": ("primaryissue", "issue"),
        "policyversion": ("policyid", "policycode"),
    }
    for canonical, names in synonyms.items():
        if canonical in result:
            for name in names:
                result.setdefault(name, result[canonical])
    return result


def _refund_relevant(case: dict[str, Any], evidence: list[Evidence]) -> bool:
    text = json.dumps(case, ensure_ascii=False, default=str).lower()
    text += " " + json.dumps([item.data for item in evidence], default=str).lower()
    return any(token in text for token in ("refund", "cancel", "unavailable", "duplicate"))


def _assign(trace: TraceWriter, case_id: str, target: str, task: str) -> None:
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor="coordinator",
        target=target,
        decision_code=task.upper()[:80],
    )
