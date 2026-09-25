from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.contracts import Contracts
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.data = {
            "get_customer_history": (
                "customer",
                {
                    "customer_unique_id": "customer-1",
                    "related_order_ids": ["order-1", "older-order"],
                },
            ),
            "get_order": (
                "order",
                {
                    "order_id": "order-1",
                    "order_status": "canceled",
                    "order_total": 100,
                },
            ),
            "get_order_items": (
                "item",
                {
                    "items": [
                        {
                            "item_id": "item-1",
                            "seller_id": "seller-1",
                            "price": 90,
                            "freight_value": 10,
                        }
                    ]
                },
            ),
            "get_product_context": ("product", {"product_id": "product-1"}),
            "get_sellers": ("seller", {"seller_ids": ["seller-1"]}),
            "get_shipment_summary": (
                "shipment",
                {
                    "shipment_id": "shipment-1",
                    "delivered_at": "2018-01-10T12:00:00+00:00",
                    "estimated_delivery_at": "2018-01-11T12:00:00+00:00",
                },
            ),
            "get_order_payments": (
                "payment",
                {
                    "payments": [
                        {
                            "payment_reference": "payment-1",
                            "payment_value": 100,
                            "status": "captured",
                        }
                    ]
                },
            ),
            "get_payment_timeline": ("payment", {"status": "captured"}),
            "get_refund_timeline": ("refund", {"refunded_total_brl": 0}),
            "get_policy": (
                "policy",
                {
                    "rules": {
                        "canceled_order_paid": {
                            "issue_code": "canceled_order_paid",
                            "recommended_refund_brl": 100,
                            "resolution_actions": ["issue_refund", "close_canceled_order"],
                        }
                    }
                },
            ),
        }

    async def list_tools(self) -> list[str]:
        return sorted(self.data)

    async def tool_schema(self, tool_name: str) -> dict[str, Any]:
        argument = {
            "get_customer_history": "customer_unique_id",
            "get_policy": "policy_version",
        }.get(tool_name, "order_id")
        return {
            "type": "object",
            "properties": {"case_id": {"type": "string"}, argument: {"type": "string"}},
            "required": ["case_id", argument],
        }

    async def call(
        self, tool_name: str, *, case_id: str, **arguments: Any
    ) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        domain, data = self.data[tool_name]
        suffix = len(self.calls)
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{suffix:024d}",
            "result_hash": "sha256:" + "a" * 64,
            "domain": domain,
            "data": data,
        }


@pytest.mark.asyncio
async def test_workflow_builds_schema_valid_auditable_output(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    contracts = Contracts(root / "contracts" / "schemas")
    trace_path = tmp_path / "trace.jsonl"
    trace = TraceWriter(trace_path, contracts)
    gateway = FakeGateway()
    case = {
        "case_id": "L3B_CASE_001",
        "customer_request": {
            "claimed_order_id": "order-1",
            "claims": [
                {"claim_id": "claim-1", "topic": "canceled_order_paid"},
                {"claim_id": "claim-2", "topic": "requested_full_refund"},
            ],
        },
        "candidate_order_ids": ["order-1", "bad-order"],
        "customer_unique_id_hint": "customer-1",
        "policy_version": "EC_POLICY_V2",
    }

    output = await solve_case(case, gateway, trace)
    contracts.validate_output(output, "test output")
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["financial_resolution"]["recommended_refund_brl"] == 100
    assert output["entity_resolution"]["resolved_order_ids"] == ["order-1"]
    assert all(case_id == case["case_id"] for _, case_id, _ in gateway.calls)

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    consumed = {
        ref
        for event in events
        if event["event_type"] == "tool_result_consumed"
        for ref in event["evidence_refs"]
    }
    assert set(output["evidence_refs"]) == consumed
    assert {"handoff", "policy_decided", "verification_completed"} <= {
        event["event_type"] for event in events
    }
