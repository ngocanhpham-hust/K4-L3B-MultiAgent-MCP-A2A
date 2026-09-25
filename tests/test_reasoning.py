from student_agent.reasoning import Evidence, decide


def evidence(domain: str, data: object, tool: str | None = None) -> Evidence:
    return Evidence(
        ref=f"ev_{domain:0<24}",
        domain=domain,
        tool=tool or f"get_{domain}",
        data=data,
    )


def baseline(payment_data: object) -> list[Evidence]:
    return [
        evidence("order", {"order_status": "delivered", "order_total": 100}),
        evidence(
            "item",
            {
                "items": [
                    {
                        "seller_id": "seller-1",
                        "price": 90,
                        "freight_value": 10,
                        "shipping_limit_date": "2018-01-03T00:00:00+00:00",
                    }
                ]
            },
        ),
        evidence(
            "shipment",
            {
                "carrier_handoff_at": "2018-01-02T00:00:00+00:00",
                "delivered_at": "2018-01-04T00:00:00+00:00",
                "estimated_delivery_at": "2018-01-05T00:00:00+00:00",
            },
        ),
        evidence("payment", payment_data, "get_order_payments"),
    ]


def test_split_payment_is_valid_when_total_reconciles() -> None:
    facts = baseline(
        {
            "payments": [
                {"payment_reference": "pay-1", "payment_value": 40},
                {"payment_reference": "pay-2", "payment_value": 60},
            ]
        }
    )
    decision, _, _ = decide(facts, "resolved")
    assert decision.primary_issue == "valid_split_payment"
    assert decision.payment_verdict == "reconciled"
    assert decision.case_status == "no_action"


def test_overcapture_is_payment_mismatch_for_one_payment() -> None:
    facts = baseline(
        {"payments": [{"payment_reference": "pay-1", "payment_value": 120}]}
    )
    decision, _, _ = decide(facts, "resolved")
    assert decision.primary_issue == "payment_mismatch"
    assert decision.recommended_refund == 20


def test_late_handoff_assigns_seller() -> None:
    facts = baseline(
        {"payments": [{"payment_reference": "pay-1", "payment_value": 100}]}
    )
    facts[2] = evidence(
        "shipment",
        {
            "carrier_handoff_at": "2018-01-04T00:00:00+00:00",
            "delivered_at": "2018-01-07T00:00:00+00:00",
            "estimated_delivery_at": "2018-01-05T00:00:00+00:00",
        },
    )
    decision, late_sellers, _ = decide(facts, "resolved")
    assert decision.primary_issue == "late_delivery_seller"
    assert decision.responsible_party == "seller"
    assert late_sellers == ["seller-1"]
