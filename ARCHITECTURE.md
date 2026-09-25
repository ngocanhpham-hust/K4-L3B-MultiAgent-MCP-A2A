# L3B Architecture Record

## 1. System overview

The implementation is an asynchronous, deterministic Python state machine. It does not call an
LLM, so the per-agent model-size constraint is satisfied by construction.

```text
Case input
   │
   ▼
Competition run bootstrap ── variant + case-set scope
   │
   ▼
Entity/customer agent ── candidate probes + customer history ──┐
   │ resolved order(s)                                         │
   ▼                                                           │
Coordinator ─┬─ Order/item agent ───────────────────────────────┤
             ├─ Shipment agent ─────────────────────────────────┤ MCP Gateway
             └─ Payment/refund agent ───────────────────────────┤ (case-scoped)
                                                                │
Specialist handoffs ──► Conflict resolver ──► Policy agent ─────┘
                                               │
                                               ▼
                                      Verifier ──► output
                                               │
                                               └──► trace.jsonl
```

Only MCP envelopes that pass the public evidence schema enter the state. `evidence_ref` values are
copied verbatim. Final output is validated again by the CLI against the L3B output schema.
Before opening MCP, the CLI provisions/refreshes `/api/v2/runs` and rejects a local bundle whose
`case_set_version` differs from the run returned by the competition service.

## 2. Agent ownership

| Actor | Input | Responsibility | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Entity/customer | case hints and candidates | Resolve the authoritative order; reject false candidates; obtain customer history | `get_order`, `get_customer_history` | resolution status, resolved/rejected IDs to coordinator |
| Coordinator | case and resolution | Assign bounded tasks and assemble specialist results | discovery only; no evidence tool | task envelopes and final output |
| Order/item | resolved order ID | Read item totals, product context and seller IDs | `get_order_items`, `get_product_context`, `get_sellers` | normalized facts to conflict resolver |
| Shipment | resolved order ID | Compare seller handoff, actual delivery and estimate | `get_shipment_summary` | shipment verdict and late seller IDs |
| Payment/refund | resolved order ID | Reconcile captures, payment events and refunds | `get_order_payments`, `get_payment_timeline`, `get_refund_timeline` | monetary totals and payment verdict |
| Policy | preliminary issue and policy version | Apply the authoritative issue-specific rule | `get_policy` | responsibility, refund and action override |
| Conflict resolver | specialist evidence | Detect same-field disagreement and apply source precedence | no direct tool; consumes specialist evidence | conflicts and selected source to policy |
| Verifier | candidate output | Enforce cross-field, amount, uniqueness and bound invariants | none | verified output to coordinator |

Tool discovery is not treated as permission. Each actor receives only the aliases listed above.
Unknown tools are ignored.

## 3. Entity resolution and A2A protocol

Messages are represented by observable trace events and in-memory typed facts, correlated by the
input `case_id`.

1. An explicit `claimed_order_id` is probed with `get_order`; a valid case-scoped envelope confirms
   it. Candidate IDs are capped at five probes.
2. Customer history is used to narrow candidates when a customer hint is present.
3. If exactly one candidate returns evidence it is selected. If several return evidence, equal-name
   input/evidence facts are scored. A unique positive winner is selected; ties remain `ambiguous`.
4. Rejected candidates are retained in output. Ambiguous/not-found resolution never becomes a
   fabricated order.

Each assignment uses `task_assigned`; each specialist completion uses `handoff`. The fixed graph has
no back edge, so handoff loops cannot occur. Each MCP operation has a 45-second client deadline and
at most one transient retry.

## 4. Evidence and conflict lifecycle

- The gateway validates every response against `mcp-evidence-response-v1.schema.json`.
- Evidence is cached by `(tool_name, arguments)` inside one case only.
- A successful response is immediately recorded as `tool_result_consumed` by the consuming actor.
- Output and claim-level references are selected only from consumed responses in the same case.
- Monetary authority is `refund > payment > order > item`; timeline authority is
  `shipment > order > item`. Conflicting observations are emitted in `data_conflicts` with
  `AUTHORITATIVE_DOMAIN_PRECEDENCE`.
- A conflict or MCP warning lowers confidence. Unresolved entity resolution caps the conclusion at
  `insufficient_evidence`/`needs_investigation`.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Observable result |
| --- | ---: | --- | --- |
| Timeout, HTTP 429/502/503/504 | 1 | Continue with remaining authoritative evidence | missing domain lowers confidence |
| Permanent MCP/tool error | 0 | Do not invent an envelope or value | relevant analysis becomes insufficient |
| Entity not found/ambiguous | 0 | No downstream order-scoped calls | `not_found`/`ambiguous` |
| Source conflict | 0 | Apply domain precedence; retain conflict | `data_conflicts` entry |
| Invalid specialist result | 0 | Exclude it before policy/verifier | schema/invariant check prevents finalize |

Efficiency controls are: one cached discovery response per gateway session, one case-local call cache,
maximum five candidate probes, maximum three resolved orders, a single scoped call per evidence domain
and order, and no speculative cross-order product/seller scan. Refund evidence is fetched only when the
case or earlier evidence mentions a cancellation, refund, unavailable item, or duplicate payment.

## 6. Verification invariants

Before returning, the verifier checks or normalizes:

- case ID remains unchanged and resolved/rejected candidates are disjoint;
- all output evidence refs came from case-scoped consumed envelopes;
- IDs and evidence refs are unique and respect schema limits;
- payment and refund values are non-negative and rounded to BRL cents;
- refund lines sum to `recommended_refund_brl`;
- `no_action` always has zero refund and no refund lines;
- seller responsibility uses a late seller ID when one is known;
- confidence stays below 1.0 and decreases for warnings, conflicts and missing evidence;
- actions are unique and bounded; and
- the public JSON Schema is the final authority in `day09 run` and `day09 validate`.

## 7. Reproducibility

- Runtime: Python 3.11+, pure async state machine.
- Model: none (0 parameters); no model provider or hidden prompt.
- Dependencies: bounded major versions in `pyproject.toml`.
- Concurrency: cases and evidence calls run sequentially to preserve deterministic scope and avoid
  gateway bursts.
- Randomness: no business-logic randomness; trace event IDs and timestamps are intentionally unique.
- Commands: `pytest -q`, `day09 validate-inputs`, `day09 mcp-tools`, `day09 run`,
  `day09 validate`, and `day09 package --output dist/submission.zip`.
- Secrets: loaded only from ignored `.env`; never written to output, trace, architecture, or ZIP.
