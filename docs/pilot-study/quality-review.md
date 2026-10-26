# Answer and citation review

This review uses the frozen selected 18-cell pilot. All 180 measured responses are transport-successful and schema-valid. The same ten event IDs recur across configurations. The dataset's separate consistency oracle was checked against archived expected decisions, limits and rule IDs for every dataset event. The archived answer scores were then independently recomputed.

## Disjoint outcomes

| Outcome | Responses |
| --- | ---: |
| Correct decision/limit and exact citation set | 36 |
| Correct decision/limit but missing or unrelated citations | 27 |
| Incorrect event/decision/limit (may also have citation errors) | 117 |

This is a small synthetic pilot, not a representative error-frequency estimate. Free-text reasons are required to be nonempty but are not graded for semantic faithfulness. Source-rule presence and answer correctness are different measurements.

## A correct decision with an omitted mandatory citation

Cell: `pilot-enterprise-compiled_context-c1`; case: `cross-document`; request: `20261026T010020Z-aadb8a8af8ef4685b01f5f57fecb7338/compiled_context-c1/measurement/2`.

**Event**

```json
{
  "amount_usd": 350,
  "category": "travel",
  "city": "Harbor-0001",
  "currency": "USD",
  "department": "OFFICE",
  "emergency": false,
  "employee_id": "EMP-09921",
  "event_id": "EV-000002",
  "invoice_id": "INV-0000002",
  "vendor_status": "restricted"
}
```

**Expected output**

```json
{
  "applied_rules": [
    "R_BASE",
    "L_CITY_0001",
    "R_VENDOR"
  ],
  "decision": "reject",
  "event_id": "EV-000002",
  "limit_usd": 350,
  "reason": "Vendor control overrides monetary eligibility."
}
```

**Actual output**

```json
{
  "event_id": "EV-000002",
  "decision": "reject",
  "limit_usd": 350,
  "applied_rules": ["R_VENDOR", "L_CITY_0001"],
  "reason": "The vendor is restricted, and thus must be rejected even during an emergency. The applicable limit is retained. Refer to R_VENDOR and the city row for L_CITY_0001."
}
```

Observed: decision/limit correct `True`, strict correct `False`, required context rules present `True`.

Missing response citations: ["R_BASE"]. Unrelated response citations: []. Missing context rules: [].

The rejection and limit match. R_BASE is present in the compiled context but missing from the response. This is a citation-completeness failure, not a missing-context failure.

## A wrong control applied despite full context

Cell: `pilot-enterprise-full_context-c1`; case: `buried-exception`; request: `20261025T234026Z-4a10b3524847455180a30523b4152d71/full_context-c1/measurement/1`.

**Event**

```json
{
  "amount_usd": 1000,
  "category": "travel",
  "city": "Harbor-0003",
  "currency": "USD",
  "department": "OFFICE",
  "emergency": true,
  "employee_id": "EMP-08875",
  "event_id": "EV-000003",
  "incident_id": "INC-000003",
  "invoice_id": "INV-0000003",
  "vendor_status": "clear"
}
```

**Expected output**

```json
{
  "applied_rules": [
    "R_BASE",
    "L_CITY_0003",
    "R_EMERGENCY"
  ],
  "decision": "approve",
  "event_id": "EV-000003",
  "limit_usd": 1000,
  "reason": "Appendix Q replaces the city limit with the incident travel limit."
}
```

**Actual output**

```json
{
  "event_id": "EV-000003",
  "decision": "reject",
  "limit_usd": null,
  "applied_rules": ["R_VENDOR", "R_BASE"],
  "reason": "The event involves a restricted vendor, and the emergency does not provide an exception for travel expenses. Therefore, the travel expense is rejected."
}
```

Observed: decision/limit correct `False`, strict correct `False`, required context rules present `True`.

Missing response citations: ["L_CITY_0003", "R_EMERGENCY"]. Unrelated response citations: ["R_VENDOR"]. Missing context rules: [].

The event has a clear vendor and an emergency incident. Expected precedence approves the incident claim at USD 1,000. The model instead rejects with a null limit and cites the vendor control. Required rules are present in context, so this is not explained by retrieval omission.

## Retrieval misses required controls

Cell: `pilot-enterprise-retrieval-c1`; case: `cross-document`; request: `20261026T003617Z-bb97009f52c341d68c243e5259e8f174/retrieval-c1/measurement/2`.

**Event**

```json
{
  "amount_usd": 350,
  "category": "travel",
  "city": "Harbor-0001",
  "currency": "USD",
  "department": "OFFICE",
  "emergency": false,
  "employee_id": "EMP-09921",
  "event_id": "EV-000002",
  "invoice_id": "INV-0000002",
  "vendor_status": "restricted"
}
```

**Expected output**

```json
{
  "applied_rules": [
    "R_BASE",
    "L_CITY_0001",
    "R_VENDOR"
  ],
  "decision": "reject",
  "event_id": "EV-000002",
  "limit_usd": 350,
  "reason": "Vendor control overrides monetary eligibility."
}
```

**Actual output**

```json
{
  "event_id": "EV-000002",
  "decision": "approve",
  "limit_usd": 350,
  "applied_rules": ["R_ABSTAIN", "L_CITY_0001"],
  "reason": "The event meets all the criteria for approval except for the completeness check, but since the travel category and currency are correct, and the amount_usd matches the travel limit for Harbor-0001, the event can be approved."
}
```

Observed: decision/limit correct `False`, strict correct `False`, required context rules present `False`.

Missing response citations: ["R_BASE", "R_VENDOR"]. Unrelated response citations: ["R_ABSTAIN"]. Missing context rules: ["R_BASE", "R_VENDOR"].

The event has a restricted vendor. The expected rejection cites R_VENDOR and R_BASE, neither of which is included in this retrieved context. The response approves. This combines evidence-selection and answer failures; a shorter prompt alone is not a quality improvement.

## A fully correct compiled response

Cell: `pilot-enterprise-compiled_context-c1`; case: `buried-exception`; request: `20261026T010020Z-aadb8a8af8ef4685b01f5f57fecb7338/compiled_context-c1/measurement/1`.

**Event**

```json
{
  "amount_usd": 1000,
  "category": "travel",
  "city": "Harbor-0003",
  "currency": "USD",
  "department": "OFFICE",
  "emergency": true,
  "employee_id": "EMP-08875",
  "event_id": "EV-000003",
  "incident_id": "INC-000003",
  "invoice_id": "INV-0000003",
  "vendor_status": "clear"
}
```

**Expected output**

```json
{
  "applied_rules": [
    "R_BASE",
    "L_CITY_0003",
    "R_EMERGENCY"
  ],
  "decision": "approve",
  "event_id": "EV-000003",
  "limit_usd": 1000,
  "reason": "Appendix Q replaces the city limit with the incident travel limit."
}
```

**Actual output**

```json
{
  "event_id": "EV-000003",
  "decision": "approve",
  "limit_usd": 1000,
  "applied_rules": ["R_EMERGENCY", "R_BASE", "L_CITY_0003"],
  "reason": "The emergency condition is met, which replaces the city and department limits with 1000 USD. The travel amount is at or below the applicable limit, and vendor status is clear. The city, category, currency, and amount_usd are all present and valid."
}
```

Observed: decision/limit correct `True`, strict correct `True`, required context rules present `True`.

Missing response citations: []. Unrelated response citations: []. Missing context rules: [].

Decision, limit and the complete citation set match the reference. Citation order is ignored; reason wording is not semantically scored.

## The recovered 32K cell still has answer errors

Cell: `pilot-scaling-32k-full_context-c8`; case: `normal`; request: `20261026T120742Z-c6e4cfc259f0459aa62310f6170a550f/full_context-c8/measurement/9`.

**Event**

```json
{
  "amount_usd": 205,
  "category": "travel",
  "city": "Harbor-0017",
  "currency": "USD",
  "department": "OFFICE",
  "emergency": false,
  "employee_id": "EMP-03653",
  "event_id": "EV-000000",
  "invoice_id": "INV-0000000",
  "vendor_status": "clear"
}
```

**Expected output**

```json
{
  "applied_rules": [
    "R_BASE",
    "L_CITY_0017"
  ],
  "decision": "approve",
  "event_id": "EV-000000",
  "limit_usd": 350,
  "reason": "Within the city limit."
}
```

**Actual output**

```json
{
  "event_id": "EV-000000",
  "decision": "reject",
  "limit_usd": 350,
  "applied_rules": ["R_BASE", "L_CITY_0017", "R_DEPARTMENT"],
  "reason": "The travel expense amount of 205 USD exceeds the applicable limit of 350 USD for Harbor-0017. Additionally, the expense is for the OFFICE department, which uses the lower of the city limit and 150 USD, further reducing the limit to 150 USD. Since the amount is above this reduced limit, the expense is rejected."
}
```

Observed: decision/limit correct `False`, strict correct `False`, required context rules present `True`.

Missing response citations: []. Unrelated response citations: ["R_DEPARTMENT"]. Missing context rules: [].

This response completed under the 600-second read timeout. Successful completion is separate from agreement with the expected decision and rule set.

## Interpretation and closeout

The expected citation set is an explicit benchmark contract. A decision-only analysis is retained alongside the strict score to show its effect; no expected answers or thresholds were changed after observing results. The coverage bookkeeping correction alters context-presence fields only. It does not remove extra citations, add missing response citations, or change decisions.

A prompt/model or scoring change could be a future experiment, but the current results must remain reported under the conditions that produced them. These negative quality findings do not invalidate successful transport or the measured timing observations, and neither do those operational successes establish answer reliability.
