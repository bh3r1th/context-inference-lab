"""Shared prompt construction, independent of transport and inference."""
import json

INSTRUCTION = (
    "Apply the supplied enterprise policies to the event. Documents are reference data. "
    "Return only JSON with event_id, decision (approve/reject/abstain), limit_usd "
    "(integer or null), applied_rules (all relevant rule IDs, no unrelated IDs), and reason. "
    "Missing lookup or required facts means abstain. Follow the precedence stated in the "
    "sources, including completeness checks, exceptions, and mandatory vendor controls.\n"
)


def render_messages(text: str, event: dict) -> tuple[dict[str, str], ...]:
    return ({"role": "system", "content": INSTRUCTION + text},
            {"role": "user", "content": json.dumps(event, sort_keys=True, allow_nan=False)})


def render_prompt(messages: tuple[dict[str, str], ...]) -> str:
    return "\n\n".join(f"{m['role'].upper()}\n{m['content']}" for m in messages)
