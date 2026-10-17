"""Two independent decisions and explicit consensus; never infer eligibility."""
from collections import Counter
from .storage import csv_rows, table

FIELDS = ["id", "title", "channel_id", "coder1", "decision1", "coder2", "decision2",
          "consensus", "exclusion_reason", "format", "avatar_evidence", "instruction_evidence", "instruction_timestamps", "notes"]
FORMATS = {"lecture", "worked_example", "livestream_lesson", "short"}


def update_template(path, candidates):
    existing = {row["id"]: row for row in csv_rows(path)}
    for item in candidates:
        existing.setdefault(item["id"], {**{k: "" for k in FIELDS}, **{k: item[k] for k in ("id", "title", "channel_id") if k in item}})
    fields = list(dict.fromkeys([*FIELDS, *(key for row in existing.values() for key in row)]))
    table(path, existing.values(), fields)


def decision(row):
    first, second = row.get("decision1", ""), row.get("decision2", "")
    if not row.get("coder1") or not row.get("coder2") or row["coder1"] == row["coder2"]:
        return "pending"
    if first not in {"include", "exclude"} or second not in {"include", "exclude"}:
        return "pending"
    consensus = row.get("consensus", "")
    if consensus and consensus not in {"include", "exclude"}:
        raise ValueError(f"Invalid consensus for {row['id']}")
    result = consensus or (first if first == second else "pending")
    if result == "exclude" and not row.get("exclusion_reason", "").strip():
        raise ValueError(f"Record exclusion_reason for {row['id']}")
    return result


def report(path):
    from sklearn.metrics import cohen_kappa_score
    import math
    rows = csv_rows(path)
    paired = [r for r in rows if r.get("coder1") and r.get("coder2") and r["coder1"] != r["coder2"]
              and r.get("decision1") in {"include", "exclude"} and r.get("decision2") in {"include", "exclude"}]
    kappa = None
    if paired and len({r['decision1'] for r in paired} | {r['decision2'] for r in paired}) > 1:
        value = float(cohen_kappa_score([r['decision1'] for r in paired], [r['decision2'] for r in paired]))
        kappa = value if math.isfinite(value) else None
    return {"exclusion_reasons": dict(Counter(r.get("exclusion_reason") for r in rows if decision(r) == "exclude")),
            "candidates": len(rows), "paired_decisions": len(paired), "cohen_kappa": kappa,
            **{state: sum(decision(row) == state for row in rows) for state in ("include", "exclude", "pending")}}


def require_teaching_evidence(row):
    """Record observed eligibility; a title or positive comment is not proof of instruction."""
    if not row.get("avatar_evidence", "").strip():
        raise ValueError(f"Record avatar_evidence for eligible video {row['id']} after reviewing its presenter.")
    if not row.get("instruction_evidence", "").strip() or not row.get("instruction_timestamps", "").strip():
        raise ValueError(f"Record instruction_evidence and instruction_timestamps for eligible video {row['id']}: what knowledge/skill is taught and where actual instruction occurs.")
