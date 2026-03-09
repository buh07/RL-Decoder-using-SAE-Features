#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _leakage_clean(eval_payload: Dict[str, Any]) -> bool:
    cv = eval_payload.get("cv_diagnostics") or {}
    pair_ov = int(cv.get("cv_pair_overlap_count", -1))
    trace_ov = int(cv.get("cv_trace_overlap_count", -1))
    return pair_ov == 0 and trace_ov == 0


def _strict_gate_pass(eval_payload: Dict[str, Any]) -> bool:
    gate = eval_payload.get("claim_gate") or {}
    return bool(gate.get("strict_gate_pass") is True)


def _stress_primary_pass(stress_payload: Dict[str, Any]) -> bool:
    return str(stress_payload.get("final_verdict_primary", "")).strip().lower() == "pass"


def _domain_block(eval_path: str, stress_path: str) -> Dict[str, Any]:
    ev = _load_json(eval_path)
    st = _load_json(stress_path)
    cv = ev.get("cv_diagnostics") or {}
    single = ev.get("single_split") or {}
    return {
        "eval_path": str(eval_path),
        "stress_path": str(stress_path),
        "strict_gate_pass": _strict_gate_pass(ev),
        "leakage_clean": _leakage_clean(ev),
        "stress_primary_pass": _stress_primary_pass(st),
        "cv_primary_pooled_auroc": cv.get("cv_primary_pooled_auroc"),
        "cv_wrong_intermediate_pooled_auroc": cv.get("cv_wrong_intermediate_pooled_auroc"),
        "cv_wrong_intermediate_pooled_ci95": cv.get("cv_wrong_intermediate_pooled_ci95"),
        "lexical_probe_auroc": single.get("lexical_probe_auroc"),
        "wrong_minus_lexical_delta": single.get("wrong_minus_lexical_delta"),
        "stress_empirical_p_value": (st.get("permutation") or {}).get("empirical_p_value"),
        "stress_regularization_pass": (st.get("regularization") or {}).get("regularization_pass"),
        "stress_multiseed_pass": (st.get("multiseed") or {}).get("multiseed_pass"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--pronto-eval", required=True)
    p.add_argument("--entail-eval", required=True)
    p.add_argument("--pronto-stress", required=True)
    p.add_argument("--entail-stress", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pronto = _domain_block(args.pronto_eval, args.pronto_stress)
    entail = _domain_block(args.entail_eval, args.entail_stress)

    publishable_cross_domain = bool(
        pronto["strict_gate_pass"]
        and entail["strict_gate_pass"]
        and pronto["stress_primary_pass"]
        and entail["stress_primary_pass"]
        and pronto["leakage_clean"]
        and entail["leakage_clean"]
    )
    final_recommendation = "publishable_cross_domain" if publishable_cross_domain else "partial_generalization_or_fail"

    out = {
        "schema_version": "phase7_g2_cross_task_stress_validated_v1",
        "status": "ok",
        "run_id": str(args.run_id),
        "decision_rule": {
            "publishable_cross_domain": "both domains strict gate pass + both stress primary pass + leakage clean in both evals",
        },
        "domains": {
            "prontoqa": pronto,
            "entailmentbank": entail,
        },
        "publishability_gate": {
            "policy": "both_domains_must_pass",
            "publishable_cross_domain_pass": publishable_cross_domain,
            "final_recommendation": final_recommendation,
            "status": "pass" if publishable_cross_domain else "fail",
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Track C G2 Cross-Task Stress-Validated Decision",
        "",
        f"- Run: `{args.run_id}`",
        f"- PrOntoQA strict gate: `{pronto['strict_gate_pass']}`",
        f"- PrOntoQA stress primary pass: `{pronto['stress_primary_pass']}`",
        f"- EntailmentBank strict gate: `{entail['strict_gate_pass']}`",
        f"- EntailmentBank stress primary pass: `{entail['stress_primary_pass']}`",
        f"- Leakage clean (both): `{pronto['leakage_clean'] and entail['leakage_clean']}`",
        f"- Final publishability: `{publishable_cross_domain}`",
        f"- Recommendation: `{final_recommendation}`",
    ]
    Path(args.output_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(out_json)


if __name__ == "__main__":
    main()
