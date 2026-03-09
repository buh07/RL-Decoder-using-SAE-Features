#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _load(path: str) -> Dict[str, Any]:
    return json.load(open(path))


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


def _domain_block(eval_json: Dict[str, Any], stress_json: Dict[str, Any], eval_path: str, stress_path: str) -> Dict[str, Any]:
    cv = dict(eval_json.get("cv_diagnostics") or {})
    ss = dict(eval_json.get("single_split") or {})
    gate = dict(eval_json.get("claim_gate") or {})
    leakage_clean = bool(
        int(cv.get("cv_pair_overlap_count", 0) or 0) == 0
        and int(cv.get("cv_trace_overlap_count", 0) or 0) == 0
    )
    return {
        "eval_path": str(eval_path),
        "stress_path": str(stress_path),
        "strict_gate_pass": bool(gate.get("strict_gate_pass", False)),
        "leakage_clean": bool(leakage_clean),
        "stress_primary_pass": str(stress_json.get("final_verdict_primary", "")) == "pass",
        "cv_primary_pooled_auroc": cv.get("cv_primary_pooled_auroc"),
        "cv_wrong_intermediate_pooled_auroc": cv.get("cv_wrong_intermediate_pooled_auroc"),
        "cv_wrong_intermediate_pooled_ci95": cv.get("cv_wrong_intermediate_pooled_ci95"),
        "lexical_probe_auroc": ss.get("lexical_probe_auroc"),
        "wrong_minus_lexical_delta": ss.get("wrong_minus_lexical_delta"),
        "stress_empirical_p_value": ((stress_json.get("permutation") or {}).get("empirical_p_value")),
        "stress_regularization_pass": bool(((stress_json.get("regularization") or {}).get("regularization_pass", False))),
        "stress_multiseed_pass": bool(((stress_json.get("multiseed") or {}).get("multiseed_pass", False))),
    }


def main() -> None:
    args = parse_args()
    pr_eval = _load(args.pronto_eval)
    en_eval = _load(args.entail_eval)
    pr_stress = _load(args.pronto_stress)
    en_stress = _load(args.entail_stress)

    pronto = _domain_block(pr_eval, pr_stress, args.pronto_eval, args.pronto_stress)
    entail = _domain_block(en_eval, en_stress, args.entail_eval, args.entail_stress)

    publishable = bool(
        pronto["strict_gate_pass"]
        and entail["strict_gate_pass"]
        and pronto["leakage_clean"]
        and entail["leakage_clean"]
        and pronto["stress_primary_pass"]
        and entail["stress_primary_pass"]
    )
    out = {
        "schema_version": "phase7_g2_cross_task_stress_validated_v1",
        "status": "ok",
        "run_id": str(args.run_id),
        "decision_rule": {
            "publishable_cross_domain": "both strict eval gates pass AND both stress primary verdicts pass AND leakage clean",
        },
        "domains": {
            "prontoqa": pronto,
            "entailmentbank": entail,
        },
        "publishability_gate": {
            "policy": "both_domains_must_pass",
            "publishable_cross_domain_pass": bool(publishable),
            "final_recommendation": ("publishable_cross_domain" if publishable else "partial_generalization_or_fail"),
            "status": ("pass" if publishable else "fail"),
        },
        "timestamp": datetime.now().isoformat(),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    md = [
        "# Track C G2 Cross-Task Stress-Validated Decision",
        "",
        f"- Run: `{args.run_id}`",
        f"- PrOntoQA strict gate: `{pronto['strict_gate_pass']}`",
        f"- PrOntoQA stress primary: `{pronto['stress_primary_pass']}`",
        f"- EntailmentBank strict gate: `{entail['strict_gate_pass']}`",
        f"- EntailmentBank stress primary: `{entail['stress_primary_pass']}`",
        f"- Publishability gate: `{out['publishability_gate']['publishable_cross_domain_pass']}`",
        f"- Final recommendation: `{out['publishability_gate']['final_recommendation']}`",
    ]
    Path(args.output_md).write_text("\n".join(md) + "\n")
    print(str(out_path))


if __name__ == "__main__":
    main()
