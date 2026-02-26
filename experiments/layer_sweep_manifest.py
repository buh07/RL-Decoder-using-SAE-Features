#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

SCHEMA_VERSION = "layer_sweep_manifest_v1"
MODEL_NAME = "gpt2-medium"
LAYERS_TOTAL = 24
PARENT_BASELINE = "spread4_current"


@dataclass(frozen=True)
class LayerSetSpec:
    layer_set_id: str
    layers: tuple[int, ...]
    family: str

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def to_manifest_row(self, *, phase6_variants: Sequence[str], phase7_variants: Sequence[str], sweep_groups: Sequence[str]) -> Dict[str, Any]:
        primary_group = "broad" if "broad" in sweep_groups else "sae_panel"
        return {
            "layer_set_id": self.layer_set_id,
            "layers": list(self.layers),
            "family": self.family,
            "num_layers": self.num_layers,
            "phase6_variants": list(phase6_variants),
            "phase7_variants": list(phase7_variants),
            "sweep_group": primary_group,
            "sweep_groups": list(sweep_groups),
        }


def normalize_layers(layers: Iterable[int], *, layers_total: int = LAYERS_TOTAL) -> tuple[int, ...]:
    vals = tuple(int(x) for x in layers)
    if not vals:
        raise ValueError("Layer list cannot be empty")
    if any(x < 0 or x >= layers_total for x in vals):
        raise ValueError(f"Layer indices must be in [0,{layers_total-1}]")
    if len(set(vals)) != len(vals):
        raise ValueError(f"Layer list contains duplicates: {vals}")
    return tuple(sorted(vals))


def format_layer_tuple_slug(layers: Sequence[int]) -> str:
    return "_".join(f"l{int(x):02d}" for x in layers)


def default_phase6_config_name(input_variant: str, layers: Sequence[int]) -> str:
    ls = normalize_layers(layers)
    if len(ls) == 1:
        return f"{input_variant}_single_l{ls[0]}"
    return f"{input_variant}_custom_{format_layer_tuple_slug(ls)}"


def default_phase7_config_name(input_variant: str, layers: Sequence[int]) -> str:
    ls = normalize_layers(layers)
    return f"state_{input_variant}_custom_{format_layer_tuple_slug(ls)}"


def _single_layers() -> List[LayerSetSpec]:
    return [LayerSetSpec(f"single_l{i:02d}", (i,), "single") for i in range(LAYERS_TOTAL)]


def _block4_nonoverlap() -> List[LayerSetSpec]:
    out = []
    for start in range(0, LAYERS_TOTAL, 4):
        layers = tuple(range(start, start + 4))
        out.append(LayerSetSpec(f"block4_{start:02d}_{start+3:02d}", layers, "block4"))
    return out


def _block8_stride4() -> List[LayerSetSpec]:
    out = []
    for start in [0, 4, 8, 12, 16]:
        layers = tuple(range(start, start + 8))
        out.append(LayerSetSpec(f"block8_{start:02d}_{start+7:02d}", layers, "block8"))
    return out


def _strided_sets() -> List[LayerSetSpec]:
    return [
        LayerSetSpec("every2_even", tuple(range(0, LAYERS_TOTAL, 2)), "strided"),
        LayerSetSpec("every2_odd", tuple(range(1, LAYERS_TOTAL, 2)), "strided"),
    ]


def _middle_sets() -> List[LayerSetSpec]:
    return [
        LayerSetSpec("middle12_06_17", tuple(range(6, 18)), "middle"),
        LayerSetSpec("middle_every2_06_16", tuple(range(6, 17, 2)), "middle"),
    ]


def _full_and_spread_sets() -> List[LayerSetSpec]:
    return [
        LayerSetSpec("all24", tuple(range(LAYERS_TOTAL)), "global"),
        LayerSetSpec("spread4_current", (7, 12, 17, 22), "spread4"),
        LayerSetSpec("spread4_quartiles", (2, 8, 14, 20), "spread4"),
        LayerSetSpec("spread4_edges", (0, 7, 15, 23), "spread4"),
    ]


def build_full_library_specs() -> List[LayerSetSpec]:
    specs = _single_layers() + _block4_nonoverlap() + _block8_stride4() + _strided_sets() + _middle_sets() + _full_and_spread_sets()
    _validate_no_duplicates(specs)
    if len(specs) != 43:
        raise AssertionError(f"Expected 43 full-library specs, found {len(specs)}")
    return specs


def build_sae_panel_ids() -> List[str]:
    ids = [
        "single_l07",
        "single_l12",
        "single_l17",
        "single_l22",
        # "block4_00_03",  # dropped to keep strict count at 16
        "block4_08_11",
        "block4_12_15",
        "block4_20_23",
        "block8_00_07",
        "block8_08_15",
        "block8_16_23",
        "middle12_06_17",
        "middle_every2_06_16",
        "every2_even",
        "every2_odd",
        "all24",
        "spread4_current",
    ]
    if len(ids) != 16:
        raise AssertionError(f"Expected 16 SAE panel ids, got {len(ids)}")
    return ids


def _validate_no_duplicates(specs: Sequence[LayerSetSpec]) -> None:
    seen_ids = set()
    seen_layers = {}
    for s in specs:
        if s.layer_set_id in seen_ids:
            raise ValueError(f"Duplicate layer_set_id: {s.layer_set_id}")
        seen_ids.add(s.layer_set_id)
        if s.layers in seen_layers:
            raise ValueError(f"Duplicate layer tuple {s.layers} for {s.layer_set_id} and {seen_layers[s.layers]}")
        seen_layers[s.layers] = s.layer_set_id


def build_manifest_payload() -> Dict[str, Any]:
    specs = build_full_library_specs()
    by_id = {s.layer_set_id: s for s in specs}
    sae_panel = set(build_sae_panel_ids())

    rows: List[Dict[str, Any]] = []
    for s in specs:
        sweep_groups = ["broad"]
        phase6_variants = ["raw", "hybrid"]
        phase7_variants = ["raw", "hybrid"]
        if s.layer_set_id in sae_panel:
            sweep_groups.append("sae_panel")
            phase6_variants = ["raw", "hybrid", "sae"]
            phase7_variants = ["raw", "hybrid", "sae"]
        rows.append(s.to_manifest_row(phase6_variants=phase6_variants, phase7_variants=phase7_variants, sweep_groups=sweep_groups))

    rows.sort(key=lambda r: (r["num_layers"], r["layer_set_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "layers_total": LAYERS_TOTAL,
        "layer_sets": rows,
        "sae_panel_layer_set_ids": sorted(sae_panel),
        "parent_baseline": PARENT_BASELINE,
        "shortlist_selection_rules": {
            "shortlist_size": 6,
            "required_slots": [
                "best_raw_phase7",
                "best_hybrid_phase7",
                "best_middle_family_phase7",
                "best_every2_family_phase7",
                "phase6_phase7_mismatch",
                "high_latent_low_causal_risk",
            ],
            "primary_rank_keys_phase7": [
                "result_token_top1",
                "operator_acc",
                "step_type_acc",
                "delta_logprob_vs_gpt2",
            ],
            "tie_breakers": ["fewer_layers", "lower_layer_variance"],
        },
        "causal_scope_defaults": {
            "variable": "subresult_value",
            "record_budget_pass1": 100,
            "record_budget_pass2_top_k": 2,
            "record_budget_pass2": 300,
            "off_manifold_checks": True,
            "subspace_builder": {"combine_policy": "union", "probe_position": "result"},
        },
    }


def save_manifest(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest_payload()
    p.write_text(json.dumps(payload, indent=2))
    return p


def load_manifest(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected manifest schema_version={payload.get('schema_version')!r}")
    return payload


def manifest_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for row in payload.get("layer_sets", []):
        idx[str(row["layer_set_id"])] = row
    return idx


def get_layer_set(payload: Dict[str, Any], layer_set_id: str) -> Dict[str, Any]:
    idx = manifest_index(payload)
    if layer_set_id not in idx:
        raise KeyError(f"Unknown layer_set_id={layer_set_id!r}; available={sorted(idx)[:8]}... ({len(idx)} total)")
    row = dict(idx[layer_set_id])
    row["layers"] = list(normalize_layers(row["layers"], layers_total=int(payload.get("layers_total", LAYERS_TOTAL))))
    row["num_layers"] = len(row["layers"])
    return row


def infer_layer_set_id_from_layers(payload: Dict[str, Any], layers: Sequence[int]) -> str | None:
    ls = normalize_layers(layers, layers_total=int(payload.get("layers_total", LAYERS_TOTAL)))
    for row in payload.get("layer_sets", []):
        try:
            row_ls = normalize_layers(row.get("layers", []), layers_total=int(payload.get("layers_total", LAYERS_TOTAL)))
        except Exception:
            continue
        if row_ls == ls:
            return str(row["layer_set_id"])
    return None

