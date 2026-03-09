from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from phase7.state_decoder_core import (
    NumericNormStats,
    OP_TO_ID,
    StateDecoderExperimentConfig,
    STEP_TO_ID,
    compute_multitask_loss,
    evaluate_state_model,
)


def _base_cfg() -> StateDecoderExperimentConfig:
    return StateDecoderExperimentConfig(
        name="mask_test",
        input_variant="raw",
        layers=(0,),
        model_hidden_dim=4,
        model_sae_dim=8,
        vocab_size=5,
        operator_operate_only_supervision=True,
        operator_loss_mode="ce",
    )


def _base_out(batch_size: int = 2) -> dict:
    return {
        "result_token_logits": torch.zeros(batch_size, 5),
        "operator_logits": torch.zeros(batch_size, 5),
        "step_type_logits": torch.zeros(batch_size, 4),
        "magnitude_logits": torch.zeros(batch_size, 6),
        "sign_logits": torch.zeros(batch_size, 3),
        "subresult_pred": torch.zeros(batch_size),
        "lhs_pred": torch.zeros(batch_size),
        "rhs_pred": torch.zeros(batch_size),
    }


def _base_batch(step_type_ids: torch.Tensor) -> dict:
    bsz = int(step_type_ids.shape[0])
    return {
        "x": torch.zeros(bsz, 1, 4),
        "result_token_id": torch.zeros(bsz, dtype=torch.long),
        "operator_id": torch.zeros(bsz, dtype=torch.long),
        "step_type_id": step_type_ids.long(),
        "magnitude_id": torch.zeros(bsz, dtype=torch.long),
        "sign_id": torch.zeros(bsz, dtype=torch.long),
        "subresult_z": torch.zeros(bsz),
        "lhs_z": torch.zeros(bsz),
        "rhs_z": torch.zeros(bsz),
        "mask_subresult": torch.ones(bsz),
        "mask_lhs": torch.ones(bsz),
        "mask_rhs": torch.ones(bsz),
        "baseline_logprob": torch.zeros(bsz),
        "operator": ["+" for _ in range(bsz)],
        "magnitude_bucket": ["[0,10)" for _ in range(bsz)],
        "trace_id": [f"t{i}" for i in range(bsz)],
        "step_idx": list(range(bsz)),
        "example_idx": list(range(bsz)),
    }


class _FakeModel:
    def __init__(self, out: dict, cfg: StateDecoderExperimentConfig):
        self._out = out
        self.exp_cfg = cfg

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor):
        bsz = int(x.shape[0])
        out = {}
        for k, v in self._out.items():
            out[k] = v[:bsz]
        return out


class OperateOnlyMaskingTests(unittest.TestCase):
    def test_operator_loss_zero_when_no_operate_rows(self) -> None:
        cfg = _base_cfg()
        out = _base_out(batch_size=2)
        batch = _base_batch(torch.tensor([STEP_TO_ID["emit_result"], STEP_TO_ID["emit_result"]]))
        losses = compute_multitask_loss(out, batch, cfg)
        self.assertAlmostEqual(float(losses["operator"].item()), 0.0, places=7)

    def test_operator_loss_matches_operate_subset(self) -> None:
        cfg = _base_cfg()
        out = _base_out(batch_size=3)
        out["operator_logits"] = torch.tensor(
            [
                [2.0, 0.1, 0.1, 0.1, 0.1],  # operate row
                [0.1, 2.0, 0.1, 0.1, 0.1],  # non-operate row (masked out)
                [0.1, 0.1, 2.0, 0.1, 0.1],  # operate row
            ],
            dtype=torch.float32,
        )
        step_ids = torch.tensor([STEP_TO_ID["operate"], STEP_TO_ID["emit_result"], STEP_TO_ID["operate"]])
        batch = _base_batch(step_ids)
        batch["operator_id"] = torch.tensor([0, 1, 2], dtype=torch.long)
        losses = compute_multitask_loss(out, batch, cfg)

        expected = F.cross_entropy(out["operator_logits"][torch.tensor([0, 2])], batch["operator_id"][torch.tensor([0, 2])])
        self.assertAlmostEqual(float(losses["operator"].item()), float(expected.item()), places=6)

    def test_operator_known_only_mask_excludes_unknown_rows(self) -> None:
        cfg = _base_cfg()
        cfg.operator_known_only_supervision = True
        out = _base_out(batch_size=3)
        out["operator_logits"] = torch.tensor(
            [
                [0.1, 0.1, 2.0, 0.1, 0.1],  # known operate (*)
                [0.1, 0.1, 0.1, 0.1, 2.0],  # unknown operate (must be masked out)
                [2.0, 0.1, 0.1, 0.1, 0.1],  # known operate (+)
            ],
            dtype=torch.float32,
        )
        step_ids = torch.tensor([STEP_TO_ID["operate"], STEP_TO_ID["operate"], STEP_TO_ID["operate"]])
        batch = _base_batch(step_ids)
        batch["operator_id"] = torch.tensor([2, OP_TO_ID["unknown"], 0], dtype=torch.long)
        losses = compute_multitask_loss(out, batch, cfg)
        expected = F.cross_entropy(out["operator_logits"][torch.tensor([0, 2])], batch["operator_id"][torch.tensor([0, 2])])
        self.assertAlmostEqual(float(losses["operator"].item()), float(expected.item()), places=6)

    def test_evaluate_emits_operate_only_operator_metrics(self) -> None:
        cfg = _base_cfg()
        out = _base_out(batch_size=3)
        # Predictions: rows 0/2 are operate; row 1 is emit_result.
        # row0 correct op=0, row2 incorrect (pred=0 true=2), row1 ignored for operate-only.
        out["operator_logits"] = torch.tensor(
            [
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        out["step_type_logits"] = torch.tensor(
            [
                [0.0, 3.0, 0.0, 0.0],  # operate
                [0.0, 0.0, 3.0, 0.0],  # emit_result
                [0.0, 3.0, 0.0, 0.0],  # operate
            ],
            dtype=torch.float32,
        )
        step_ids = torch.tensor([STEP_TO_ID["operate"], STEP_TO_ID["emit_result"], STEP_TO_ID["operate"]])
        batch = _base_batch(step_ids)
        batch["operator_id"] = torch.tensor([0, 1, 2], dtype=torch.long)
        loader = [batch]
        stats = {
            "subresult_value": NumericNormStats(0.0, 1.0),
            "lhs_value": NumericNormStats(0.0, 1.0),
            "rhs_value": NumericNormStats(0.0, 1.0),
        }
        metrics = evaluate_state_model(_FakeModel(out, cfg), loader, "cpu", stats)
        self.assertEqual(metrics["operator_num_operate_rows"], 2)
        self.assertAlmostEqual(float(metrics["operator_acc_operate_only"]), 0.5, places=6)
        self.assertEqual(metrics["operator_num_operate_known_rows"], 2)
        self.assertAlmostEqual(float(metrics["operator_acc_operate_known_only"]), 0.5, places=6)
        self.assertIn("operator_head_confusion_operate_only", metrics)
        self.assertIn("operator_head_confusion_operate_known_only", metrics)
        self.assertIn("operator_head_ece_operate_only", metrics)
        self.assertIn("operator_head_ece_operate_known_only", metrics)
        self.assertIn("operator_head_per_class_accuracy_operate_only", metrics)
        self.assertIn("operator_head_per_class_accuracy_operate_known_only", metrics)


if __name__ == "__main__":
    unittest.main()
