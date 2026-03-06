from __future__ import annotations

import unittest

from phase7.trajectory_coherence import score_trajectory_coherence


class TrajectoryCoherenceTests(unittest.TestCase):
    def test_single_step_returns_agnostic(self) -> None:
        score, detail = score_trajectory_coherence(
            [{"step_idx": 0, "latent_pred_state": {"subresult_value": 1.0}}],
            [{"step_idx": 0, "text_claim_state": {"subresult_value": 1.0}}],
        )
        self.assertAlmostEqual(score, 0.5)
        self.assertFalse(detail["defined"])
        self.assertEqual(detail["common_step_count"], 1)

    def test_coherent_chain_scores_high(self) -> None:
        rows = [
            {
                "step_idx": 0,
                "latent_pred_state": {"subresult_value": 5.0},
                "text_claim_state": {"subresult_value": 5.0},
            },
            {
                "step_idx": 1,
                "latent_pred_state": {"lhs_value": 5.0, "rhs_value": 2.0, "subresult_value": 7.0},
                "text_claim_state": {"lhs_value": 5.0, "rhs_value": 2.0, "subresult_value": 7.0},
            },
            {
                "step_idx": 2,
                "latent_pred_state": {"lhs_value": 7.0, "rhs_value": 1.0, "subresult_value": 8.0},
                "text_claim_state": {"lhs_value": 7.0, "rhs_value": 1.0, "subresult_value": 8.0},
            },
        ]
        score, detail = score_trajectory_coherence(rows, rows)
        self.assertTrue(detail["defined"])
        self.assertGreater(score, 0.9)
        self.assertEqual(detail["common_step_count"], 3)
        self.assertEqual(detail["pairable_step_count"], 2)

    def test_divergent_chain_scores_lower(self) -> None:
        preds = [
            {"step_idx": 0, "latent_pred_state": {"subresult_value": 5.0}},
            {"step_idx": 1, "latent_pred_state": {"lhs_value": 5.0, "rhs_value": 2.0, "subresult_value": 7.0}},
            {"step_idx": 2, "latent_pred_state": {"lhs_value": 7.0, "rhs_value": 1.0, "subresult_value": 8.0}},
        ]
        claims = [
            {"step_idx": 0, "text_claim_state": {"subresult_value": 5.0}},
            {"step_idx": 1, "text_claim_state": {"lhs_value": 10.0, "rhs_value": 2.0, "subresult_value": 12.0}},
            {"step_idx": 2, "text_claim_state": {"lhs_value": 1.0, "rhs_value": 1.0, "subresult_value": 2.0}},
        ]
        score, detail = score_trajectory_coherence(preds, claims)
        self.assertTrue(detail["defined"])
        self.assertLess(score, 0.8)
        self.assertGreater(detail["trajectory_divergence"], 0.0)

    def test_mismatched_steps_join_by_step_idx(self) -> None:
        preds = [
            {"step_idx": 0, "latent_pred_state": {"subresult_value": 5.0}},
            {"step_idx": 1, "latent_pred_state": {"lhs_value": 5.0, "rhs_value": 2.0, "subresult_value": 7.0}},
            {"step_idx": 2, "latent_pred_state": {"lhs_value": 7.0, "rhs_value": 1.0, "subresult_value": 8.0}},
        ]
        claims = [
            {"step_idx": 0, "text_claim_state": {"subresult_value": 5.0}},
            {"step_idx": 2, "text_claim_state": {"lhs_value": 7.0, "rhs_value": 1.0, "subresult_value": 8.0}},
        ]
        score, detail = score_trajectory_coherence(preds, claims)
        self.assertTrue(detail["defined"])
        self.assertEqual(detail["common_step_count"], 2)
        self.assertEqual(detail["pairable_step_count"], 1)
        self.assertEqual(detail["pair_details"][0]["from_step_idx"], 0)
        self.assertEqual(detail["pair_details"][0]["to_step_idx"], 2)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
