from __future__ import annotations

import json
import tempfile
import unittest

from phase7.model_registry import ModelSpec, create_adapter, list_model_specs, resolve_model_spec, validate_model_spec


class ModelRegistryTests(unittest.TestCase):
    def test_list_model_specs_contains_expected_models(self) -> None:
        specs = list_model_specs()
        self.assertIn("gpt2-medium", specs)
        self.assertIn("qwen2.5-7b", specs)

    def test_create_adapter_contract_fields(self) -> None:
        for model_key in ("gpt2-medium", "qwen2.5-7b"):
            adapter = create_adapter(model_key, device="cpu")
            meta = adapter.metadata()
            self.assertEqual(meta["model_key"], model_key)
            self.assertIn("model_family", meta)
            self.assertGreater(int(meta["num_layers"]), 0)
            self.assertGreater(int(meta["hidden_dim"]), 0)
            self.assertTrue(meta["tokenizer_id"])

    def test_validate_model_spec_rejects_bad_shapes(self) -> None:
        bad = ModelSpec(
            model_key="bad",
            hf_model_id="x/y",
            tokenizer_id="x/y",
            num_layers=0,
            hidden_dim=128,
            default_dtype="float32",
            adapter_class="GPT2MediumAdapter",
        )
        with self.assertRaises(ValueError):
            validate_model_spec(bad)

    def test_resolve_model_spec_rejects_mismatched_override_model_key(self) -> None:
        payload = {"model_key": "qwen2.5-7b"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        with self.assertRaises(ValueError):
            resolve_model_spec("gpt2-medium", path)


if __name__ == "__main__":
    unittest.main()
