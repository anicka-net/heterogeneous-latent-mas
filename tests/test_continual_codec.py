"""CPU-only checks for intermittent codec training (no model downloads)."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from continual_codec import load_texts, split_and_mix, write_jsonl, make_command, DEFAULT_ARCH


class TestSampling(unittest.TestCase):
    def test_deterministic_and_no_current_leakage(self):
        old = [f"old {i}" for i in range(150)] + ["new 0"]
        fresh = [f"new {i}" for i in range(90)]
        kwargs = dict(seed=17, train_new=32, replay_count=32, holdout_fraction=0.2)
        one = split_and_mix(old, fresh, **kwargs)
        self.assertEqual(one, split_and_mix(old, fresh, **kwargs))
        train, holdout, stats = one
        self.assertEqual(len(train), 64)
        self.assertTrue(holdout)
        self.assertFalse(set(train) & set(holdout))
        self.assertEqual(stats["replay_selected"], 32)

    def test_excludes_reserved_prior_holdouts(self):
        fresh = [f"round two {i}" for i in range(90)]
        old = ["private test A", "private test B"] + [f"old {i}" for i in range(90)]
        train, holdout, _ = split_and_mix(
            old, fresh, seed=11, train_new=30, replay_count=90,
            holdout_fraction=.2, reserved_holdout=["private test A", "private test B"])
        self.assertNotIn("private test A", train)
        self.assertNotIn("private test B", train)
        self.assertFalse(set(train) & set(holdout))

    def test_invalid_sampling(self):
        with self.assertRaises(ValueError):
            split_and_mix([], ["one"], seed=1, train_new=1,
                          replay_count=0, holdout_fraction=.2)
        with self.assertRaises(ValueError):
            split_and_mix([], [], seed=1, train_new=0,
                          replay_count=0, holdout_fraction=.2)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "anchors.jsonl"
            write_jsonl(path, ["Hello \\u263a", "Hello \\u263a", "Second"])
            self.assertEqual(load_texts(path), ["Hello \\u263a", "Second"])
            path.write_text('{"not_text": "x"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_texts(path)


class TestOrchestration(unittest.TestCase):
    def test_trainer_python_parses_and_contains_warm_start(self):
        code = (ROOT / "train_vision_latent_mas_codec_new.py").read_text(encoding="utf-8")
        ast.parse(code)
        self.assertIn("--vision_codec_resume_path", code)
        self.assertIn("enc.load_state_dict(initial_encoder_state, strict=True)", code)
        self.assertIn("ckpt[\"continual_history\"]", code)

    def test_command_includes_config_and_parent(self):
        cmd = make_command(
            trainer=Path("train.py"), candidate=Path("candidate.pt"),
            parent=Path("parent.pt"), anchors=Path("anchors.jsonl"),
            models="Qwen/foo,google/bar", arch=DEFAULT_ARCH,
            ref_idx=1, steps=25, lr=2e-5, seed=7, latent_steps=16)
        self.assertIn("--vision_codec_resume_path", cmd)
        self.assertIn("--vision_codec_gate_init_bias", cmd)
        self.assertEqual(cmd[cmd.index("--vision_codec_ref_idx") + 1], "1")

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fresh = root / "fresh.jsonl"
            replay = root / "replay.jsonl"
            write_jsonl(fresh, [f"fresh {i}" for i in range(80)])
            write_jsonl(replay, [f"replay {i}" for i in range(80)])
            trainer = root / "stub.py"
            trainer.write_text("pass\n", encoding="utf-8")
            cmd = [
                sys.executable, str(ROOT / "scripts" / "continual_codec.py"),
                "--new-file", str(fresh), "--replay-file", str(replay),
                "--trainer", str(trainer), "--burst-dir", str(root / "burst"),
                "--candidate", str(root / "candidate.pt"), "--dry-run",
                "--train-new", "15", "--replay", "15",
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertIn("Command:", completed.stdout)
            self.assertFalse((root / "burst").exists())
            self.assertFalse((root / "candidate.pt").exists())

    def test_mock_training_creates_unvalidated_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fresh, replay = root / "fresh.jsonl", root / "replay.jsonl"
            write_jsonl(fresh, [f"fresh {i}" for i in range(80)])
            write_jsonl(replay, [f"replay {i}" for i in range(80)])
            trainer = root / "stub.py"
            trainer.write_text(
                "import sys\nfrom pathlib import Path\n"
                "args=sys.argv\n"
                "Path(args[args.index('--vision_codec_path')+1]).write_bytes(b'mock')\n",
                encoding="utf-8",
            )
            out = root / "candidate.pt"
            burst = root / "burst"
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "continual_codec.py"),
                "--new-file", str(fresh), "--replay-file", str(replay),
                "--trainer", str(trainer), "--burst-dir", str(burst),
                "--candidate", str(out), "--steps", "2", "--train-new", "15",
                "--replay", "15",
            ], capture_output=True, text=True, check=True)
            manifest = json.loads((burst / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "trained_unvalidated_candidate")
            self.assertEqual(out.read_bytes(), b"mock")
            self.assertFalse(set(load_texts(burst / "train.jsonl")) &
                             set(load_texts(burst / "holdout.jsonl")))


if __name__ == "__main__":
    unittest.main()
