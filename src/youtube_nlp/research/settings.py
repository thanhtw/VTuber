"""Versioned research settings, separate from the exploratory pipeline."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from ..config import Config, DEFAULT_ENV, PROJECT_ROOT


@dataclass
class Study:
    api: Config
    root: Path
    options: dict = field(default_factory=dict)

    @classmethod
    def load(cls, env=DEFAULT_ENV, config=PROJECT_ROOT / "research/study.json"):
        path = Path(config).resolve()
        options = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(options, dict):
            raise ValueError("Study settings must be a JSON object.")
        root = (path.parent / options.get("output_dir", "../research_output")).resolve()
        for key in ("bootstrap_samples", "permutations", "audit_sample_per_language"):
            value = options.get(key, 1000 if key != "audit_sample_per_language" else 30)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        for key in ("topic_counts", "seeds"):
            values = options.get(key, [2, 3, 4, 5] if key == "topic_counts" else [42, 123, 2026])
            if not isinstance(values, list) or not values or any(type(v) is not int or v < (1 if key == "topic_counts" else 0) for v in values):
                raise ValueError(f"{key} must be a nonempty integer array.")
        for query in options.get("queries", []):
            if not query.get("q", "").strip() or query.get("type", "video") not in {"video", "channel"}:
                raise ValueError("Every query needs q and a video/channel type.")
        if options.get("sentiment_backend", "transformer") not in {"transformer", "vader"}:
            raise ValueError("sentiment_backend must be transformer or vader.")
        rules = Path(options.get("aspect_rules", "aspect_rules.json"))
        options["aspect_rules_path"] = str((path.parent / rules).resolve())
        return cls(Config.load(env, require_key=False), root, options)

    def get(self, name, default=None):
        return self.options.get(name, default)

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
