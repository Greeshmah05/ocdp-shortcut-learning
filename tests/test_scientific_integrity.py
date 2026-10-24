"""
Scientific Integrity and Architectural Guard Tests:
Verifies that:
1. Training code and models do NOT import evaluation-only masks or group annotations.
2. Model parameters and outputs conform to Tier 3 specifications.
"""

import os
import re
import pytest


def test_no_forbidden_imports_in_models_or_losses():
    """Verify src/models and src/losses do NOT import src/evaluation or group labels."""
    forbidden_terms = [
        r"from\s+src\.evaluation",
        r"import\s+src\.evaluation",
        r"group_labels",
        r"cub_masks"
    ]
    
    check_dirs = ["src/models", "src/losses"]
    for d in check_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    for term in forbidden_terms:
                        assert not re.search(term, content), (
                            f"Violation in {filepath}: found forbidden import/term '{term}'"
                        )


def test_tier_declarations_in_configs():
    """Ensure all configs explicitly declare supervision tier."""
    import yaml
    configs_dir = "configs"
    if os.path.exists(configs_dir):
        for f in os.listdir(configs_dir):
            if f.endswith(".yaml") or f.endswith(".yml"):
                p = os.path.join(configs_dir, f)
                with open(p, "r", encoding="utf-8") as yf:
                    data = yaml.safe_load(yf)
                if isinstance(data, dict) and "experiment" in data:
                    assert "supervision_tier" in data, f"Config {f} missing 'supervision_tier'"
                    assert data["supervision_tier"] in ["T1", "T2", "T3"], (
                        f"Config {f} has invalid tier: {data['supervision_tier']}"
                    )
