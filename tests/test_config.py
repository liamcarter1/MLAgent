from mlagent import config


def test_defaults_are_sane():
    assert config.MODEL_ID.startswith("claude-")
    assert config.MAX_TOKENS >= 4096
    assert config.EFFORT in {"low", "medium", "high", "xhigh", "max"}
    assert config.COST_FILE == "cost.json"
    assert config.DRY_RUN_FILE == "dry_run.json"
