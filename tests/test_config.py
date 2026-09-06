from mlagent import config


def test_defaults_are_sane():
    assert config.MODEL_ID.startswith("claude-")
    assert config.MAX_TOKENS >= 4096
    assert config.EFFORT in {"low", "medium", "high", "xhigh", "max"}
    assert set(config.DEFAULT_RATES) == {"T4", "L4", "A100"}
    assert config.PRICE_PER_100_UNITS_USD > 0
