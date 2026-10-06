from ron.model_config import RonModelConfig

def test_default_model_config_is_valid():
    config = RonModelConfig()
    assert config.hidden_size % config.num_heads == 0

def test_invalid_head_dimension_is_rejected():
    try:
        RonModelConfig(hidden_size=130, num_heads=4)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid attention geometry must be rejected")
