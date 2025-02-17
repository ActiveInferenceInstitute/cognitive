
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "basic: mark test as basic initialization and property test"
    )
    config.addinivalue_line(
        "markers", 
        "single_step: mark test as single step dynamics test"
    )
    config.addinivalue_line(
        "markers", 
        "multi_step: mark test as multi-step evolution test"
    )
    config.addinivalue_line(
        "markers", 
        "complex: mark test as complex dynamical behavior test"
    )
