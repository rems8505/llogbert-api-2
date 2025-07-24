from app.parser.validator import validate_line
import pytest

def test_valid_line():
    line = "nova-compute.log.2025-07-20_21:27:09 2025-07-20 21:08:19.735 2931 INFO nova.module [] Some log message"
    assert validate_line(line).startswith("nova-compute")

def test_invalid_line():
    with pytest.raises(ValueError):
        validate_line("not a valid log line")
