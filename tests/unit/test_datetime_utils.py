from max_assistant.utils.datetime_utils import current_datetime


def test_current_datetime_returns_expected_keys():
    result = current_datetime()

    assert set(result.keys()) == {"ISODateTime", "Day", "Month", "Year"}


def test_current_datetime_returns_expected_types():
    result = current_datetime()

    assert isinstance(result["ISODateTime"], str)
    assert isinstance(result["Day"], int)
    assert isinstance(result["Month"], str)
    assert isinstance(result["Year"], int)


def test_current_datetime_iso_format_is_minute_precision():
    result = current_datetime()

    assert "T" in result["ISODateTime"]
    assert len(result["ISODateTime"].split(":")) == 2