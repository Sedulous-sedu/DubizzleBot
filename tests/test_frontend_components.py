"""Offline unit tests for DubizzleBot Streamlit component formatters and card helpers."""

import pytest

from frontend.components import (
    format_price,
    format_monthly,
    format_mileage,
    format_specs,
    format_warranty,
    format_body_type,
)

def test_format_price_valid_and_none():
    """Verify price formatting produces readable AED string or truthful 'Not stated' fallback."""
    assert format_price(150000.0) == "AED 150,000"
    assert format_price(0.0) == "AED 0"
    assert format_price(None) == "Price: Not stated"

def test_format_monthly_valid_and_none():
    """Verify monthly payment formatting produces readable string or truthful fallback."""
    assert format_monthly(2750.0) == "AED 2,750 / mo"
    assert format_monthly(None) == "Monthly: Not stated"

def test_format_mileage_valid_and_none():
    """Verify mileage formatting produces readable km string or truthful fallback."""
    assert format_mileage(318.0) == "318 km"
    assert format_mileage(45000.0) == "45,000 km"
    assert format_mileage(None) == "Mileage: Not stated"

def test_format_specs_valid_and_none():
    """Verify regional specs formatting handles GCC and None safely."""
    assert format_specs("GCC") == "GCC Specs"
    assert format_specs("GCC Specs") == "GCC Specs"
    assert format_specs("American") == "American Specs"
    assert format_specs(None) == "Specs: Not stated"
    assert format_specs("") == "Specs: Not stated"

def test_format_warranty_valid_and_none():
    """Verify warranty formatting uses warranty_status directly without guessing."""
    assert format_warranty("Under dealership warranty until 2027") == "Under dealership warranty until 2027"
    assert format_warranty("Yes") == "Yes"
    assert format_warranty("1 Year Dealer Warranty") == "1 Year Dealer Warranty"
    assert format_warranty(None) == "Warranty: Not stated"
    assert format_warranty("") == "Warranty: Not stated"

def test_format_body_type_valid_and_none():
    """Verify body type formatting handles values and None safely."""
    assert format_body_type("coupe") == "Coupe"
    assert format_body_type("SUV") == "Suv"
    assert format_body_type(None) == "Body: Not stated"
    assert format_body_type("") == "Body: Not stated"
