"""Comprehensive unit tests and mandatory regression probes for InventoryService extraction rules and deterministic search."""

import pytest
from backend.services.inventory import InventoryService

@pytest.fixture
def service():
    """Fixture providing initialized InventoryService instance loaded with dataset."""
    return InventoryService(dataset_path="Copy_of_sample_cars_dataset.xlsx")

# --- MANDATORY NEW REGRESSION PROBE TESTS ---

def test_bare_usa_geography_is_not_usa_spec(service):
    """Verify bare geography 'Owner relocated to USA.' or 'Contact us' is NOT classified as USA spec."""
    spec_val, snip = service._extract_regional_specs("Owner relocated to USA. Contact us today!")
    assert spec_val is None
    assert snip is None

def test_korean_nationality_is_not_korean_spec(service):
    """Verify 'Korean owner, excellent condition.' is NOT classified as Korean spec."""
    spec_val, snip = service._extract_regional_specs("Korean owner, excellent condition.")
    assert spec_val is None
    assert snip is None

def test_explicit_korea_spec_is_korean_spec(service):
    """Verify explicit 'Korea Spec' is classified as Korean spec."""
    spec_val, snip = service._extract_regional_specs("Lincoln Aviator Luxury SUV 2022 korea specs - Full option")
    assert spec_val == "Korean"

def test_provenance_returns_actual_matched_regional_spec_substring(service):
    """Verify provenance returns the exact matched substring from source text."""
    spec_val, snip = service._extract_regional_specs("Excellent condition. Korea Spec. Full service history.")
    assert spec_val == "Korean"
    assert snip == "Korea Spec"

def test_gargash_alone_is_not_active_warranty(service):
    """Verify 'Gargash maintained vehicle. No accident history.' does NOT establish active warranty."""
    has_pos, status, snip = service._extract_warranty("Gargash maintained vehicle. No accident history.")
    assert has_pos is None
    assert status is None
    assert snip is None

def test_4x4_alone_is_not_suv(service):
    """Verify '4x4, automatic transmission.' does NOT establish body_type=SUV."""
    btype = service._extract_body_type("Brand", "Model", "Title", "4x4, automatic transmission.")
    assert btype is None

def test_4x4_pickup_resolves_to_pickup(service):
    """Verify '4x4 pickup truck.' resolves to body_type=Pickup."""
    btype = service._extract_body_type("Brand", "Model", "Title", "4x4 pickup truck.")
    assert btype == "Pickup"

def test_explicit_suv_resolves_to_suv(service):
    """Verify explicit 'Premium SUV with full options.' resolves to body_type=SUV."""
    btype = service._extract_body_type("Brand", "Model", "Title", "Premium SUV with full options.")
    assert btype == "SUV"

def test_service_price_is_not_vehicle_cash_price(service):
    """Verify 'Service price: AED 5,000.' is NOT parsed as vehicle cash price."""
    price_val, snip = service._extract_cash_price("Service price: AED 5,000.")
    assert price_val is None
    assert snip is None

def test_electric_range_is_not_odometer_mileage(service):
    """Verify 'Electric range 650 km.' is NOT parsed as odometer mileage."""
    m_val, snip = service._extract_mileage("Electric range 650 km.")
    assert m_val is None
    assert snip is None

# --- REGRESSION TESTS FOR FAILURE RISKS ---

def test_regression_pm_time_not_monthly_installment(service):
    """Confirm showroom opening times like '9 PM' or '8 PM' are not parsed as monthly installments."""
    m_val, snip = service._extract_monthly_payment("Showroom open Monday to Saturday 8am to 9pm. Call 0501234567.")
    assert m_val is None

def test_regression_warranty_can_be_arranged_not_active(service):
    """Confirm 'warranty can be arranged' is not classified as active warranty."""
    has_pos, status, snip = service._extract_warranty("RTA approved - A warranty can be arranged - A service contract can be arranged")
    assert has_pos is False
    assert status == "Warranty Option Available (Not Active)"

def test_regression_warranty_expired_not_active(service):
    """Confirm 'warranty expired' is not classified as active warranty."""
    has_pos, status, snip = service._extract_warranty("Vehicle in good condition, warranty expired last year.")
    assert has_pos is False
    assert status == "No Warranty / Expired"

def test_regression_no_warranty_not_active(service):
    """Confirm 'no warranty' is not classified as active warranty."""
    has_pos, status, snip = service._extract_warranty("Sold as is, no warranty included.")
    assert has_pos is False
    assert status == "No Warranty / Expired"

def test_regression_phone_numbers_not_cash_price(service):
    """Confirm phone numbers like '+971552011671' or '0503900650' are not parsed as cash price."""
    price_val, snip = service._extract_cash_price("For details call Mobile No : +971552011671 or 0503900650.")
    assert price_val is None

def test_regression_monthly_payments_not_cash_price(service):
    """Confirm monthly payments like 'AED 2,111 monthly' are not parsed as cash price."""
    price_val, snip = service._extract_cash_price("Payment plans: AED 2,111.00 monthly for 5 years with 10% Down-Payment.")
    assert price_val is None

def test_regression_unknown_derived_values_become_none(service):
    """Confirm unknown/unextracted derived facts become None instead of fabricated string defaults."""
    all_cars = service.search_cars()
    no_warr_cars = [c for c in all_cars if c["has_positive_warranty"] is None]
    assert len(no_warr_cars) > 0
    assert no_warr_cars[0]["warranty_status"] is None

def test_regression_strict_filters_exclude_missing_values(service):
    """Confirm strict price and mileage filters exclude records where that value is unavailable."""
    all_cars = service.search_cars()
    total_count = len(all_cars)
    price_filtered = service.search_cars(max_price=5000000.0)
    valid_price_count = sum(1 for c in all_cars if c["price_aed"] is not None)
    assert len(price_filtered) == valid_price_count
    assert len(price_filtered) < total_count

def test_regression_returned_listing_ids_exist_in_source(service):
    """Confirm all returned Listing_ID values exist in the supplied dataset."""
    all_cars = service.search_cars()
    assert len(all_cars) == 100
    for car in all_cars:
        assert 1 <= car["listing_id"] <= 100

def test_regression_identical_searches_deterministic_ordering(service):
    """Confirm identical searches always return identical deterministic ordering."""
    run1 = service.search_cars(make="mercedes-benz", limit=10)
    run2 = service.search_cars(make="mercedes-benz", limit=10)
    assert run1 == run2
    ids = [c["listing_id"] for c in run1]
    assert ids == sorted(ids)

# --- DATASET INTEGRATION TESTS ---

def test_dataset_loading(service):
    """Verify dataset loads successfully with expected listing count."""
    stats = service.get_summary_statistics()
    assert stats["total_listings"] == 100
    assert stats["unique_makes"] == 35

def test_cash_price_and_installment_distinction(service):
    """Verify distinct cash price and monthly payment extraction on Listing 3 (Velar)."""
    velar = service.search_cars(keywords="velar")[0]
    assert velar["price_aed"] == 119750.0
    assert velar["monthly_payment_aed"] == 2111.0
    assert velar["price_aed"] != velar["monthly_payment_aed"]
