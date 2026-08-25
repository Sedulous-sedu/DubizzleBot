"""Comprehensive offline unit tests for Phase 3A Natural Language Query Interpreter with mocked LLM transport."""

import pytest
from pydantic import ValidationError
from backend.models.intent import (
    UserIntentEnum,
    RegionalSpecEnum,
    SearchReadinessState,
    UnsupportedConstraint,
    ParsedInventoryQuery,
    LLMIntentPayload,
    ParsedUserIntent,
)
from backend.services.llm import LLMService
from backend.services.query_interpreter import QueryInterpreter

def create_mock_interpreter(mock_payload: LLMIntentPayload) -> QueryInterpreter:
    """Helper to create a QueryInterpreter with a deterministic mock handler."""
    def handler(messages, response_model):
        return mock_payload

    llm = LLMService(mock_handler=handler)
    return QueryInterpreter(llm_service=llm)

def test_interpret_simple_make():
    """Verify 'Show me Bentleys' extracts make=Bentley with ready readiness state."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Show me Bentleys")
    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.query_filters.make == "Bentley"
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_make_and_model():
    """Verify 'Bentley Bentayga' extracts make and model."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley", model="Bentayga")
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Bentley Bentayga")
    assert res.query_filters.make == "Bentley"
    assert res.query_filters.model == "Bentayga"
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_price_ceiling():
    """Verify 'cars below AED 100,000' extracts max_price_aed."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(max_price_aed=100000.0)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("cars below AED 100,000")
    assert res.query_filters.max_price_aed == 100000.0
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_price_range():
    """Verify 'between AED 70k and AED 120k' extracts min and max price."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(min_price_aed=70000.0, max_price_aed=120000.0)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("between AED 70k and AED 120k")
    assert res.query_filters.min_price_aed == 70000.0
    assert res.query_filters.max_price_aed == 120000.0
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_year_minimum():
    """Verify 'Land Rover 2018 or newer' extracts make and min_year."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover", min_year=2018)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Land Rover 2018 or newer")
    assert res.query_filters.make == "Land Rover"
    assert res.query_filters.min_year == 2018
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_mileage_ceiling():
    """Verify 'less than 50,000 km' extracts max_mileage_km."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(max_mileage_km=50000)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("less than 50,000 km")
    assert res.query_filters.max_mileage_km == 50000

def test_interpret_monthly_payment_ceiling():
    """Verify 'monthly payment under 2500' extracts max_monthly_aed."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(max_monthly_aed=2500.0)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("monthly payment under 2500")
    assert res.query_filters.max_monthly_aed == 2500.0

def test_interpret_regional_specs():
    """Verify 'Korean spec' extracts regional_specs=Korean."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(regional_specs=RegionalSpecEnum.KOREAN)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Korean spec")
    assert res.query_filters.regional_specs == RegionalSpecEnum.KOREAN

def test_interpret_positive_warranty():
    """Verify 'with warranty' extracts warranty=True."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(warranty=True)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("with warranty")
    assert res.query_filters.warranty is True

def test_interpret_negative_warranty():
    """Verify 'without warranty' extracts warranty=False."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(warranty=False)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("without warranty")
    assert res.query_filters.warranty is False

def test_interpret_combined_multi_filter():
    """Verify combined multi-filter query 'GCC Bentleys from 2020 onwards'."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley", min_year=2020, regional_specs=RegionalSpecEnum.GCC)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("GCC Bentleys from 2020 onwards")
    assert res.query_filters.make == "Bentley"
    assert res.query_filters.min_year == 2020
    assert res.query_filters.regional_specs == RegionalSpecEnum.GCC
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_textual_features_in_keywords():
    """Verify explicitly requested textual features like 'Mansory' and 'panoramic roof' are captured via keywords."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley", keywords="Mansory panoramic")
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Mansory Bentley with panoramic roof")
    assert res.query_filters.make == "Bentley"
    assert res.query_filters.keywords == "Mansory panoramic"
    assert res.readiness_state == SearchReadinessState.READY

def test_interpret_vague_request_triggers_clarification():
    """Verify vague input 'I want a cheap car' emits clarification_required."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        requires_clarification=True,
        clarification_question="What is your maximum target budget in AED?"
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("I want a cheap car")
    assert res.requires_clarification is True
    assert res.clarification_question is not None
    assert res.readiness_state == SearchReadinessState.CLARIFICATION_REQUIRED

def test_interpret_unsupported_constraints():
    """Verify unsupported criteria like out-of-domain filters are structured cleanly."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        unsupported_constraints=[
            UnsupportedConstraint(field="service_history_guarantee", requested_value="guaranteed 10 year dealer service", reason="not_supported_by_inventory_filter"),
            UnsupportedConstraint(field="financing_rate", requested_value="0% interest", reason="not_supported_by_inventory_filter")
        ]
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("I want a Bentley with 0% interest financing and guaranteed 10 year dealer service")
    assert len(res.unsupported_constraints) == 2
    assert res.unsupported_constraints[0].field == "service_history_guarantee"
    assert res.unsupported_constraints[1].field == "financing_rate"
    assert res.readiness_state == SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT

def test_interpret_unsupported_ranking():
    """Verify ranking requests like '5 cheapest Bentleys' are captured as unsupported constraints and limit is None."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley", limit=None),
        unsupported_constraints=[
            UnsupportedConstraint(field="ranking", requested_value="cheapest", reason="ranking_not_supported_by_inventory_filter")
        ]
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("5 cheapest Bentleys")
    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.query_filters.make == "Bentley"
    assert res.query_filters.limit is None
    assert len(res.unsupported_constraints) == 1
    assert res.unsupported_constraints[0].field == "ranking"
    assert res.unsupported_constraints[0].requested_value == "cheapest"
    assert res.readiness_state == SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT

def test_interpret_general_chat_intent():
    """Verify non-inventory general chat intent."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.GENERAL_CHAT
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("Hello, what can you do?")
    assert res.intent == UserIntentEnum.GENERAL_CHAT
    assert res.readiness_state == SearchReadinessState.NON_INVENTORY_INTENT

def test_interpret_viewing_intent_preserves_filters():
    """Verify test drive request preserves specified vehicle filters."""
    payload = LLMIntentPayload(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        query_filters=ParsedInventoryQuery(make="Bentley", max_price_aed=150000.0, regional_specs=RegionalSpecEnum.GCC)
    )
    interpreter = create_mock_interpreter(payload)
    res = interpreter.interpret("I want to test drive a GCC Bentley under 150k")
    assert res.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST
    assert res.query_filters.make == "Bentley"
    assert res.query_filters.max_price_aed == 150000.0
    assert res.query_filters.regional_specs == RegionalSpecEnum.GCC

def test_interpret_malformed_llm_output_fallback():
    """Verify malformed model exception falls back safely to UNKNOWN intent."""
    def error_handler(messages, response_model):
        raise RuntimeError("Malformed JSON payload from model")

    llm = LLMService(mock_handler=error_handler)
    interpreter = QueryInterpreter(llm_service=llm)
    res = interpreter.interpret("Some invalid query")
    assert res.intent == UserIntentEnum.UNKNOWN
    assert res.readiness_state == SearchReadinessState.NON_INVENTORY_INTENT

def test_interpret_prompt_injection_rejection():
    """Verify injected schema fields (e.g. listing_id) are rejected by extra='forbid'."""
    with pytest.raises(ValidationError):
        ParsedInventoryQuery(
            make="Bentley",
            listing_id=38  # Extra forbidden attribute
        )

def test_interpret_schema_range_invariant_violation():
    """Verify range invariant validation fails when min_price > max_price."""
    with pytest.raises(ValidationError):
        ParsedInventoryQuery(
            min_price_aed=200000.0,
            max_price_aed=100000.0
        )
