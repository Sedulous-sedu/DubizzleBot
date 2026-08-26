"""Reusable Streamlit UI component renderers and field formatting helpers."""

from typing import Optional, List, Dict, Any, MutableMapping, Callable
import streamlit as st

def format_price(price_aed: Optional[float]) -> str:
    """Formats price in AED or returns truthful 'Not stated' fallback."""
    if price_aed is None:
        return "Price: Not stated"
    return f"AED {price_aed:,.0f}"

def format_monthly(monthly_payment_aed: Optional[float]) -> str:
    """Formats monthly installment payment in AED or returns truthful fallback."""
    if monthly_payment_aed is None:
        return "Monthly: Not stated"
    return f"AED {monthly_payment_aed:,.0f} / mo"

def format_mileage(mileage_km: Optional[float]) -> str:
    """Formats mileage in kilometers or returns truthful fallback."""
    if mileage_km is None:
        return "Mileage: Not stated"
    return f"{mileage_km:,.0f} km"

def format_specs(regional_specs: Optional[str]) -> str:
    """Formats regional specification string or returns truthful fallback."""
    if not regional_specs or not regional_specs.strip():
        return "Specs: Not stated"
    clean = regional_specs.strip()
    if "specs" in clean.lower():
        return clean
    return f"{clean} Specs"

def format_warranty(warranty_status: Optional[str]) -> str:
    """Formats warranty description or returns truthful fallback without guessing."""
    if not warranty_status or not warranty_status.strip():
        return "Warranty: Not stated"
    return warranty_status.strip()

def format_body_type(body_type: Optional[str]) -> str:
    """Formats vehicle body type or returns truthful fallback."""
    if not body_type or not body_type.strip():
        return "Body: Not stated"
    return body_type.strip().title()

def render_vehicle_card(listing: Dict[str, Any], global_index: int) -> None:
    """
    Renders an individual vehicle result card with truthful attributes and global ordinal numbering.
    global_index is 1-based (e.g. 1, 2, ..., N).
    """
    year = listing.get("year", "Year N/A")
    make = listing.get("make", "")
    model = listing.get("model", "")
    trim = listing.get("trim") or ""
    title_display = f"[{global_index}] {year} {make} {model} {trim}".strip()

    price_str = format_price(listing.get("price_aed"))
    monthly_str = format_monthly(listing.get("monthly_payment_aed"))
    mileage_str = format_mileage(listing.get("mileage_km"))
    specs_str = format_specs(listing.get("regional_specs"))
    warranty_str = format_warranty(listing.get("warranty_status"))
    body_str = format_body_type(listing.get("body_type"))
    listing_id = listing.get("listing_id")

    with st.container(border=True):
        st.markdown(f"##### {title_display}")

        photo_url = listing.get("photo_url")
        if photo_url and isinstance(photo_url, str) and photo_url.strip().startswith("http"):
            try:
                st.image(photo_url, use_container_width=True)
            except Exception:
                st.markdown("🚗 *[Vehicle photo not available]*")
        else:
            st.markdown("🚗 *[Vehicle photo not available]*")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"💰 **{price_str}**")
            st.caption(f"📅 {monthly_str}")
            st.markdown(f"📍 `{specs_str}`")
        with col2:
            st.markdown(f"🛣️ **{mileage_str}**")
            st.caption(f"🛡️ {warranty_str}")
            st.caption(f"🚘 {body_str} | ID: #{listing_id}")

def render_matched_cars(
    matched_cars: List[Dict[str, Any]],
    total_matches: int,
    message_id: str,
    state: MutableMapping,
) -> None:
    """
    Renders the list of matched vehicles in a responsive layout while preserving
    the full received array and global ordinal positions.
    """
    if not matched_cars:
        return

    shown_count = len(matched_cars)
    st.markdown(
        f"**🚘 Verified Matching Inventory ({shown_count} of {total_matches} total matched):**"
    )

    # Initial limit for responsive layout
    INITIAL_LIMIT = 6
    if shown_count <= INITIAL_LIMIT:
        cols = st.columns(2)
        for i, car in enumerate(matched_cars):
            with cols[i % 2]:
                render_vehicle_card(car, global_index=i + 1)
    else:
        # Show first 6 directly
        cols = st.columns(2)
        for i in range(INITIAL_LIMIT):
            with cols[i % 2]:
                render_vehicle_card(matched_cars[i], global_index=i + 1)

        # Remaining results in expander with exact continuing global indices
        remaining_count = shown_count - INITIAL_LIMIT
        with st.expander(f"➕ View {remaining_count} more matching vehicles..."):
            rem_cols = st.columns(2)
            for j in range(INITIAL_LIMIT, shown_count):
                with rem_cols[(j - INITIAL_LIMIT) % 2]:
                    render_vehicle_card(matched_cars[j], global_index=j + 1)

def render_empty_state(on_prompt_select: Callable[[str], None]) -> None:
    """Renders the welcome hero banner and sample starter chips for evaluators."""
    st.markdown("### 👋 Welcome to DubizzleBot")
    st.markdown(
        "I am your AI assistant for exploring verified UAE used-car inventory, remembering your preferences, "
        "and booking test drives or viewings."
    )
    st.markdown("#### 💡 Try an example query:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔍 Inventory Search**")
        if st.button("Show me Bentleys", key="sample_bentley", use_container_width=True):
            on_prompt_select("Show me Bentleys")
        if st.button("Show me GCC cars under AED 150,000", key="sample_gcc", use_container_width=True):
            on_prompt_select("Show me GCC cars under AED 150,000")

        st.markdown("**🧠 Returning-User Memory**")
        if st.button("What cars did I like?", key="sample_liked", use_container_width=True):
            on_prompt_select("What cars did I like?")

    with col2:
        st.markdown("**📅 Test-Drive Booking**")
        if st.button("I want to test drive a Bentley", key="sample_testdrive", use_container_width=True):
            on_prompt_select("I want to test drive a Bentley")

        st.markdown("**📋 Sales Enquiry**")
        if st.button("I'd like someone to contact me about buying a GCC SUV", key="sample_lead", use_container_width=True):
            on_prompt_select("I'd like someone to contact me about buying a GCC SUV")
