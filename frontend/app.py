"""Streamlit Client UI for DubizzleBot AI Assistant."""

import streamlit as st

def main():
    st.set_page_config(page_title="dubizzle cars - AI Assistant", page_icon="🚗", layout="wide")
    st.title("🚗 dubizzle cars AI Assistant")
    st.caption("Explore car inventory, book test drive viewing slots, and manage preferences.")

    st.sidebar.title("User Profile & Session")
    st.sidebar.text_input("User ID", value="user_123", key="user_id")
    
    st.info("UI skeleton initialized. Full chat interface & backend integration to be connected in subsequent phase.")

if __name__ == "__main__":
    main()
