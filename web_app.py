import pathlib

import streamlit as st

from swing_screener import run_full_system


st.set_page_config(
    page_title="NSE Swing Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NSE Swing Screener – Predicta Style")
st.markdown(
    "Screen all NSE stocks for swing trades using technical confluence, VCP/SFP setups, "
    "and fundamental quality – then rank and export the Top‑N."
)

with st.sidebar:
    st.header("Scan Settings")

    universe_limit = st.number_input(
        "Universe limit (0 = all NSE stocks)",
        min_value=0,
        max_value=5000,
        value=500,
        step=50,
    )

    start_index = st.number_input(
        "Start index (for chunking, 0-based)",
        min_value=0,
        max_value=5000,
        value=0,
        step=100,
    )

    period = st.selectbox("History period", options=["6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", options=["1d"], index=0)

    min_score = st.slider(
        "Minimum confluence score (0–8)",
        min_value=0,
        max_value=8,
        value=6,
    )

    top_n = st.slider("Top N to show/export", min_value=5, max_value=50, value=10, step=5)

    run_button = st.button("🚀 Run Screener", type="primary")


placeholder = st.empty()

if run_button:
    with st.spinner("Running NSE swing screener... this can take a few minutes for large universes."):
        top_df = run_full_system(
            universe_limit=universe_limit,
            min_confluence_score=min_score,
            period=period,
            interval=interval,
            top_n=top_n,
            start_index=start_index,
        )

    if top_df is None or top_df.empty:
        st.warning("No candidates produced (even after fallback scoring). Try lowering minimum score or shortening period.")
    else:
        st.subheader("Top Candidates")
        st.dataframe(top_df, use_container_width=True)

        # Offer download of the Excel file if present
        excel_path = pathlib.Path("Predicta_Top10.xlsx")
        if excel_path.exists():
            with excel_path.open("rb") as f:
                st.download_button(
                    label="⬇️ Download Excel (Predicta_Top10.xlsx)",
                    data=f,
                    file_name="Predicta_Top10.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.info("Excel file not found; rerun the screener if needed.")

