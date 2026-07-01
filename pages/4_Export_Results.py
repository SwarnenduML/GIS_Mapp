import pandas as pd
import streamlit as st

from utils import init_state, image_to_png_bytes

init_state()

st.title("4. Export Results")

if st.session_state.overlay_image is None or st.session_state.mask_image is None:
    st.warning("Please run classification first.")
    st.stop()

st.subheader("Classification overlay")
st.image(st.session_state.overlay_image, use_container_width=True)

st.download_button(
    "Download overlay PNG",
    data=image_to_png_bytes(st.session_state.overlay_image),
    file_name="classified_map_overlay.png",
    mime="image/png",
)

st.subheader("Pure classified mask")
st.image(st.session_state.mask_image, use_container_width=True)

st.download_button(
    "Download classified mask PNG",
    data=image_to_png_bytes(st.session_state.mask_image),
    file_name="classified_map_mask.png",
    mime="image/png",
)

if st.session_state.summary_df is not None:
    st.subheader("Area summary")
    st.dataframe(st.session_state.summary_df, use_container_width=True)

    st.download_button(
        "Download area summary CSV",
        data=st.session_state.summary_df.to_csv(index=False).encode("utf-8"),
        file_name="classification_area_summary.csv",
        mime="text/csv",
    )

if len(st.session_state.samples) > 0:
    samples_df = pd.DataFrame(st.session_state.samples)

    st.subheader("Training samples")
    st.dataframe(samples_df, use_container_width=True)

    st.download_button(
        "Download training samples CSV",
        data=samples_df.to_csv(index=False).encode("utf-8"),
        file_name="map_label_samples.csv",
        mime="text/csv",
    )