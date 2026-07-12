import pandas as pd
import streamlit as st

from utils import (
    classification_to_shapefile_zip,
    image_to_png_bytes,
    init_state,
)

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


# -----------------------------------------------------------------------------
# Georeferenced polygon/Shapefile export
# -----------------------------------------------------------------------------

st.divider()
st.subheader("Download classified polygons as an ESRI Shapefile")

if st.session_state.world_corners is None:
    st.warning(
        "A georeferenced Shapefile needs real-world corner coordinates. "
        "Load the map from OpenStreetMap or upload it again with all four "
        "longitude/latitude corner coordinates."
    )
elif (
    st.session_state.classification_mask is None
    or st.session_state.class_names is None
):
    st.warning(
        "The stored classification predates Shapefile support. "
        "Run the classification once more, then return to this page."
    )
else:
    st.caption(
        "The raster classes are converted into vector polygons in WGS 84 "
        "longitude/latitude (EPSG:4326). Because a Shapefile consists of "
        "multiple files, the download is provided as one ZIP archive."
    )

    c1, c2 = st.columns(2)

    with c1:
        min_region_pixels = st.number_input(
            "Minimum connected-region size (pixels)",
            min_value=1,
            max_value=1_000_000,
            value=25,
            step=10,
            help=(
                "Small isolated regions below this size are omitted. "
                "Increase this value when the classified map is noisy."
            ),
        )

    with c2:
        simplify_tolerance = st.number_input(
            "Polygon simplification tolerance (pixels)",
            min_value=0.0,
            max_value=25.0,
            value=1.5,
            step=0.5,
            help=(
                "Higher values create smaller, smoother Shapefiles with fewer "
                "vertices. Set to 0 to preserve the original pixel boundary."
            ),
        )

    include_unknown = st.checkbox(
        "Include unknown / low-confidence areas",
        value=False,
    )

    if st.button("Prepare Shapefile ZIP", type="primary"):
        try:
            with st.spinner("Converting classified pixels to vector polygons..."):
                zip_bytes, polygon_summary = classification_to_shapefile_zip(
                    pred_mask=st.session_state.classification_mask,
                    classes=st.session_state.class_names,
                    world_corners=st.session_state.world_corners,
                    confidence_mask=st.session_state.classification_confidence,
                    min_region_pixels=int(min_region_pixels),
                    simplify_tolerance_pixels=float(simplify_tolerance),
                    include_unknown=include_unknown,
                    basename="classified_map",
                )

            st.session_state.shapefile_zip = zip_bytes
            st.session_state.shapefile_summary_df = polygon_summary
            st.success(
                f"Shapefile prepared with {len(polygon_summary):,} polygon features."
            )
        except Exception as exc:
            st.session_state.shapefile_zip = None
            st.session_state.shapefile_summary_df = None
            st.error(f"Could not create the Shapefile: {exc}")

    if st.session_state.shapefile_summary_df is not None:
        st.write("Polygon feature summary")
        st.dataframe(
            st.session_state.shapefile_summary_df,
            use_container_width=True,
        )

    if st.session_state.shapefile_zip is not None:
        st.download_button(
            "Download classified map Shapefile ZIP",
            data=st.session_state.shapefile_zip,
            file_name="classified_map_shapefile.zip",
            mime="application/zip",
        )


if len(st.session_state.samples) > 0:
    st.divider()
    samples_df = pd.DataFrame(st.session_state.samples)

    st.subheader("Training samples")
    st.dataframe(samples_df, use_container_width=True)

    st.download_button(
        "Download training samples CSV",
        data=samples_df.to_csv(index=False).encode("utf-8"),
        file_name="map_label_samples.csv",
        mime="text/csv",
    )