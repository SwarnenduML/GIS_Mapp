import hashlib

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from utils import (
    init_state,
    draw_samples_on_image,
    add_world_coordinates,
)


init_state()

st.title("2. Label Sample Points")

if st.session_state.map_image is None:
    st.warning("Please upload or fetch a map first.")
    st.stop()

img = st.session_state.map_image
h, w = img.shape[:2]


def invalidate_classification_outputs():
    """Remove outputs that no longer match the current training samples."""
    for key in (
        "overlay_image",
        "mask_image",
        "summary_df",
        "classification_mask",
        "classification_confidence",
        "class_names",
        "shapefile_zip",
        "shapefile_summary_df",
    ):
        st.session_state[key] = None


# Clear stale click state when a different map is loaded.
map_hash = hashlib.md5(img.tobytes()).hexdigest()

if st.session_state.get("_label_map_hash") != map_hash:
    st.session_state["_label_map_hash"] = map_hash
    st.session_state["_label_message"] = None
    st.session_state.pop("map_click", None)


st.subheader("Current label")

control_col, undo_col, clear_col = st.columns([3, 1, 1])

with control_col:
    st.text_input(
        "Label name",
        key="sample_label",
        placeholder="Example: water, highway, forest, building",
        help="Enter a label first. Every subsequent map click is saved to this label.",
    )

with undo_col:
    st.write("")
    st.write("")
    if st.button("Undo last", use_container_width=True):
        if st.session_state.samples:
            removed = st.session_state.samples.pop()
            invalidate_classification_outputs()
            st.session_state["_label_message"] = (
                "success",
                f"Removed the last point from label: {removed['label']}",
            )
            st.rerun()
        else:
            st.session_state["_label_message"] = (
                "warning",
                "There are no sample points to remove.",
            )

with clear_col:
    st.write("")
    st.write("")
    if st.button("Clear all", use_container_width=True):
        st.session_state.samples = []
        invalidate_classification_outputs()
        st.session_state["_label_message"] = (
            "success",
            "All sample points were cleared.",
        )
        st.rerun()


# Display-width control. The image is resized explicitly, so clicks can be
# converted accurately back to coordinates in the working classification image.
if w > 450:
    default_width = min(w, 900)
    display_width = st.slider(
        "Map display width",
        min_value=450,
        max_value=w,
        value=default_width,
        step=25,
    )
else:
    display_width = w

display_height = max(1, int(round(h * display_width / w)))

samples_df = pd.DataFrame(st.session_state.samples)

if len(samples_df) > 0:
    clickable_img = draw_samples_on_image(img, samples_df)
else:
    clickable_img = img.copy()

clickable_pil = Image.fromarray(clickable_img).resize(
    (display_width, display_height),
    Image.Resampling.LANCZOS,
)


def add_clicked_point():
    """Save one newly clicked point using coordinates from the displayed image."""
    click = st.session_state.get("map_click")
    label = st.session_state.get("sample_label", "").strip()

    if not click:
        return

    if not label:
        st.session_state["_label_message"] = (
            "error",
            "Enter a label before clicking the map.",
        )
        return

    try:
        display_x = float(click["x"])
        display_y = float(click["y"])
    except (KeyError, TypeError, ValueError):
        st.session_state["_label_message"] = (
            "error",
            "The click coordinates could not be read.",
        )
        return

    # Convert from displayed-image coordinates to working-image coordinates.
    x = int(round(display_x * w / display_width))
    y = int(round(display_y * h / display_height))

    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    r, g, b = img[y, x].tolist()

    st.session_state.samples.append(
        {
            "label": label,
            "x": x,
            "y": y,
            "r": int(r),
            "g": int(g),
            "b": int(b),
        }
    )

    invalidate_classification_outputs()
    st.session_state["_label_message"] = (
        "success",
        f"Added point ({x}, {y}) to label: {label}",
    )


st.subheader("Click sample points")

st.caption(
    "Enter a label above and click directly on the visible map. "
    "Each click is saved immediately."
)

streamlit_image_coordinates(
    clickable_pil,
    width=display_width,
    key="map_click",
    cursor="crosshair",
    on_click=add_clicked_point,
)

message = st.session_state.get("_label_message")
if message:
    level, text = message
    if level == "error":
        st.error(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.success(text)


samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("Collected samples")

if len(samples_df) == 0:
    st.info("No samples added yet.")
else:
    preview_img = draw_samples_on_image(img, samples_df)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.image(
            preview_img,
            caption="Labelled sample points",
            use_container_width=True,
        )

    with c2:
        st.dataframe(samples_df, use_container_width=True)

        st.download_button(
            "Download samples CSV",
            data=samples_df.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.world_corners is not None:
        geo_samples = add_world_coordinates(
            samples_df,
            st.session_state.world_corners,
            w,
            h,
        )

        st.subheader("Samples with real-world coordinates")
        st.dataframe(geo_samples, use_container_width=True)

        st.download_button(
            "Download georeferenced samples CSV",
            data=geo_samples.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples_georeferenced.csv",
            mime="text/csv",
        )