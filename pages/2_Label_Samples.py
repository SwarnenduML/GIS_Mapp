import hashlib

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from utils import (
    add_world_coordinates,
    draw_samples_on_image,
    init_state,
)


init_state()

st.title("2. Label Sample Points")

if st.session_state.map_image is None:
    st.warning("Please upload or fetch a map first.")
    st.stop()

img = st.session_state.map_image
h, w = img.shape[:2]


# ---------------------------------------------------------------------
# Page/session helpers
# ---------------------------------------------------------------------

def invalidate_classification_outputs() -> None:
    """Remove classification outputs that no longer match the samples."""
    keys = (
        "overlay_image",
        "mask_image",
        "summary_df",
        "classification_mask",
        "classification_confidence",
        "class_names",
        "shapefile_zip",
        "shapefile_summary_df",
    )
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None


map_hash = hashlib.md5(img.tobytes()).hexdigest()

if st.session_state.get("_label_map_hash") != map_hash:
    st.session_state["_label_map_hash"] = map_hash
    st.session_state["_label_click_version"] = 0
    st.session_state["_label_message"] = None

if "_label_click_version" not in st.session_state:
    st.session_state["_label_click_version"] = 0

if "sample_label" not in st.session_state:
    st.session_state.sample_label = ""


# ---------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------

st.subheader("Current label")

label_col, undo_col, clear_col = st.columns([3, 1, 1])

with label_col:
    st.text_input(
        "Label name",
        key="sample_label",
        placeholder="Example: water, highway, forest, building",
    )

with undo_col:
    st.write("")
    st.write("")
    if st.button("Undo last", width="stretch"):
        if st.session_state.samples:
            removed = st.session_state.samples.pop()
            invalidate_classification_outputs()
            st.session_state["_label_click_version"] += 1
            st.session_state["_label_message"] = (
                "success",
                f"Removed the last point from '{removed['label']}'.",
            )
            st.rerun()
        else:
            st.session_state["_label_message"] = (
                "warning",
                "There are no points to remove.",
            )

with clear_col:
    st.write("")
    st.write("")
    if st.button("Clear all", width="stretch"):
        st.session_state.samples = []
        invalidate_classification_outputs()
        st.session_state["_label_click_version"] += 1
        st.session_state["_label_message"] = (
            "success",
            "All sample points were cleared.",
        )
        st.rerun()


message = st.session_state.get("_label_message")
if message is not None:
    level, text = message
    if level == "error":
        st.error(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.success(text)


# ---------------------------------------------------------------------
# Stable click component
#
# Important:
# - The image passed to the component stays unchanged.
# - No on_click callback is used.
# - A click is added only when the user presses the Add button.
# - The component key is reset after a point is added.
#
# This avoids the repeated rerender/resize loop that can occur when the
# component image is changed inside its own callback.
# ---------------------------------------------------------------------

st.subheader("Select a point")

st.caption(
    "Enter a label, click once on the map, and then press "
    "'Add selected point'."
)

display_width = min(w, 900)
display_height = max(1, round(h * display_width / w))

display_img = Image.fromarray(img).resize(
    (display_width, display_height),
    Image.Resampling.LANCZOS,
)

click = streamlit_image_coordinates(
    display_img,
    width=display_width,
    key=(
        f"map_click_{map_hash[:12]}_"
        f"{st.session_state['_label_click_version']}"
    ),
    cursor="crosshair",
    image_format="PNG",
    png_compression_level=6,
)


# ---------------------------------------------------------------------
# Save the selected point
# ---------------------------------------------------------------------

if click is None:
    st.info("Click a point on the map.")
else:
    click_x = float(click.get("x", 0))
    click_y = float(click.get("y", 0))
    rendered_w = max(float(click.get("width", display_width)), 1.0)
    rendered_h = max(float(click.get("height", display_height)), 1.0)

    x = int(round(click_x * w / rendered_w))
    y = int(round(click_y * h / rendered_h))

    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    st.write(f"Selected image coordinate: **x={x}, y={y}**")

    add_disabled = not st.session_state.sample_label.strip()

    if add_disabled:
        st.warning("Enter a label before adding the selected point.")

    if st.button(
        "Add selected point",
        type="primary",
        disabled=add_disabled,
    ):
        label = st.session_state.sample_label.strip()
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

        # Reset the component so the same click cannot be submitted twice.
        st.session_state["_label_click_version"] += 1
        st.session_state["_label_message"] = (
            "success",
            f"Added point ({x}, {y}) to '{label}'.",
        )
        st.rerun()


# ---------------------------------------------------------------------
# Samples and downloads
# ---------------------------------------------------------------------

samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("Collected samples")

if samples_df.empty:
    st.info("No samples added yet.")
else:
    preview_img = draw_samples_on_image(img, samples_df)

    preview_col, table_col = st.columns([2, 1])

    with preview_col:
        st.image(
            preview_img,
            caption="Labelled sample points",
            width="stretch",
        )

    with table_col:
        st.dataframe(samples_df, width="stretch")

        st.download_button(
            "Download samples CSV",
            data=samples_df.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples.csv",
            mime="text/csv",
            width="stretch",
        )

    if st.session_state.world_corners is not None:
        geo_samples = add_world_coordinates(
            samples_df,
            st.session_state.world_corners,
            w,
            h,
        )

        st.subheader("Samples with real-world coordinates")
        st.dataframe(geo_samples, width="stretch")

        st.download_button(
            "Download georeferenced samples CSV",
            data=geo_samples.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples_georeferenced.csv",
            mime="text/csv",
        )