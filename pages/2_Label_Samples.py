import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils import (
    init_state,
    extract_canvas_points,
    draw_samples_on_image,
    add_world_coordinates,
)

init_state()

st.title("2. Label Sample Points")

if st.session_state.map_image is None:
    st.warning("Please upload a map first.")
    st.stop()

img = st.session_state.map_image
h, w = img.shape[:2]

left, right = st.columns([2, 1])

with right:
    st.subheader("Current label")

    current_label = st.text_input(
        "Label name",
        placeholder="Example: water, highway, forest, building",
    )

    st.write(
        """
        Click points on the map that belong to this label.
        Then press the button below.
        """
    )

    add_points = st.button("Add clicked points to label")
    clear_samples = st.button("Clear all samples")

    if clear_samples:
        st.session_state.samples = []
        st.session_state.canvas_version += 1
        st.rerun()

with left:
    st.subheader("Click sample points")

    canvas_background = Image.fromarray(img).convert("RGB")

    canvas_result = st_canvas(
        background_image=canvas_background,
        background_color="#FFFFFF",
        height=h,
        width=w,
        drawing_mode="point",
        stroke_width=8,
        stroke_color="#FF0000",
        fill_color="rgba(255, 0, 0, 0.35)",
        update_streamlit=True,
        key=f"label_canvas_{st.session_state.canvas_version}",
    )
    
if add_points:
    if not current_label.strip():
        st.error("Please enter a label first.")
    else:
        points = extract_canvas_points(canvas_result.json_data, w, h)

        if len(points) == 0:
            st.error("No points found. Click points on the map first.")
        else:
            for x, y in points:
                r, g, b = img[y, x].tolist()

                st.session_state.samples.append(
                    {
                        "label": current_label.strip(),
                        "x": x,
                        "y": y,
                        "r": r,
                        "g": g,
                        "b": b,
                    }
                )

            st.success(f"Added {len(points)} points for label: {current_label}")
            st.session_state.canvas_version += 1
            st.rerun()

samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("Collected samples")

if len(samples_df) == 0:
    st.info("No samples added yet.")
else:
    preview_img = draw_samples_on_image(img, samples_df)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.image(preview_img, caption="Labelled sample points", use_container_width=True)

    with c2:
        st.dataframe(samples_df, use_container_width=True)

        st.download_button(
            "Download samples CSV",
            data=samples_df.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples.csv",
            mime="text/csv",
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