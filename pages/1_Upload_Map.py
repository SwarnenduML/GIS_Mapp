import numpy as np
import streamlit as st
from PIL import Image

from utils import init_state, resize_to_max_side, parse_corner_text

init_state()

st.title("1. Upload Map")

st.write(
    """
    Upload a map image. You can also provide the four real-world corner 
    coordinates if you want georeferenced outputs.
    """
)

uploaded_file = st.file_uploader(
    "Upload map image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

max_side = st.slider(
    "Resize working image to max side",
    min_value=500,
    max_value=2000,
    value=1200,
    step=100,
)

use_coordinates = st.checkbox("Add real-world corner coordinates")

world_corners = None

if use_coordinates:
    st.subheader("Corner coordinates")

    st.caption(
        "Enter coordinates as longitude,latitude. "
        "Use the same order as the image corners."
    )

    tl = st.text_input("Top-left lon,lat")
    tr = st.text_input("Top-right lon,lat")
    br = st.text_input("Bottom-right lon,lat")
    bl = st.text_input("Bottom-left lon,lat")

    try:
        if tl and tr and br and bl:
            world_corners = [
                parse_corner_text(tl),
                parse_corner_text(tr),
                parse_corner_text(br),
                parse_corner_text(bl),
            ]
    except Exception:
        st.error("Coordinate format should be longitude,latitude")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = np.array(img)
    img = resize_to_max_side(img, max_side=max_side)

    st.session_state.map_image = img
    st.session_state.map_filename = uploaded_file.name
    st.session_state.world_corners = world_corners
    st.session_state.samples = []
    st.session_state.overlay_image = None
    st.session_state.mask_image = None
    st.session_state.summary_df = None
    st.session_state.canvas_version += 1

    h, w = img.shape[:2]

    st.success("Map uploaded successfully.")
    st.write(f"Working image size: **{w} × {h} pixels**")
    st.image(img, caption="Uploaded map", use_container_width=True)

    if world_corners is not None:
        st.success("Corner coordinates saved.")
        st.write(world_corners)

else:
    st.info("Upload a map image to start.")