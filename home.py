import streamlit as st

st.set_page_config(
    page_title="Interactive Map Mapper",
    layout="wide"
)

st.title("Interactive Map Mapper")

st.write(
    """
    This app lets you upload a map image, optionally add real-world corner 
    coordinates, click labelled sample points, and automatically classify 
    the whole map based on pixel values.
    """
)

st.header("Workflow")

st.markdown(
    """
    1. **Upload Map**  
       Upload a map image and optionally define its four real-world corner coordinates.

    2. **Label Samples**  
       Enter labels such as water, highway, forest, building, etc.  
       Then click points on the map that belong to that label.

    3. **Run Classification**  
       Train a simple pixel-based classifier and automatically segment the full map.

    4. **Export Results**  
       Download the classified map, overlay, and area summary.
    """
)

st.info("Use the sidebar to move between pages.")