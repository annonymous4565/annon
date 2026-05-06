import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("Image Generation Comparison")

# Path to your main data folder
base_path = "./output/test"

def load_data(base_path):
    # Load prompts from text file
    prompt_file = os.path.join(base_path, "prompts.txt")
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # Helper to get sorted image paths
    def get_images(subfolder):
        folder_path = os.path.join(base_path, subfolder)
        files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return files

    sg_images = get_images("sg")
    # sdxl_images = get_images("sdxl")
    raw_images = get_images("images")

    return prompts, sg_images, raw_images

try:
    prompts, sg, images = load_data(base_path)

    # Header Row
    cols = st.columns([2, 2, 2, 2])
    cols[0].write("**Prompt**")
    cols[1].write("**Scene Graph (SG)**")
    # cols[2].write("**SDXL**")
    cols[3].write("**Gen Images**")
    st.divider()

    # Data Rows
    # Iterate based on the number of prompts available
    for i in range(len(prompts)):
        row_cols = st.columns([2, 2, 2, 2])
        
        # Column 1: Prompt
        row_cols[0].info(prompts[i])
        
        # Column 2: SG Image
        if i < len(sg):
            row_cols[1].image(sg[i], use_container_width=True)
            
        # Column 3: SDXL Image
        # if i < len(sdxl):
        #     row_cols[2].image(sdxl[i], use_container_width=True)
            
        # Column 4: Raw Image
        if i < len(images):
            row_cols[3].image(images[i], use_container_width=True)
            
        st.divider()

except FileNotFoundError:
    st.error(f"Could not find the folder at {base_path}. Please check the path.")
