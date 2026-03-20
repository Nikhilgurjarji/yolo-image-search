import streamlit as st
import sys
from pathlib import Path
from PIL import Image ,ImageDraw , ImageFont
import base64
import json
import io
import time
from src.inference import YOLOv11Inference
from src.utils import save_metadata , load_metadata , get_unique_class_counts


#streamlit run app.py --server.port 8080

sys.path.append(str(Path(__file__).parents))

def img_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered,format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def init_session_state():
    session_defaults = {
        "metadata" : None,
        "unique_classes" : [],
        "count_options" : {},
        "search_results" : [],
        "search_params" :{
            "search_mode" : "Any of the selected classes",
            "selected_classes" : [],
            "thresholds" : {}
        } ,
        "show_boxes" : True,
        "grid_columns" : 3,
        "highlight_matches" : True
    }

    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

st.set_page_config(page_title="YOLOv11 Image search" , layout="wide")
st.title("Computer Vision powered Image Search Application")

# imagecard css 

st.markdown("""
<style>

/* Make columns align nicely */
div[data-testid="column"] {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* Card styling */
.image-card {
    position: relative;
    border-radius: 14px;
    overflow: hidden;
    background: #111;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* Equal height cards */
.image-container {
    width: 100%;
    height: 220px;   /* 🔥 Fixes uneven grid */
    overflow: hidden;
}

/* Image fit */
.image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;   /* keeps images consistent */
    display: block;
}

/* Hover effect */
.image-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}

/* Overlay */
.meta-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 8px 10px;
    font-size: 13px;
    color: white;
    background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
}

/* Optional: make text cleaner */
.meta-overlay strong {
    font-size: 14px;
    color: #00E0FF;
}

/* Reduce spacing issues between Streamlit elements */
.block-container {
    padding-top: 1rem;
}

/* Fix random vertical misalignment */
.element-container {
    margin-bottom: 0.5rem !important;
}

</style>
""", unsafe_allow_html=True)



option = st.radio("choose a option" ,
                  ("Process new images" , "Load existing metadata"),
                  horizontal=True
                  )
if option == "Process new images":
    with st.expander(label="Process New images" , expanded=True):
        col1 , col2 = st.columns(2)
        with col1:
            image_dir = st.text_input("Image Dir path" , placeholder="path/to/image")
        with col2:
            model_path = st.text_input("Model weights path" , "yolo11m.pt")

        if st.button("start Inference"):
            if image_dir:
                try:
                    with st.spinner("Running object detection"):
                        inferencer = YOLOv11Inference(model_path,'cpu')
                        metadata = inferencer.process_directory(image_dir)
                        metadata_path = save_metadata(metadata , image_dir)
                        st.success(f"Processed {len(metadata)} images . metadata saved to : ")
                        st.code(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes , st.session_state.count_options = get_unique_class_counts(metadata)
                except Exception as e:
                    st.error(f"Error During Inference :  ,{str(e)}")
            else:
                st.warning(f"Please enter a directory path")
else:
    with st.expander("Load Existing Metadata", expanded=True):
        metadata_path = st.text_input("Path to metadata" , placeholder="path/to/metadata.json")
        if st.button("Load Metadata"):
            if metadata_path:
                try:
                    with st.spinner("Loading metadata for images"):
                        metadata = load_metadata(metadata_path)
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes , st.session_state.count_options = get_unique_class_counts(metadata)
                        st.success(f"Metadata Loaded for {len(metadata)} images ")
                except Exception as e:
                    st.error(f"Error Loading Metadata : ,{str(e)}")
            else:
                st.warning(f"Please enter matadata path")



if st.session_state.metadata:
    st.header("🔍 Search Engine")
    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Choose option : " , ("Any of the selected classes" , "All the selected classes") ,horizontal=True)
        st.session_state.search_params["selected_classes"] = st.multiselect("Classes to search for", options=st.session_state.unique_classes)
        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Threshold (Optional) : ")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            for i,cls in enumerate(st.session_state.search_params["selected_classes"]) :
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max no of {cls}",
                        options=["None"] + st.session_state.count_options[cls]
                    )
        if st.button("Search Images ",type="primary")and st.session_state.search_params["selected_classes"] :
            results = []
            search_params = st.session_state.search_params

            for item in st.session_state.metadata:
                matches = False
                class_matches = {}

                for cls in search_params["selected_classes"]:
                    class_detections = [d for d in item["detection"] if d["class"] == cls]
                    class_count = len(class_detections)
                    class_matches[cls] = False
                    threshold = search_params["thresholds"].get(cls,"None")

                    if threshold == "None":
                        class_matches[cls] = (class_count>=1)
                    else:
                        class_matches[cls] = (class_count>=1 & class_count<=int(threshold))
                
                if search_params["search_mode"] == "Any of the selected classes" :
                    matches = any(class_matches.values())
                else:
                    matches = all(class_matches.values())
                
                if matches:
                    results.append(item)
            
            st.session_state.search_results = results
            
if st.session_state.search_results:
    results = st.session_state.search_results
    search_params = st.session_state.search_params

    st.subheader(f"📸 Results : {len(results)} Maching classes")
    with st.expander("Display options",expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.show_boxes = st.checkbox("Show Bounding boxes" ,
                        value=st.session_state.show_boxes)
        with cols[1]:
            st.session_state.grid_columns = st.slider("grid_columns" , min_value=2 ,max_value=6,
                        value=st.session_state.grid_columns)
        with cols[2]:
            st.session_state.highlight_matches = st.checkbox("Highlight Matching Classe" ,
                        value=st.session_state.highlight_matches)
    
    grid_cols = st.columns(st.session_state.grid_columns)
    col_index = 0

    for result in results:
        with grid_cols[col_index]:
            try:
                img = Image.open(result["image_path"])
                draw = ImageDraw.Draw(img)

                if st.session_state.show_boxes:
                    try:
                        font = ImageFont.truetype("arial.ttf" ,12)
                    except : 
                        font = ImageFont.load_default()
                    for det in result["detection"]:
                        cls = det["class"]
                        bbox = det["bbox"]

                        if cls in search_params["selected_classes"]:
                            color = "#36D513"
                            thickness = 3
                        elif not st.session_state.highlight_matches:
                            color = "#666666"
                            thickness = 1
                        else:
                            continue
                        
                        draw.rectangle(bbox , outline = color , width = thickness)

                        if cls in search_params["selected_classes"] or not st.session_state.highlight_matches:
                            label = f"{cls}{det["conf"]:.2f}"
                            text_bbox = draw.textbbox((0,0),label ,font)
                            text_width = text_bbox[2] - text_bbox[0] #x2 - x1
                            text_height = text_bbox[3] - text_bbox[1] #y2 - y1

                            draw.rectangle((bbox[0] , bbox[1] , bbox[0] + text_width +8 , bbox[1] + 8 +text_height ),fill = color)

                            draw.text((bbox[0] + 4 , bbox[1] + 2) , label , fill ="white" , font = font)


                meta_items = [f"{key}:{val}"for key , val in result["class_counts"].items() if key in search_params["selected_classes"]]

                st.markdown(f"""
                <div class = "image-card">
                    <div class = "image-container">
                        <img src = "data:image/png;base64,{img_to_base64(img)}">
                    </div>
                    <div class = "meta-overlay">
                        <strong>{Path(result['image_path']).name} </strong><br>
                        {", ".join(meta_items) if meta_items else "No matches"}
                    </div>
                </div>
                """,unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading {result["image_path"]} , {str(e)}")
        
        col_index = (col_index + 1) % st.session_state.grid_columns
    
    with st.expander("Export Options " , expanded=True):
        st.download_button(label="Download Results",
                           data = json.dumps(results ,indent=2),
                           file_name="search_results.json",
                           mime="application/json"
                           )