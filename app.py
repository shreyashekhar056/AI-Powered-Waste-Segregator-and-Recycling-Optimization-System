import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import time

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AI Waste Segregator",
    page_icon="♻️",
    layout="wide", 
    initial_sidebar_state="collapsed",
)

# --- 2. ENHANCED CSS (Branding + Intro Animations) ---
st.markdown("""
    <style>
    /* Hide Sidebar */
    [data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    .main { background-color: #0e1117; }
    
    /* Title Styling */
    .main-title { 
        color: #4CAF50; 
        font-family: 'Helvetica Neue', sans-serif; 
        text-align: center; 
        font-size: 42px;
        font-weight: bold;
        padding-bottom: 10px;
    }
    
    /* Realistic Falling Animation */
    @keyframes fall {
        0% { transform: translateY(-50px) rotate(0deg); opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { transform: translateY(120px) rotate(20deg); opacity: 0; }
    }
    .falling-object {
        display: block;
        width: 80px;
        height: auto;
        margin: 0 auto -40px auto;
        animation: fall 3s infinite ease-in;
    }

    .bin-visual {
        font-size: 50px;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #333;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
    }
    .bin-label { font-size: 14px; color: #aaa; margin-top: 5px; text-align: center; font-weight: bold; }

    /* Fix camera width for side-by-side balance */
    [data-testid="stCameraInput"] { width: 100% !important; max-width: 420px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "saved_models/best_model.pth"
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train your model first.")
        return None, None, device

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, class_names, device

model, class_names, device = load_model()

# --- 4. RECYCLING GUIDELINES ---
def get_eco_advice(label):
    db = {
        "cardboard": {"recyclable": True, "info": "YES: High recyclability. Keep it dry.", "impact": "Saves 17 trees per ton."},
        "paper": {"recyclable": True, "info": "YES: Clean paper only.", "impact": "Reduces landfill waste by 40%."},
        "glass": {"recyclable": True, "info": "YES: Rinse well.", "impact": "Infinitely recyclable."},
        "metal": {"recyclable": True, "info": "YES: Clean cans and foil.", "impact": "Saves 95% energy vs raw production."},
        "plastic": {"recyclable": "Conditional", "info": "CHECK: Type 1 & 2 usually accepted.", "impact": "Prevents ocean pollution."},
        "trash": {"recyclable": False, "info": "NO: General waste bin.", "impact": "Try to reduce usage."}
    }
    return db.get(label, {"recyclable": "Unknown", "info": "Consult local guidelines.", "impact": "N/A"})

# --- 5. THE WEBSITE STRUCTURE ---
def main():
    # Session State to manage page navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'intro'

    # --- INTRO PAGE ---
    if st.session_state.page == 'intro':
        st.markdown("<div class='main-title'>AI-Powered Waste Segregation and Recycling Optimization System</div>", unsafe_allow_html=True)
        
        # Visual Animation Section
        st.write("### ♻️ Intelligent Sorting in Action")
        v_col1, v_col2, v_col3 = st.columns(3)
        
        # Visual Assets (Hosted Icons for Realism)
        img_plastic = "https://cdn-icons-png.flaticon.com/512/81/81929.png" # Bottle
        img_paper = "https://cdn-icons-png.flaticon.com/512/2541/2541991.png" # Box
        img_metal = "https://cdn-icons-png.flaticon.com/512/2659/2659360.png" # Can

        with v_col1:
            st.markdown(f"<img src='{img_plastic}' class='falling-object'>", unsafe_allow_html=True)
            st.markdown(f"<div class='bin-visual' style='border-color: #4CAF50;'>🗑️<div class='bin-label'>WET WASTE</div></div>", unsafe_allow_html=True)
        
        with v_col2:
            st.markdown(f"<img src='{img_paper}' class='falling-object' style='animation-delay: 1s;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='bin-visual' style='border-color: #2196F3;'>🗑️<div class='bin-label'>DRY WASTE</div></div>", unsafe_allow_html=True)
            
        with v_col3:
            st.markdown(f"<img src='{img_metal}' class='falling-object' style='animation-delay: 0.5s;'>", unsafe_allow_html=True)
            st.markdown(f"<div class='bin-visual' style='border-color: #FFC107;'>🗑️<div class='bin-label'>METALLIC WASTE</div></div>", unsafe_allow_html=True)

        st.write("---")
        
        col_text, col_stats = st.columns([1, 1], gap="large")
        with col_text:
            st.write("### Why Automated Segregation?")
            st.write("""
                Manual waste sorting is inefficient and leads to high contamination rates in recycling plants. 
                Our system utilizes **ResNet-18 Deep Learning** architecture to categorize waste with high precision.
                
                By optimizing the sorting process at the source, we can:
                * **Reduce Landfill Volume:** Ensure recyclable materials don't end up in the ground.
                * **Increase Purity:** Keep 'Wet' waste from contaminating 'Dry' recyclables.
                * **Energy Efficiency:** Lower the carbon footprint of recycling facilities.
            """)
            if st.button("🚀 Launch AI Scanner", use_container_width=True):
                st.session_state.page = 'app'
                st.rerun()
        
        with col_stats:
            st.info("💡 **Dry Waste:** Paper and cardboard recycling saves up to 60% of energy compared to virgin production.")
            st.info("💡 **Metals:** Aluminum can be recycled indefinitely without losing its properties.")
            st.info("💡 **Plastics:** Effective segregation prevents microplastics from entering our water systems.")

    # --- MAIN SCANNER PAGE ---
    elif st.session_state.page == 'app':
        st.markdown("<div class='main-title'>AI Waste Segregator</div>", unsafe_allow_html=True)
        
        # Navigation Integrated at the top
        menu = st.selectbox("Navigation Menu", ["🏠 Home / Scanner", "📊 Statistics", "💡 Tips"], label_visibility="collapsed")
        
        if st.button("⬅️ Back to Introduction"):
            st.session_state.page = 'intro'
            st.rerun()
            
        st.write("---")

        if menu == "🏠 Home / Scanner":
            col_input, col_analysis = st.columns([1, 1], gap="large")

            with col_input:
                st.write("### 📸 Input Source")
                input_type = st.radio("Choose Method:", ["Upload Image", "Real-time Camera"], horizontal=True)
                
                source = None
                if input_type == "Upload Image":
                    source = st.file_uploader("Upload waste photo", type=["jpg", "png", "jpeg"])
                else:
                    source = st.camera_input("Snapshot for Analysis")

            with col_analysis:
                st.write("### 🔍 Real-time Analysis")
                if source:
                    img = Image.open(source).convert("RGB")
                    
                    preprocess = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(tensor)
                        probs = torch.nn.functional.softmax(output[0], dim=0)
                        conf, idx = torch.max(probs, dim=0)
                    
                    label = class_names[idx.item()]
                    advice = get_eco_advice(label)

                    st.info(f"Detected Material: **{label.upper()}**")
                    st.write(f"Confidence Level: **{conf*100:.1f}%**")
                    st.progress(float(conf))
                    
                    if advice['recyclable'] is True:
                        st.success(f"**Status:** {advice['info']}")
                    elif advice['recyclable'] == "Conditional":
                        st.warning(f"**Status:** {advice['info']}")
                    else:
                        st.error(f"**Status:** {advice['info']}")
                    
                    with st.expander("🌍 Environmental Impact"):
                        st.write(advice['impact'])
                    
                    st.image(img, caption="Captured Image", width=250)
                else:
                    st.markdown("> **Awaiting Input...**")
                    st.caption("Please provide an image on the left to begin segregation.")

        elif menu == "📊 Statistics":
            st.title("Environmental Impact Dashboard")
            st.bar_chart({"Recyclable": [70], "Landfill": [30]})

        elif menu == "💡 Tips":
            st.title("Disposal Best Practices")
            st.info("Tip: Always rinse containers. Food residue can ruin an entire batch of recycling!")

if __name__ == "__main__":
    main()