import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(page_title="Generative VLM - Image Captioning", layout="wide")

st.title("📸 Image Describing with Fine-Tuned BLIP-2")
st.write("Upload an image and generate rich, creative descriptions using a fine-tuned vision-language model.")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # Replace with fine-tuned model path if available
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

col1, col2, col3 = st.columns(3)
with col1:
    decoding_method = st.selectbox("Decoding Strategy", ["Beam Search", "Top-k Sampling", "Top-p Sampling"])
with col2:
    num_beams = st.slider("Beam Width", 1, 10, 5) if decoding_method == "Beam Search" else 1
    top_k = st.slider("Top-k", 1, 100, 50) if decoding_method == "Top-k Sampling" else None
with col3:
    top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.9) if decoding_method == "Top-p Sampling" else None
    temperature = st.slider("Temperature", 0.5, 2.0, 1.0)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = processor(image, return_tensors="pt").to(model.device)

    generate_kwargs = {
        "max_new_tokens": 50,
        "temperature": temperature,
    }

    if decoding_method == "Beam Search":
        generate_kwargs["num_beams"] = num_beams
    elif decoding_method == "Top-k Sampling":
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_k"] = top_k
    elif decoding_method == "Top-p Sampling":
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.subheader("📝 Generated Caption:")
    st.success(caption)
