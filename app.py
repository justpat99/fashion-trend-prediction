import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import os
import torch.nn as nn

# 1. Load the Fine-tuned Model

model = torch.hub.load('pytorch/vision:main', 'efficientnet_b0', pretrained=False)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 5)  #5 classes 
model.load_state_dict(torch.load('styles_fine_tuning.pth', map_location=torch.device('cpu'))) # Load to CPU
model.eval()

# Define the image transformations (same as in training/validation)
transform = T.Compose([
    T.Resize(256),       # Resize for EfficientNet
    T.CenterCrop(224),   # Center crop for consistent input size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

classes = ['athleisure', 'fairycore', 'old money' , 'streetwear', 'y2k']  # Class names (same as in training)

# 2. Create the Streamlit App

st.title("Trendcatcher: Fashion Style Predictor")
st.markdown("""
Let AI predict which of the trending styles your outfit belongs to!

#### 5 Trending Styles:
- **Athleisure**
- **Fairycore**
- **Old money**
- **Streetwear**
- **Y2K**
""")

uploaded_file = st.file_uploader("Upload your outfit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        with st.spinner("Classifying..."):  # Show a spinner while processing
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0) # Softmax for probabilities
                predicted_class_index = torch.argmax(probabilities).item()
                predicted_class = classes[predicted_class_index]
                confidence = probabilities[predicted_class_index].item() * 100

            st.header("Style Prediction")
            st.write(f"The image is {predicted_class} style with {confidence:.2f}% confidence.")

            # Display probabilities for each class 
            st.subheader("Class Probabilities")
            for i, class_name in enumerate(classes):
              st.write(f"{class_name}: {probabilities[i].item()*100:.2f}%")
