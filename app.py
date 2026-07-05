import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from rembg import remove

# 1. Page Configuration
st.set_page_config(page_title="Plant Disease Diagnostician", layout="centered")
st.title("🌿 Plant Leaf Disease Detector")
st.write("Upload a photo of an infected plant leaf to get an instant diagnosis and remedy.")

# 2. Load the Model & Labels
@st.cache_resource
def load_plant_model():
    # Load your Teachable Machine model
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    
    # Load labels mapping
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

try:
    model, class_names = load_plant_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'keras_model.h5' and 'labels.txt' are in the same folder.")

# 3. Knowledge Base for Remedies
# DISEASE_GUIDE = {
#     "Tomato_Healthy": {
#         "reasons": ["Your tomato plant is thriving with proper sunlight and watering."],
#         "remedies": ["Maintain current care routine."]
#     },
#     "Tomato_Mosaic_Virus": {
#         "reasons": ["Spread by aphids, whiteflies, or contaminated tools/hands."],
#         "remedies": ["Remove and destroy infected plants immediately.", "Control aphid populations.", "Sanitize tools regularly."]
#     },
#     "Tomato_Yellow_Leaf_Curl_Virus": {
#         "reasons": ["Transmitted primarily by silverleaf whiteflies."],
#         "remedies": ["Use whitefly traps or neem oil.", "Remove nearby weeds that host the virus."]
#     },
#     "Tomato_Target_Spot": {
#         "reasons": ["Fungus Corynespora cassiicola, favored by warm, humid weather."],
#         "remedies": ["Apply fungicides containing copper.", "Improve plant spacing for airflow."]
#     },
#     "Tomato_Spider_Mites": {
#         "reasons": ["Tiny pests that thrive in hot, dry conditions."],
#         "remedies": ["Spray plants with a strong stream of water to knock them off.", "Apply insecticidal soap or neem oil."]
#     },
#     "Tomato_Septoria_Leaf_Spot": {
#         "reasons": ["Fungus Septoria lycopersici, spreads via splashing water."],
#         "remedies": ["Mulch the base to prevent soil splash.", "Avoid overhead watering."]
#     },
#     "Tomato_Leaf_Mold": {
#         "reasons": ["Fungus Passalora fulva, caused by high humidity in poorly ventilated spaces."],
#         "remedies": ["Increase ventilation.", "Prune lower leaves to decrease humidity around the base."]
#     },
#     "Tomato_Late_Blight": {
#         "reasons": ["Oomycete Phytophthora infestans, highly destructive in cool, wet weather."],
#         "remedies": ["Destroy infected plants (do not compost).", "Apply preventative copper fungicides."]
#     },
#     "Tomato_Early_Blight": {
#         "reasons": ["Fungus Alternaria solani, thrives in warm, humid conditions."],
#         "remedies": ["Prune lower leaves.", "Apply fungicides early in the season."]
#     },
#     "Tomato_Bacterial_Spot": {
#         "reasons": ["Bacterial pathogen Xanthomonas, spreads during warm, rainy periods."],
#         "remedies": ["Avoid working among wet plants.", "Use copper-based bactericides."]
#     },
#     "Potato_Healthy": {
#         "reasons": ["The potato plant is healthy and vigorous."],
#         "remedies": ["Continue standard hilling and watering practices."]
#     },
#     "Potato_Late_Blight": {
#         "reasons": ["Same pathogen as tomato late blight, loves cool, wet conditions."],
#         "remedies": ["Remove infected foliage.", "Harvest tubers only in dry weather."]
#     },
#     "Potato_Early_Blight": {
#         "reasons": ["Fungal pathogen causing dark, concentric rings on older leaves."],
#         "remedies": ["Maintain adequate soil fertility.", "Apply specific fungal treatments if severe."]
#     },
#     "Pepper_Bell_Healthy": {
#         "reasons": ["The pepper plant shows excellent health."],
#         "remedies": ["Ensure consistent soil moisture as peppers fruit."]
#     },
#     "Pepper_Bell_Bacterial_Spot": {
#         "reasons": ["Bacterial infection favored by high moisture and warm temperatures."],
#         "remedies": ["Rotate crops annually.", "Apply copper sprays at first sign of spots."]
#     }
# }

DISEASE_GUIDE = {
  "Tomato Healthy": {
    "reasons": [
      "The plant receives adequate sunlight (6–8 hours daily).",
      "Proper watering keeps the soil consistently moist but not waterlogged.",
      "Nutrient levels are balanced, especially nitrogen, phosphorus, and potassium.",
      "Good airflow prevents fungal growth and disease development.",
      "No visible signs of pests, discoloration, wilting, or leaf lesions."
    ],
    "remedies": [
      "Continue current watering and fertilization practices.",
      "Inspect leaves weekly for early signs of pests or diseases.",
      "Remove weeds around the plant.",
      "Apply mulch to conserve moisture and suppress weeds.",
      "Support plants with stakes or cages to improve airflow."
    ]
  },
  "Tomato Mosaic Virus": {
    "reasons": [
      "Caused by Tobacco Mosaic Virus (TMV), a highly contagious viral disease.",
      "Spread through contaminated gardening tools and equipment.",
      "Handling infected plants and then touching healthy plants.",
      "Tobacco products carried on hands or clothing can transmit the virus.",
      "Infected seeds and plant debris can harbor the virus.",
      "Aphids and other sap-sucking insects may contribute to transmission."
    ],
    "remedies": [
      "Remove and destroy infected plants immediately.",
      "Do not compost infected plant material.",
      "Use certified disease-free seeds.",
      "Disinfect tools with a 10% bleach solution or alcohol.",
      "Wash hands before handling plants.",
      "Avoid smoking while working with tomato plants.",
      "Control aphids and other insect vectors.",
      "Rotate crops every 2–3 years.",
      "Plant resistant tomato varieties when available."
    ]
  },
  "Tomato Yellow Leaf Curl Virus": {
    "reasons": [
      "Caused by Tomato Yellow Leaf Curl Virus (TYLCV).",
      "Primarily transmitted by silverleaf whiteflies.",
      "Warm temperatures favor whitefly populations and disease spread.",
      "Nearby infected weeds serve as virus reservoirs.",
      "Heavy whitefly infestations significantly increase infection risk."
    ],
    "remedies": [
      "Install yellow sticky traps to monitor and reduce whitefly populations.",
      "Apply neem oil regularly.",
      "Use insecticidal soap against whiteflies.",
      "Remove and destroy infected plants.",
      "Eliminate nearby weeds that may host the virus.",
      "Use reflective mulches to repel whiteflies.",
      "Grow TYLCV-resistant tomato varieties.",
      "Use row covers during early plant growth."
    ]
  },
  "Tomato Target Spot": {
    "reasons": [
      "Caused by the fungus Corynespora cassiicola.",
      "Warm temperatures between 25°C and 30°C favor infection.",
      "High humidity promotes fungal growth.",
      "Poor air circulation around plants increases disease pressure.",
      "Extended periods of leaf wetness encourage infection."
    ],
    "remedies": [
      "Space plants properly to improve airflow.",
      "Remove infected leaves and plant debris.",
      "Avoid overhead irrigation.",
      "Apply copper-based fungicides.",
      "Use chlorothalonil or mancozeb fungicides when necessary.",
      "Practice crop rotation.",
      "Remove crop debris after harvest."
    ]
  },
  "Tomato Spider Mites": {
    "reasons": [
      "Spider mites thrive in hot and dry environmental conditions.",
      "Dusty growing conditions encourage infestations.",
      "Plants under water or nutrient stress are more susceptible.",
      "Mites feed on plant sap, weakening leaves and reducing vigor."
    ],
    "remedies": [
      "Spray plants with a strong stream of water to dislodge mites.",
      "Apply insecticidal soap.",
      "Use neem oil or horticultural oils.",
      "Introduce predatory mites for biological control.",
      "Encourage beneficial insects such as ladybugs.",
      "Maintain adequate humidity levels.",
      "Reduce plant stress through proper watering."
    ]
  },
  "Tomato Septoria Leaf Spot": {
    "reasons": [
      "Caused by the fungus Septoria lycopersici.",
      "Spreads through splashing rain or irrigation water.",
      "Overhead watering increases disease transmission.",
      "Contaminated tools and infected crop residue can spread spores.",
      "Warm, wet conditions favor disease development."
    ],
    "remedies": [
      "Remove infected leaves promptly.",
      "Destroy infected plant debris.",
      "Water at the soil level instead of overhead.",
      "Apply mulch to reduce soil splash.",
      "Practice crop rotation.",
      "Apply copper fungicides when needed.",
      "Use chlorothalonil or mancozeb fungicides for severe infections."
    ]
  },
  "Tomato Leaf Mold": {
    "reasons": [
      "Caused by the fungus Passalora fulva.",
      "Common in greenhouses and enclosed growing spaces.",
      "High humidity above 85% promotes disease development.",
      "Poor ventilation creates ideal conditions for fungal growth.",
      "Dense foliage increases humidity around leaves."
    ],
    "remedies": [
      "Increase greenhouse or garden ventilation.",
      "Reduce humidity levels around plants.",
      "Space plants properly.",
      "Prune lower foliage to improve airflow.",
      "Apply copper fungicides.",
      "Use sulfur-based products if appropriate.",
      "Water early in the day so foliage dries quickly."
    ]
  },
  "Tomato Late Blight": {
    "reasons": [
      "Caused by the oomycete Phytophthora infestans.",
      "Cool temperatures between 10°C and 24°C favor disease spread.",
      "High humidity and prolonged rainfall promote infection.",
      "Spores spread rapidly through wind and water.",
      "The disease can infect leaves, stems, and fruits."
    ],
    "remedies": [
      "Remove and destroy infected plants immediately.",
      "Do not compost infected material.",
      "Apply preventative copper fungicides.",
      "Improve airflow around plants.",
      "Avoid overhead watering.",
      "Monitor weather conditions for early disease risk.",
      "Rotate crops regularly.",
      "Remove volunteer tomato and potato plants."
    ]
  },
  "Tomato Early Blight": {
    "reasons": [
      "Caused by the fungus Alternaria solani.",
      "Warm temperatures and humid weather encourage infection.",
      "Nutrient-deficient plants are more susceptible.",
      "Old infected plant debris serves as a source of spores.",
      "Disease often begins on older, lower leaves."
    ],
    "remedies": [
      "Remove infected lower leaves.",
      "Improve airflow through proper spacing.",
      "Maintain balanced fertilization.",
      "Correct potassium deficiencies.",
      "Apply copper-based fungicides.",
      "Use chlorothalonil or mancozeb fungicides.",
      "Practice crop rotation.",
      "Remove infected debris after harvest."
    ]
  },
  "Tomato Bacterial Spot": {
    "reasons": [
      "Caused by Xanthomonas bacterial species.",
      "Spreads through rain splash and irrigation water.",
      "Wind-driven rain can move bacteria between plants.",
      "Contaminated tools and infected seeds contribute to spread.",
      "Warm and wet conditions favor disease development."
    ],
    "remedies": [
      "Avoid handling wet plants.",
      "Disinfect gardening tools regularly.",
      "Apply copper-based bactericides.",
      "Use copper and mancozeb combinations when recommended.",
      "Use certified disease-free seeds.",
      "Practice crop rotation.",
      "Ensure adequate plant spacing."
    ]
  },
  "Potato Healthy": {
    "reasons": [
      "The plant receives adequate nutrients and water.",
      "Healthy foliage supports strong tuber development.",
      "Proper hilling protects developing tubers.",
      "No visible signs of pests, diseases, or nutrient deficiencies."
    ],
    "remedies": [
      "Continue regular watering practices.",
      "Hill soil around stems as needed.",
      "Monitor for blight and insect pests.",
      "Apply balanced fertilizers according to soil needs."
    ]
  },
  "Potato Late Blight": {
    "reasons": [
      "Caused by Phytophthora infestans, the same pathogen responsible for tomato late blight.",
      "Cool and wet weather promotes disease outbreaks.",
      "Spores spread through wind, rain, and infected plant material.",
      "The pathogen can infect leaves, stems, and tubers."
    ],
    "remedies": [
      "Remove infected foliage immediately.",
      "Destroy severely infected plants.",
      "Apply copper-based fungicides.",
      "Use preventative fungicide programs during favorable conditions.",
      "Harvest tubers only in dry weather.",
      "Allow tubers to cure before storage.",
      "Use certified seed potatoes.",
      "Maintain adequate plant spacing."
    ]
  },
  "Potato Early Blight": {
    "reasons": [
      "Caused by the fungus Alternaria solani.",
      "Warm weather favors disease development.",
      "Nutrient stress increases susceptibility.",
      "Older plants are more commonly affected.",
      "Infected crop residue serves as a source of fungal spores."
    ],
    "remedies": [
      "Remove infected foliage.",
      "Improve soil fertility.",
      "Maintain adequate nitrogen and potassium levels.",
      "Apply chlorothalonil fungicides.",
      "Use mancozeb or copper-based fungicides when necessary.",
      "Practice crop rotation.",
      "Remove infected crop residues after harvest."
    ]
  },
  "Pepper Bell Healthy": {
    "reasons": [
      "The plant receives sufficient sunlight and nutrients.",
      "Consistent soil moisture supports healthy growth.",
      "No signs of pest infestation or disease are present.",
      "The plant exhibits vigorous foliage and fruit development."
    ],
    "remedies": [
      "Maintain consistent watering.",
      "Use mulch to conserve soil moisture.",
      "Support plants during heavy fruit production.",
      "Continue regular pest and disease monitoring."
    ]
  },
  "Pepper Bell Bacterial Spot": {
    "reasons": [
      "Caused by Xanthomonas bacterial species.",
      "Spread through rain splash and overhead irrigation.",
      "Infected seeds can introduce the disease.",
      "Contaminated tools contribute to bacterial transmission.",
      "Warm and humid conditions favor bacterial growth."
    ],
    "remedies": [
      "Rotate crops every 2–3 years.",
      "Avoid overhead watering.",
      "Increase plant spacing to improve airflow.",
      "Apply copper sprays at the first sign of disease.",
      "Use copper-mancozeb mixtures when appropriate.",
      "Remove infected plant material.",
      "Sterilize gardening tools regularly.",
      "Use certified disease-free seeds."
    ]
  },
  "Unknown": {
        "reasons": ["The uploaded image does not appear to be a plant leaf."],
        "remedies": ["Please upload a clear, well-lit photograph focusing purely on a single infected plant leaf."]
    }
}

# 4. File Uploader UI
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Processing image and isolating leaf..."):
        # Image Processing Layer: Remove Background
        processed_img = remove(image)
        # Convert back to RGB format because rembg outputs RGBA (transparent background)
        processed_img = processed_img.convert("RGB")
        
    with col2:
        st.subheader("Processed Leaf Focus")
        st.image(processed_img, use_container_width=True)

    # Convert image to HSV color space to inspect color distribution
    hsv_image = np.array(processed_img.convert('HSV'))
    hue_channel = hsv_image[:, :, 0]

    # In HSV, greens, yellows, and browns generally fall under hue values < 90
    leaf_pixels = np.sum((hue_channel < 90) & (hue_channel > 10))
    total_pixels = hue_channel.size
    leaf_ratio = leaf_pixels / total_pixels

    # Reject non-leaf data if color parameters aren't met
    if leaf_ratio < 0.12:
        st.error("**Invalid Image Detected:** The uploaded photo doesn't contain standard plant leaf colors (greens/yellows/browns). The AI model has been blocked to prevent a false diagnosis. Please upload a clear photo of an actual leaf.")
    else:
        # 5. Model Inference (Prediction)
        with st.spinner("Running AI Diagnosis..."):
            # Teachable Machine models expect 224x224 images
            size = (224, 224)
            image_resized = ImageOps.fit(processed_img, size, Image.Resampling.LANCZOS)

            # Turn the image into a numpy array and normalize it (Teachable Machine standard preprocessing)
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Add a batch dimension so the shape becomes (1, 224, 224, 3)
            data = np.expand_dims(normalized_image_array, axis=0)

            # Predict
            prediction = model.predict(data)
            index = np.argmax(prediction)
            
            # Clean up label string (Teachable machine prefixes labels with numbers like "0 Tomato_Early_Blight")
            raw_label = class_names[index]
            predicted_class = raw_label.split(" ", 1)[1] if " " in raw_label else raw_label
            confidence_score = prediction[0][index]

        # 6. Display Output Results
        st.write("---")
        if confidence_score > 0.7:  # Threshold to filter out random non-leaf images
            st.success(f"**Diagnosis:** {predicted_class} (Confidence: {confidence_score*100:.2f}%)")
            
            # Pull advice from our guide
            guide = DISEASE_GUIDE.get(predicted_class, {"reasons": ["Information pending updates."], "remedies": ["Consult a local agricultural extension."] })
            
            col_res, col_rem = st.columns(2)
            with col_res:
                st.markdown("### Possible Reasons")
                for reason in guide["reasons"]:
                    st.markdown(f"- {reason}")
                    
            with col_rem:
                st.markdown("### Recommended Remedies")
                for remedy in guide["remedies"]:
                    st.markdown(f"- {remedy}")
        else:
            st.warning("The model is unsure about this image. Please upload a clearer close-up of an infected leaf.")