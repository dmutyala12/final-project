import streamlit as st
import json
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Define the class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

# Initialize history in session state if it doesn't exist
if 'history' not in st.session_state:
    st.session_state.history = []

def save_result_to_history(filename, predicted_class, confidence):
    st.session_state.history.append({
        'filename': filename,
        'predicted_class': predicted_class,
        'confidence': confidence
    })

def load_users(filename='users.json'):
    if not os.path.isfile(filename):
        return {"users": []}
    with open(filename, "r") as file:
        return json.load(file)

def save_users(users, filename='users.json'):
    with open(filename, "w") as file:
        json.dump(users, file)

# User verification and registration
def verify_login(username, password):
    users = load_users()["users"]
    for user in users:
        if user["username"] == username and user["password"] == password:
            return True
    return False

def register_user(username, password):
    users_data = load_users()
    users = users_data["users"]
    for user in users:
        if user["username"] == username:
            return False
    users.append({"username": username, "password": password})
    save_users(users_data)
    return True

# Page configuration
st.set_page_config(page_title='Leaf Disease Detection', page_icon=':leaves:')

# Custom CSS styles
st.markdown(
    """
    <style>
    .header {
        font-size: 40px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        color: #228B22;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Login'


# Function to display login page
def show_login_page():
    st.title("Login to Leaf Disease Detection")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_login(username, password):
            st.session_state.page = 'Main'
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# Function to display signup page
def show_signup_page():
    st.title("Signup for Leaf Disease Detection")
    new_username = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    if st.button("Signup"):
        if register_user(new_username, new_password):
            st.success("User registered successfully. Please login.")
            st.session_state.page = 'Login'
        else:
            st.error("Username already exists. Please choose another.")

# Function to display the main application

# Main app logic goes here, only if logged in
def show_main_app():
    # Add title and description
    st.markdown('<div class="header">Leaf Disease Detection</div>', unsafe_allow_html=True)
    st.write('Upload leaf images to detect their diseases.')

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(['Upload Images', 'Help', 'About', 'History'])

    with tab1:
        # File uploader for multiple images
        uploaded_files = st.file_uploader('Choose images...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if uploaded_files:
            uploaded_files = list(reversed(uploaded_files))
            
            results_container = st.container()
            
            for uploaded_file in uploaded_files:
                try:
                    img = image.load_img(uploaded_file, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = x / 255.0
                    
                    predictions = model.predict(x)
                    predicted_class = class_labels[np.argmax(predictions)]
                    confidence = np.max(predictions) * 100
                    save_result_to_history(uploaded_file.name, predicted_class, confidence)
                    with results_container:
                        st.markdown(f'<div class="subheader">{uploaded_file.name}</div>', unsafe_allow_html=True)
                        st.image(img, caption='Uploaded Image', use_column_width=True)
                        st.write(f'Predicted Disease: {predicted_class}')
                        st.write(f'Confidence: {confidence:.2f}%')
                        
                        if predicted_class == 'Healthy':
                            st.write('The leaf appears to be healthy.')
                        elif predicted_class == 'Powdery':
                            st.write('The leaf is affected by powdery mildew.')
                        elif predicted_class == 'Rust':
                            st.write('The leaf is affected by rust disease.')
                        
                        fig, ax = plt.subplots()
                        ax.bar(class_labels, predictions[0])
                        ax.set_xlabel('Disease Class')
                        ax.set_ylabel('Probability')
                        ax.set_title('Model Confidence')
                        st.pyplot(fig)
                        
                        img_bytes = BytesIO()
                        fig.savefig(img_bytes, format='png')
                        img_bytes.seek(0)
                        st.download_button(
                            label='Download Result',
                            data=img_bytes,
                            file_name=f'{uploaded_file.name}_result.png',
                            mime='image/png'
                        )
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
     

                    
    with tab2:
        st.markdown('<div class="subheader">Help</div>', unsafe_allow_html=True)
        st.write('1. Click on the "Upload Images" tab.')
        st.write('2. Click on the "Choose images..." button and select one or more leaf images.')
        st.write('3. Wait for the app to process the images and display the results.')
        st.write('4. View the predicted disease, confidence score, and additional information for each image.')
        st.write('5. Use the "Download Result" button to download the result image with the visualization.')
    with tab3:
        st.markdown('<div class="subheader">About</div>', unsafe_allow_html=True)
        st.write('This app is designed to detect diseases in leaf images using a trained machine learning model.')
        st.write('It can identify three types of leaf conditions: Healthy, Powdery, and Rust.')
        st.write('The app provides predictions, confidence scores, and additional information about each detected disease.')
        st.write('It also allows you to download the result image with the visualization of the model\'s confidence.')
        st.write('Please note that the app\'s performance may depend on the quality and clarity of the uploaded images.')
    with tab4:
        st.markdown('<div class="subheader">Prediction History</div>', unsafe_allow_html=True)
        for result in st.session_state.history:
            st.write(f"Filename: {result['filename']}, Predicted Disease: {result['predicted_class']}, Confidence: {result['confidence']:.2f}%")
                    
if st.session_state.page == 'Login':
    show_login_page()
elif st.session_state.page == 'Signup':
    show_signup_page()
else:
    show_main_app()

# Allow navigation between Login and Signup
if st.session_state.page in ['Login', 'Signup']:
    if st.button('Go to Signup' if st.session_state.page == 'Login' else 'Back to Login'):
        st.session_state.page = 'Signup' if st.session_state.page == 'Login' else 'Login'