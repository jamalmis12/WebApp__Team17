import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
import math
import cv2
from yolov5 import YOLOv5  # Import YOLOv5
import base64  # Import base64 module
import pydicom  # For DICOM support

# Load the YOLOv5 model (adjust path to your actual best.onnx file)
model = YOLOv5('best.onnx', device='cpu')  # Use 'cuda' if you have a GPU

# Function to calculate the Cobb angle
def calculate_cobb_angle(points):
    (x1, y1), (x2, y2) = points
    if x1 - x2 != 0:
        slope = (y1 - y2) / (x1 - x2)
        angle = math.degrees(math.atan(abs(slope)))
        return angle
    return 0  # In case the line is vertical (undefined slope)

# Function to resize image
def resize_image(image, target_size=(640, 640)):
    return image.resize(target_size, Image.Resampling.LANCZOS)

# DICOM creation function with force reading enabled
def create_dicom_from_image(output_img):
    try:
        dicom_template_path = './static/template.dcm'
        
        try:
            dicom_img = pydicom.dcmread(dicom_template_path, force=True)
        except FileNotFoundError:
            raise Exception(f"Template DICOM file not found at: {dicom_template_path}")
        except Exception as e:
            raise Exception(f"Error reading DICOM template: {e}")
        
        dicom_img.PixelData = np.array(output_img).tobytes()
        
        dicom_io = io.BytesIO()
        dicom_img.save(dicom_io)
        dicom_io.seek(0)
        
        return dicom_io
    except Exception as e:
        st.error(f"Error creating DICOM: {e}")
        return None

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default to the home page

# Create a simple in-memory "database" for user credentials
user_db = {}

# Home page
if st.session_state.page == 'home':
    def image_to_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    image_path = "./static/verr.png"
    encoded_image = image_to_base64(image_path)

    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{encoded_image}" style="width:450px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<h1 style='text-align: center;'>VERTEBRAI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Automated Keypoint Detection for Scoliosis Assessment</h3>", unsafe_allow_html=True)

    if st.button('Get Started', key='get_started', help='Go to login page', use_container_width=True):
        st.session_state.page = 'signup'  # Navigate to the sign-up page

# Sign-up page
elif st.session_state.page == 'signup':
    st.subheader("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if username and password and password == confirm_password:
            if username in user_db:
                st.error("Username already exists. Please try a different one.")
            else:
                user_db[username] = password
                st.success("Sign-up successful! Please log in.")
                st.session_state.page = 'login'  # Redirect to the login page
        else:
            st.error("Please ensure all fields are filled and passwords match.")

# Login page
elif st.session_state.page == 'login':
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in user_db and user_db[username] == password:
            st.session_state.page = 'upload'  # Navigate to the upload page
        else:
            st.error("Invalid credentials. Please try again.")

# Upload page
elif st.session_state.page == 'upload':
    st.subheader("Upload an X-ray Image")
    file = st.file_uploader("Upload an X-ray image of the spine (JPG, PNG)", type=["jpg", "png"])

    if file is not None:
        try:
            img = Image.open(file)
            img_resized = resize_image(img, target_size=(640, 640))
            img_array = np.array(img_resized)
            img_array = img_array[..., ::-1]
            
            results = model.predict(img_array)
            output_img = results.render()[0]
            output_img = np.array(output_img, copy=True)
            
            detections = results.xywh[0].numpy()
            vertebrae_points = []

            for detection in detections:
                x_center, y_center, width, height, confidence, class_id = detection
                if int(class_id) == 0:
                    vertebrae_points.append((int(x_center), int(y_center)))

            cobb_angle = None
            if len(vertebrae_points) >= 3:
                apex_point = vertebrae_points[len(vertebrae_points) // 2]
                upper_point = vertebrae_points[0]
                lower_point = vertebrae_points[-1]
                
                cobb_angle = calculate_cobb_angle([upper_point, lower_point])
                
                cv2.line(output_img, upper_point, lower_point, (255, 0, 255), 2)
                cv2.circle(output_img, apex_point, 10, (0, 255, 255), -1)
                cv2.circle(output_img, upper_point, 5, (255, 255, 0), -1)
                cv2.circle(output_img, lower_point, 5, (0, 0, 255), -1)

                if cobb_angle is not None and cobb_angle != 0:
                    angle_text = f'Cobb Angle: {cobb_angle:.2f}°'
                    text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    x_position = output_img.shape[1] - text_size[0] - 15
                    y_position = upper_point[1] - 10
                    cv2.putText(output_img, angle_text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    angle_text = "Cobb Angle: Invalid"
                    text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    x_position = output_img.shape[1] - text_size[0] - 10
                    y_position = upper_point[1] - 10
                    cv2.putText(output_img, angle_text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                angle_text = "Not enough vertebrae detected"
                text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                x_position = output_img.shape[1] - text_size[0] - 10
                y_position = 30
                cv2.putText(output_img, angle_text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            output_pil = Image.fromarray(output_img)
            st.image(output_pil, caption="Processed Image with Bounding Boxes and Cobb Angle", use_container_width=True)

            img_io = io.BytesIO()
            st.selectbox("Choose image format for download", ["PNG", "JPEG", "JPG", "DICOM"], key="image_format")
            
            image_format = st.session_state.get('image_format', 'PNG')
            
            if image_format == "PNG":
                output_pil.save(img_io, 'PNG')
            elif image_format == "JPG" or image_format == "JPEG":
                output_pil.save(img_io, 'JPEG')
            elif image_format == "DICOM":
                dicom_io = create_dicom_from_image(output_img)
                
                if dicom_io:
                    st.download_button(
                        label="Download Processed Image (DICOM)", 
                        data=dicom_io, 
                        file_name="processed_image.dcm", 
                        mime="application/dicom"
                    )
            
            img_io.seek(0)
            st.markdown(""" 
                <style>
                    .stDownloadButton>button {
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        text-align: center;
                        position: relative;
                        border: none;
                        background: grey;
                        font-size: 16px;
                        color: white;
                    }
                    
                    .stDownloadButton>button:before {
                        content: '';
                        position: absolute;
                        bottom: -4px;  
                        left: 0;
                        width: 100%;
                        height: 2px;
                        background-color: non;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.download_button(
                label="Download Processed Image", 
                data=img_io, 
                file_name="processed_image." + image_format.lower(), 
                mime="image/" + image_format.lower()
            )
        except Exception as e:
            st.error(f"Error processing the image: {e}")
