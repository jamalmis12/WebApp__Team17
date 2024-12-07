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
        dicom_img = pydicom.dcmread(dicom_template_path, force=True)  # Force reading
        dicom_img.PixelData = np.array(output_img).tobytes()
        
        dicom_io = io.BytesIO()
        dicom_img.save(dicom_io)
        dicom_io.seek(0)
        
        return dicom_io
    except Exception as e:
        st.error(f"Error creating DICOM: {e}")
        return None

import streamlit as st
import base64
from PIL import Image

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default to the home page
if 'users' not in st.session_state:
    st.session_state.users = {}  # Store users' credentials

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
        st.session_state.page = 'login'  # Navigate to the login page

# Login page
elif st.session_state.page == 'login':
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.page = 'upload'  # Navigate to the upload page
        else:
            st.error("Invalid credentials. Please try again.")

    # Center the existing "Don't have an account? Sign Up" button using columns
    col1, col2, col3 = st.columns([1, 1, 1])  # Create 3 columns for centering

    with col2:  # Place the button in the center column
        if st.button("Don't have an account? Sign    Up", help="Redirect to Sign Up page"):
            st.session_state.page = 'signup'  # Navigate to the sign-up page

# Sign-up page
elif st.session_state.page == 'signup':
    st.subheader("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if username and password and password == confirm_password:
            if username in st.session_state.users:
                st.error("Username already exists. Please choose another one.")
            else:
                st.session_state.users[username] = password  # Save the user's credentials
                st.success("Sign-up successful! Please log in.")
                st.session_state.page = 'login'  # Navigate to the login page
        else:
            st.error("Please ensure all fields are filled correctly.")
    
# Upload page (after successful login)
elif st.session_state.page == 'upload':
    st.subheader("Upload an X-ray Image")
    file = st.file_uploader("Upload an X-ray image of the spine (JPG, PNG)", type=["jpg", "png"])

    # Handle the image prediction process (your logic here)
    if file is not None:
        # Add image processing logic
        st.image(file, caption="Uploaded Image", use_container_width=True)


    if file is not None:
        try:
            # Open the image
            img = Image.open(file)
            
            # Resize the image to the target size
            img_resized = resize_image(img, target_size=(640, 640))
            
            # Convert PIL image to numpy array (BGR for OpenCV)
            img_array = np.array(img_resized)
            img_array = img_array[..., ::-1]  # Convert RGB to BGR
            
            # Perform inference with YOLOv5
            results = model.predict(img_array)  # Make predictions on the image
            
            # Get the image with bounding boxes drawn
            output_img = results.render()[0]  # This will draw the boxes on the image
            output_img = np.array(output_img, copy=True)  # Ensure it's a writable array
            
            # Extract the coordinates of the detected vertebrae
            detections = results.xywh[0].numpy()  # Get the bounding boxes (x, y, width, height, confidence)
            vertebrae_points = []

            for detection in detections:
                x_center, y_center, width, height, confidence, class_id = detection
                
                # Check if class_id corresponds to vertebrae (class_id == 0)
                if int(class_id) == 0:
                    vertebrae_points.append((int(x_center), int(y_center)))

            # Calculate Cobb angle and draw lines if enough vertebrae are detected
            cobb_angle = None
            if len(vertebrae_points) >= 3:
                apex_point = vertebrae_points[len(vertebrae_points) // 2]
                upper_point = vertebrae_points[0]
                lower_point = vertebrae_points[-1]
                cobb_angle = calculate_cobb_angle([upper_point, lower_point])
                
                # Draw lines connecting upper and lower vertebrae
                cv2.line(output_img, upper_point, lower_point, (255, 0, 255), 2)  # Yellow line for upper-lower connection
                cv2.circle(output_img, apex_point, 10, (0, 255, 255), -1)  # Blue dot for apex vertebra
                cv2.circle(output_img, upper_point, 5, (255, 255, 0), -1)  # Green circle for upper vertebra
                cv2.circle(output_img, lower_point, 5, (0, 0, 255), -1)  # Red circle for lower vertebra

                # Annotate the Cobb angle
                if cobb_angle is not None and cobb_angle != 0:
                    angle_text = f'Cobb Angle: {cobb_angle:.2f}°'
                    text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    x_position = output_img.shape[1] - text_size[0] - 15
                    y_position = upper_point[1] - 10
                    cv2.putText(output_img, angle_text, (x_position, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    angle_text = "Cobb Angle: Invalid"
                    text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    x_position = output_img.shape[1] - text_size[0] - 10
                    y_position = upper_point[1] - 10
                    cv2.putText(output_img, angle_text, (x_position, y_position),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                angle_text = "Not enough vertebrae detected"
                text_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                x_position = output_img.shape[1] - text_size[0] - 10
                y_position = 30
                cv2.putText(output_img, angle_text, (x_position, y_position),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Convert the processed image to PIL object
            output_pil = Image.fromarray(output_img)
            
            # Display the image with bounding boxes and Cobb angle
            st.image(output_pil, caption="Processed Image with Bounding Boxes and Cobb Angle", use_container_width=True)

            # Provide download button for the processed image in different formats
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
                st.download_button(
                    label="Download Processed Image", 
                    data=img_io, 
                    file_name="processed_image.png", 
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Error processing the image: {e}")

