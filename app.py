import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from tensorflow.keras.models import load_model
import time
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Cache the model loading to avoid reloading on each run
@st.cache_resource
def load_lstm_model():
    """Load the trained LSTM model"""
    try:
        model = load_model('LRCN_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define your actions/signs (update this list based on your trained model)
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

def mediapipe_detection(image, model):
    """Process image with MediaPipe"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def draw_landmarks(image, results):
    """Draw MediaPipe landmarks on the image"""
    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image

def process_video(video_path, model, threshold=0.8, show_annotations=False):
    """Process uploaded video and return predictions"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    sequence = []
    sentence = []
    predictions = []
    annotated_frames = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            progress = frame_idx / frame_count
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{frame_count}")
            
            # MediaPipe detection
            image, results = mediapipe_detection(frame, holistic)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep only last 30 frames
            
            # Make prediction if we have enough frames
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                confidence = res[np.argmax(res)]
                
                if confidence > threshold:
                    predicted_word = actions[np.argmax(res)]
                    
                    # Add to sentence if it's different from the last prediction
                    if len(sentence) == 0 or predicted_word != sentence[-1]:
                        sentence.append(predicted_word)
                        predictions.append({
                            'frame': frame_idx,
                            'time': frame_idx / fps,
                            'sign': predicted_word,
                            'confidence': confidence
                        })
            
            # Draw landmarks if annotations are enabled
            if show_annotations:
                annotated_frame = draw_landmarks(image.copy(), results)
                annotated_frames.append(annotated_frame)
            
            frame_idx += 1
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return sentence, predictions, annotated_frames, {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }

def main():
    st.title("🤟 Sign Language Recognition")
    st.write("Upload a video file to recognize sign language gestures")
    
    # Load model
    model = load_lstm_model()
    if model is None:
        st.error("Could not load the LSTM model. Please ensure 'LRCN_model.h5' is in the current directory.")
        return
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.8, 0.1)
    show_annotations = st.sidebar.checkbox("Show Landmark Annotations", value=False)
    show_predictions_table = st.sidebar.checkbox("Show Detailed Predictions", value=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file containing sign language gestures"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Display video info
            st.subheader("Uploaded Video")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.video(uploaded_file)
            
            with col2:
                st.write("**File Details:**")
                st.write(f"Name: {uploaded_file.name}")
                st.write(f"Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            # Process video button
            if st.button("🚀 Process Video", type="primary"):
                st.subheader("Processing Video...")
                
                # Process the video
                sentence, predictions, annotated_frames, video_info = process_video(
                    tmp_file_path, model, threshold, show_annotations
                )
                
                # Display results
                st.subheader("📝 Recognition Results")
                
                if sentence:
                    # Display recognized sentence
                    st.success("**Recognized Signs:**")
                    st.write(" ".join(sentence))
                    
                    # Display video information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{video_info['duration']:.2f}s")
                    with col2:
                        st.metric("Frame Rate", f"{video_info['fps']:.1f} FPS")
                    with col3:
                        st.metric("Signs Detected", len(predictions))
                    
                    # Show detailed predictions table
                    if show_predictions_table and predictions:
                        st.subheader("📊 Detailed Predictions")
                        predictions_df = []
                        for pred in predictions:
                            predictions_df.append({
                                'Time (s)': f"{pred['time']:.2f}",
                                'Frame': pred['frame'],
                                'Sign': pred['sign'],
                                'Confidence': f"{pred['confidence']:.3f}"
                            })
                        
                        st.dataframe(predictions_df, use_container_width=True)
                    
                    # Show annotated frames if enabled
                    if show_annotations and annotated_frames:
                        st.subheader("🎯 Annotated Frames")
                        st.write("Showing sample frames with MediaPipe landmarks:")
                        
                        # Display every 30th frame to avoid too many images
                        sample_frames = annotated_frames[::30]
                        
                        cols = st.columns(min(3, len(sample_frames)))
                        for i, frame in enumerate(sample_frames[:3]):
                            with cols[i]:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.image(frame_rgb, caption=f"Frame {i*30}", use_column_width=True)
                
                else:
                    st.warning("No signs detected in the video. Try adjusting the threshold or ensure the video contains clear sign language gestures.")
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    else:
        st.info("👆 Please upload a video file to get started")
        
        # Show example/instructions
        st.subheader("ℹ️ Instructions")
        st.write("""
        1. **Upload a video** containing sign language gestures
        2. **Adjust settings** in the sidebar if needed
        3. **Click 'Process Video'** to analyze the gestures
        4. **View results** including recognized signs and confidence scores
        
        **Tips:**
        - Ensure good lighting and clear hand visibility
        - The model recognizes individual letters (A-Z)
        - Adjust the threshold to filter out low-confidence predictions
        - Enable annotations to see MediaPipe landmark detection
        """)

if __name__ == "__main__":
    main()