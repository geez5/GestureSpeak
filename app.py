import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import time
from PIL import Image, ImageTk
import threading
import os
import math

# Try to import TensorFlow, but make it optional for testing
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Using dummy classifier.")

# Try to import cvzone, but make it optional
try:
    from cvzone.HandTrackingModule import HandDetector
    CVZONE_AVAILABLE = True
    print("CVZone imported successfully")
except ImportError:
    CVZONE_AVAILABLE = False
    print("Warning: CVZone not available. Using OpenCV for hand detection.")

class SimpleHandDetector:
    """Simple hand detector using OpenCV as fallback"""
    def __init__(self, maxHands=1, detectionCon=0.7):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        
    def findHands(self, img, draw=True):
        """Simple hand detection - returns empty for now"""
        return [], img

class DummyClassifier:
    """Dummy classifier for testing when TensorFlow is not available"""
    def __init__(self):
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                      "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.is_trained = True
        self.debug_mode = True
        print("Dummy classifier initialized")
        
    def getPrediction(self, img, draw=False):
        """Return dummy prediction"""
        import random
        pred_idx = random.randint(0, len(self.labels) - 1)
        confidence = random.uniform(0.3, 0.9)
        
        # Create dummy confidence scores
        scores = [0.1] * len(self.labels)
        scores[pred_idx] = confidence
        
        return scores, pred_idx
    
    def save_model(self, filepath):
        return True
        
    def load_model(self, filepath):
        return True

class ImprovedASLClassifier:
    """Improved ASL classifier with better hand pattern recognition"""
    
    def __init__(self):
        self.model = None
        # Complete ASL alphabet + numbers
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                      "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.img_size = 224  # Better size for hand recognition
        self.is_trained = False
        self.debug_mode = True
        
        if TF_AVAILABLE:
            self.create_model()
        else:
            print("TensorFlow not available, classifier disabled")
        
    def create_model(self):
        """Create improved CNN model for ASL recognition"""
        try:
            print("Creating improved ASL model...")
            
            # Create model architecture with better feature extraction
            self.model = keras.Sequential([
                layers.Input(shape=(self.img_size, self.img_size, 3)),
                layers.Lambda(lambda x: x / 255.0),
                
                # More sophisticated data augmentation
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.15),
                layers.RandomZoom(0.15),
                layers.RandomBrightness(0.1),
                layers.RandomContrast(0.1),
                
                # Feature extraction layers
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),
                
                # Classification layers
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(len(self.labels), activation='softmax')
            ])
            
            # Compile model with better optimizer
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize with dummy prediction
            dummy_input = np.random.random((1, self.img_size, self.img_size, 3))
            self.model.predict(dummy_input, verbose=0)
            
            # Create improved synthetic training data
            self.create_realistic_training()
            
            self.is_trained = True
            print("Improved ASL model created successfully!")
            
        except Exception as e:
            print(f"Error creating model: {e}")
            self.is_trained = False
    
    def create_realistic_hand_pattern(self, img, class_idx, variation=0):
        """Create more realistic hand patterns based on actual ASL signs"""
        h, w, c = img.shape
        center_x, center_y = w // 2, h // 2 + 10
        
        # Add some random variation
        offset_x = np.random.randint(-15, 16)
        offset_y = np.random.randint(-15, 16)
        center_x += offset_x
        center_y += offset_y
        
        # Scale factor for different hand sizes
        scale = np.random.uniform(0.8, 1.2)
        
        # Hand color variations
        hand_colors = [(120, 120, 120), (100, 100, 100), (140, 140, 140), (90, 90, 90)]
        color = hand_colors[variation % len(hand_colors)]
        
        label = self.labels[class_idx]
        
        if label == "A":
            # Closed fist with thumb on side
            cv2.ellipse(img, (center_x, center_y), (int(35*scale), int(45*scale)), 0, 0, 360, color, -1)
            cv2.ellipse(img, (center_x-int(25*scale), center_y-int(5*scale)), (int(12*scale), int(20*scale)), 0, 0, 360, color, -1)
            
        elif label == "B":
            # Open palm, fingers together, thumb tucked
            cv2.rectangle(img, (center_x-int(25*scale), center_y-int(50*scale)), 
                         (center_x+int(25*scale), center_y+int(30*scale)), color, -1)
            # Fingers
            for i in range(4):
                cv2.rectangle(img, (center_x-int(20*scale)+i*int(12*scale), center_y-int(55*scale)), 
                             (center_x-int(15*scale)+i*int(12*scale), center_y-int(25*scale)), color, -1)
                             
        elif label == "C":
            # Curved hand like letter C
            cv2.ellipse(img, (center_x, center_y), (int(45*scale), int(55*scale)), 0, 45, 315, color, int(15*scale))
            
        elif label == "D":
            # Index finger up, other fingers curved, thumb touching middle finger
            cv2.ellipse(img, (center_x, center_y+int(10*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(10*scale), center_y-int(45*scale)), 
                         (center_x+int(18*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "E":
            # Curved fingers, thumb across palm
            cv2.ellipse(img, (center_x, center_y), (int(35*scale), int(40*scale)), 0, 0, 180, color, -1)
            # Curved fingers
            for i in range(4):
                cv2.ellipse(img, (center_x-int(15*scale)+i*int(10*scale), center_y-int(20*scale)), 
                           (int(6*scale), int(15*scale)), 0, 0, 180, color, -1)
                           
        elif label == "F":
            # Index and thumb circle, other fingers up
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.circle(img, (center_x-int(10*scale), center_y-int(15*scale)), int(12*scale), (255, 255, 255), int(8*scale))
            # Three fingers up
            for i in range(3):
                cv2.rectangle(img, (center_x+int(5*scale)+i*int(8*scale), center_y-int(45*scale)), 
                             (center_x+int(10*scale)+i*int(8*scale), center_y-int(10*scale)), color, -1)
                             
        elif label == "G":
            # Index finger pointing horizontally
            cv2.ellipse(img, (center_x-int(10*scale), center_y), (int(25*scale), int(30*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(15*scale), center_y-int(8*scale)), 
                         (center_x+int(50*scale), center_y+int(8*scale)), color, -1)
                         
        elif label == "H":
            # Index and middle finger horizontal
            cv2.ellipse(img, (center_x-int(10*scale), center_y), (int(25*scale), int(30*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(15*scale), center_y-int(12*scale)), 
                         (center_x+int(45*scale), center_y-int(4*scale)), color, -1)
            cv2.rectangle(img, (center_x+int(15*scale), center_y+int(4*scale)), 
                         (center_x+int(45*scale), center_y+int(12*scale)), color, -1)
                         
        elif label == "I":
            # Pinky finger up
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(20*scale), center_y-int(45*scale)), 
                         (center_x+int(26*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "J":
            # Like I but with a hook motion (show curved pinky)
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.ellipse(img, (center_x+int(25*scale), center_y-int(20*scale)), (int(8*scale), int(25*scale)), 0, 0, 180, color, int(6*scale))
            
        elif label == "K":
            # Index up, middle finger angled
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(5*scale), center_y-int(45*scale)), 
                         (center_x+int(12*scale), center_y-int(5*scale)), color, -1)
            # Angled middle finger
            pts = np.array([[center_x+int(12*scale), center_y-int(25*scale)], 
                           [center_x+int(25*scale), center_y-int(35*scale)], 
                           [center_x+int(30*scale), center_y-int(30*scale)],
                           [center_x+int(17*scale), center_y-int(20*scale)]], np.int32)
            cv2.fillPoly(img, [pts], color)
            
        elif label == "L":
            # Index up, thumb out
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(5*scale), center_y-int(45*scale)), 
                         (center_x+int(12*scale), center_y-int(5*scale)), color, -1)
            cv2.rectangle(img, (center_x-int(30*scale), center_y-int(10*scale)), 
                         (center_x-int(5*scale), center_y-int(3*scale)), color, -1)
                         
        elif label == "M":
            # Three fingers over thumb
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(35*scale), int(40*scale)), 0, 0, 360, color, -1)
            for i in range(3):
                cv2.rectangle(img, (center_x-int(10*scale)+i*int(8*scale), center_y-int(35*scale)), 
                             (center_x-int(5*scale)+i*int(8*scale), center_y+int(5*scale)), color, -1)
                             
        elif label == "N":
            # Two fingers over thumb
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(35*scale), int(40*scale)), 0, 0, 360, color, -1)
            for i in range(2):
                cv2.rectangle(img, (center_x-int(5*scale)+i*int(8*scale), center_y-int(35*scale)), 
                             (center_x+i*int(8*scale), center_y+int(5*scale)), color, -1)
                             
        elif label == "O":
            # Circle with fingers
            cv2.circle(img, (center_x, center_y), int(35*scale), color, int(12*scale))
            
        elif label == "P":
            # Like K but pointing down
            cv2.ellipse(img, (center_x, center_y-int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(5*scale), center_y+int(5*scale)), 
                         (center_x+int(12*scale), center_y+int(45*scale)), color, -1)
            # Angled middle finger
            pts = np.array([[center_x+int(12*scale), center_y+int(25*scale)], 
                           [center_x+int(25*scale), center_y+int(35*scale)], 
                           [center_x+int(30*scale), center_y+int(30*scale)],
                           [center_x+int(17*scale), center_y+int(20*scale)]], np.int32)
            cv2.fillPoly(img, [pts], color)
            
        elif label == "Q":
            # Like G but pointing down
            cv2.ellipse(img, (center_x-int(10*scale), center_y), (int(25*scale), int(30*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(10*scale), center_y+int(15*scale)), 
                         (center_x+int(18*scale), center_y+int(45*scale)), color, -1)
                         
        elif label == "R":
            # Index and middle crossed
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            # Crossed fingers
            cv2.rectangle(img, (center_x+int(5*scale), center_y-int(45*scale)), 
                         (center_x+int(12*scale), center_y-int(5*scale)), color, -1)
            cv2.rectangle(img, (center_x+int(8*scale), center_y-int(35*scale)), 
                         (center_x+int(20*scale), center_y-int(28*scale)), color, -1)
                         
        elif label == "S":
            # Closed fist with thumb over fingers
            cv2.ellipse(img, (center_x, center_y), (int(35*scale), int(40*scale)), 0, 0, 360, color, -1)
            cv2.ellipse(img, (center_x-int(15*scale), center_y-int(10*scale)), (int(15*scale), int(8*scale)), 0, 0, 360, color, -1)
            
        elif label == "T":
            # Thumb between index and middle
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.circle(img, (center_x, center_y-int(20*scale)), int(10*scale), color, -1)
            
        elif label == "U":
            # Index and middle up together
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(2*scale), center_y-int(45*scale)), 
                         (center_x+int(18*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "V":
            # Peace sign
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(0*scale), center_y-int(45*scale)), 
                         (center_x+int(7*scale), center_y-int(5*scale)), color, -1)
            cv2.rectangle(img, (center_x+int(13*scale), center_y-int(45*scale)), 
                         (center_x+int(20*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "W":
            # Three fingers up
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            for i in range(3):
                cv2.rectangle(img, (center_x-int(2*scale)+i*int(8*scale), center_y-int(45*scale)), 
                             (center_x+int(5*scale)+i*int(8*scale), center_y-int(5*scale)), color, -1)
                             
        elif label == "X":
            # Index finger hooked
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.ellipse(img, (center_x+int(15*scale), center_y-int(20*scale)), (int(6*scale), int(20*scale)), 0, 0, 180, color, int(5*scale))
            
        elif label == "Y":
            # Thumb and pinky out
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x-int(30*scale), center_y-int(10*scale)), 
                         (center_x-int(5*scale), center_y-int(3*scale)), color, -1)
            cv2.rectangle(img, (center_x+int(20*scale), center_y-int(30*scale)), 
                         (center_x+int(26*scale), center_y+int(5*scale)), color, -1)
                         
        elif label == "Z":
            # Index finger drawing Z (show as diagonal line)
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.line(img, (center_x+int(10*scale), center_y-int(30*scale)), 
                    (center_x+int(25*scale), center_y-int(10*scale)), color, int(6*scale))
                    
        # Numbers
        elif label == "1":
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(10*scale), center_y-int(45*scale)), 
                         (center_x+int(17*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "2":
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            cv2.rectangle(img, (center_x+int(5*scale), center_y-int(45*scale)), 
                         (center_x+int(12*scale), center_y-int(5*scale)), color, -1)
            cv2.rectangle(img, (center_x+int(15*scale), center_y-int(45*scale)), 
                         (center_x+int(22*scale), center_y-int(5*scale)), color, -1)
                         
        elif label == "3":
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            for i in range(3):
                cv2.rectangle(img, (center_x+int(2*scale)+i*int(8*scale), center_y-int(45*scale)), 
                             (center_x+int(9*scale)+i*int(8*scale), center_y-int(5*scale)), color, -1)
                             
        elif label == "4":
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            for i in range(4):
                cv2.rectangle(img, (center_x-int(2*scale)+i*int(7*scale), center_y-int(45*scale)), 
                             (center_x+int(4*scale)+i*int(7*scale), center_y-int(5*scale)), color, -1)
                             
        elif label == "5":
            # Open hand
            cv2.ellipse(img, (center_x, center_y+int(5*scale)), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            for i in range(4):
                cv2.rectangle(img, (center_x-int(5*scale)+i*int(8*scale), center_y-int(45*scale)), 
                             (center_x+int(2*scale)+i*int(8*scale), center_y-int(5*scale)), color, -1)
            cv2.rectangle(img, (center_x-int(30*scale), center_y-int(10*scale)), 
                         (center_x-int(5*scale), center_y-int(3*scale)), color, -1)
                         
        else:  # Numbers 6-9
            # Create variations for other numbers
            cv2.ellipse(img, (center_x, center_y), (int(30*scale), int(35*scale)), 0, 0, 360, color, -1)
            if label in ["6", "7", "8", "9"]:
                num_fingers = int(label) - 5
                for i in range(num_fingers):
                    cv2.rectangle(img, (center_x-int(10*scale)+i*int(8*scale), center_y-int(40*scale)), 
                                 (center_x-int(5*scale)+i*int(8*scale), center_y-int(10*scale)), color, -1)
        
        return img
    
    def create_realistic_training(self):
        """Create more realistic training data"""
        try:
            print("Creating realistic training data...")
            
            num_samples = 100  # More samples per class
            X_train = []
            y_train = []
            
            for class_idx in range(len(self.labels)):
                print(f"Creating samples for {self.labels[class_idx]}...")
                for variation in range(num_samples):
                    # Create background with some variation
                    bg_color = np.random.randint(200, 256)
                    synthetic_img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * bg_color
                    
                    # Add realistic hand pattern
                    synthetic_img = self.create_realistic_hand_pattern(synthetic_img, class_idx, variation)
                    
                    # Add realistic augmentations
                    if np.random.random() > 0.3:
                        # Add gaussian noise
                        noise = np.random.normal(0, 8, synthetic_img.shape).astype(np.int16)
                        synthetic_img = np.clip(synthetic_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    if np.random.random() > 0.5:
                        # Add slight blur
                        synthetic_img = cv2.GaussianBlur(synthetic_img, (3, 3), 0.5)
                    
                    if np.random.random() > 0.7:
                        # Add brightness variation
                        brightness = np.random.randint(-20, 21)
                        synthetic_img = np.clip(synthetic_img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
                    
                    X_train.append(synthetic_img)
                    y_train.append(class_idx)
            
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train)
            
            print(f"Training on {len(X_train)} samples...")
            
            # Train with more epochs for better learning
            history = self.model.fit(
                X_train, y_train, 
                epochs=15,  # More epochs
                batch_size=32, 
                validation_split=0.2,
                verbose=1
            )
            
            print("Training completed!")
            print(f"Final accuracy: {history.history['accuracy'][-1]:.3f}")
            if 'val_accuracy' in history.history:
                print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
            
        except Exception as e:
            print(f"Error in training: {e}")
    
    def preprocess_image(self, img):
        """Preprocess image for prediction"""
        if img is None or img.size == 0:
            return None
        
        # Resize and normalize
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to float and normalize
        img_normalized = img_resized.astype(np.float32)
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def getPrediction(self, img, draw=False):
        """Get prediction from image with confidence filtering"""
        if not self.is_trained or self.model is None:
            return [0.0] * len(self.labels), 0
            
        processed_img = self.preprocess_image(img)
        if processed_img is None:
            return [0.0] * len(self.labels), 0
        
        try:
            predictions = self.model.predict(processed_img, verbose=0)
            confidence_scores = predictions[0]
            predicted_class = np.argmax(confidence_scores)
            
            # Apply confidence threshold and smoothing
            max_confidence = np.max(confidence_scores)
            if max_confidence < 0.3:  # Very low confidence
                return [0.0] * len(self.labels), 0
            
            return confidence_scores.tolist(), predicted_class
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return [0.0] * len(self.labels), 0
    
    def save_model(self, filepath):
        """Save model"""
        if self.model is not None:
            try:
                self.model.save(filepath)
                return True
            except Exception as e:
                print(f"Error saving: {e}")
                return False
        return False
    
    def load_model(self, filepath):
        """Load model"""
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading: {e}")
            return False

class ASLRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤟 Improved ASL Recognition")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_camera_running = False
        self.detector = None
        self.classifier = None
        self.sentence = []
        self.predictions = []
        self.offset = 20
        self.imgSize = 224  # Match classifier size
        self.current_prediction = None
        self.current_confidence = 0.0
        self.prediction_buffer = []  # For smoothing predictions
        self.last_prediction_time = 0
        
        # Full ASL alphabet + numbers
        self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                      "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                      "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        print("Setting up improved GUI...")
        self.setup_gui()
        
        print("Initializing components...")
        self.initialize_components()
        
        print("GUI ready!")
        
    def setup_gui(self):
        """Setup improved GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="🤟 Improved ASL Recognition", font=("Arial", 18, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Left panel - Controls
        controls = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        camera_frame = ttk.LabelFrame(controls, text="Camera", padding="5")
        camera_frame.pack(fill="x", pady=(0, 10))
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=2)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        
        # Status
        status_frame = ttk.LabelFrame(controls, text="Status", padding="5")
        status_frame.pack(fill="x", pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(anchor="w")
        
        self.model_status = ttk.Label(status_frame, text="Model: Loading...")
        self.model_status.pack(anchor="w")
        
        # Training progress
        self.training_progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.training_progress.pack(fill="x", pady=2)
        
        # Action buttons
        action_frame = ttk.LabelFrame(controls, text="Actions", padding="5")
        action_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(action_frame, text="Test Classification", command=self.test_classification).pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Add Space", command=self.add_space).pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Delete Last", command=self.delete_last).pack(fill="x", pady=2)
        ttk.Button(action_frame, text="Clear All", command=self.clear_sentence).pack(fill="x", pady=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(controls, text="Settings", padding="5")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor="w")
        self.confidence_threshold = tk.DoubleVar(value=0.7)  # Higher threshold
        confidence_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.confidence_threshold, 
                                   orient="horizontal")
        confidence_scale.pack(fill="x")
        
        self.confidence_value_label = ttk.Label(settings_frame, text="0.70")
        self.confidence_value_label.pack()
        
        # Update confidence display
        def update_confidence_display(*args):
            self.confidence_value_label.config(text=f"{self.confidence_threshold.get():.2f}")
        self.confidence_threshold.trace('w', update_confidence_display)
        
        ttk.Label(settings_frame, text="Prediction Smoothing:").pack(anchor="w", pady=(10,0))
        self.smoothing_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Enable smoothing", 
                       variable=self.smoothing_enabled).pack(anchor="w")
        
        # Model controls
        model_frame = ttk.LabelFrame(controls, text="Model", padding="5")
        model_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(model_frame, text="Save Model", command=self.save_model).pack(fill="x", pady=2)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill="x", pady=2)
        
        # Supported signs - scrollable
        signs_frame = ttk.LabelFrame(controls, text="Supported Signs", padding="5")
        signs_frame.pack(fill="both", expand=True)
        
        # Create scrollable text widget for signs
        signs_text_frame = ttk.Frame(signs_frame)
        signs_text_frame.pack(fill="both", expand=True)
        
        signs_scrollbar = ttk.Scrollbar(signs_text_frame)
        signs_scrollbar.pack(side="right", fill="y")
        
        self.signs_text = tk.Text(signs_text_frame, height=8, wrap=tk.WORD, 
                                 yscrollcommand=signs_scrollbar.set)
        self.signs_text.pack(side="left", fill="both", expand=True)
        signs_scrollbar.config(command=self.signs_text.yview)
        
        # Populate signs
        alphabet_text = "Letters: " + ", ".join(self.labels[:26]) + "\n\n"
        numbers_text = "Numbers: " + ", ".join(self.labels[26:])
        self.signs_text.insert(tk.END, alphabet_text + numbers_text)
        self.signs_text.config(state="disabled")
        
        # Right panel - Video and results
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=2)
        right_frame.rowconfigure(1, weight=1)
        
        # Video frame
        video_frame = ttk.LabelFrame(right_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Camera not started\n\nPosition your hand in front of the camera\nEnsure good lighting and clear background", 
                                   anchor="center", font=("Arial", 12))
        self.video_label.pack(expand=True, fill="both")
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Recognition Results", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        
        # Current prediction with larger display
        pred_frame = ttk.Frame(results_frame)
        pred_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        pred_frame.columnconfigure(1, weight=1)
        
        ttk.Label(pred_frame, text="Current Sign:", font=("Arial", 12)).grid(row=0, column=0, sticky="w")
        self.prediction_label = ttk.Label(pred_frame, text="None", font=("Arial", 16, "bold"), 
                                         foreground="blue")
        self.prediction_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(pred_frame, text="Confidence:", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
        self.confidence_label = ttk.Label(pred_frame, text="0.000", font=("Arial", 12))
        self.confidence_label.grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        # Sentence display
        sentence_frame = ttk.Frame(results_frame)
        sentence_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        sentence_frame.columnconfigure(0, weight=1)
        sentence_frame.rowconfigure(1, weight=1)
        
        ttk.Label(sentence_frame, text="Sentence:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")
        
        sentence_text_frame = ttk.Frame(sentence_frame)
        sentence_text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        sentence_text_frame.columnconfigure(0, weight=1)
        sentence_text_frame.rowconfigure(0, weight=1)
        
        sentence_scrollbar = ttk.Scrollbar(sentence_text_frame)
        sentence_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.sentence_text = tk.Text(sentence_text_frame, height=4, wrap=tk.WORD, 
                                   font=("Arial", 14), yscrollcommand=sentence_scrollbar.set)
        self.sentence_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        sentence_scrollbar.config(command=self.sentence_text.yview)
        
        # Recent predictions
        pred_history_frame = ttk.Frame(results_frame)
        pred_history_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pred_history_frame.columnconfigure(0, weight=1)
        pred_history_frame.rowconfigure(1, weight=1)
        
        ttk.Label(pred_history_frame, text="Recent Predictions:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w")
        
        pred_list_frame = ttk.Frame(pred_history_frame)
        pred_list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pred_list_frame.columnconfigure(0, weight=1)
        pred_list_frame.rowconfigure(0, weight=1)
        
        pred_scrollbar = ttk.Scrollbar(pred_list_frame)
        pred_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.predictions_list = tk.Listbox(pred_list_frame, height=6, font=("Arial", 10),
                                         yscrollcommand=pred_scrollbar.set)
        self.predictions_list.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pred_scrollbar.config(command=self.predictions_list.yview)
        
    def initialize_components(self):
        """Initialize detector and classifier with progress indication"""
        try:
            self.training_progress.start()
            
            # Initialize hand detector
            if CVZONE_AVAILABLE:
                self.detector = HandDetector(maxHands=1, detectionCon=0.8)  # Higher confidence
                print("CVZone hand detector initialized")
            else:
                self.detector = SimpleHandDetector()
                print("Using simple hand detector (CVZone not available)")
            
            # Initialize classifier
            if TF_AVAILABLE:
                self.classifier = ImprovedASLClassifier()  # Use improved classifier
                if self.classifier.is_trained:
                    self.model_status.config(text="Model: Trained & Ready", foreground="green")
                else:
                    self.model_status.config(text="Model: Training Failed", foreground="red")
            else:
                self.classifier = DummyClassifier()
                self.model_status.config(text="Model: Dummy (TF not available)", foreground="orange")
            
            self.training_progress.stop()
            self.status_label.config(text="Components initialized successfully", foreground="green")
            
        except Exception as e:
            self.training_progress.stop()
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            print(f"Error initializing components: {e}")
            
            # Fallback to dummy components
            self.detector = SimpleHandDetector()
            self.classifier = DummyClassifier()
    
    def test_classification(self):
        """Test the classifier with better feedback"""
        if not self.classifier:
            messagebox.showerror("Error", "Classifier not ready!")
            return
        
        # Create test image that resembles letter 'A'
        test_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        # Create a simple 'A' pattern
        cv2.ellipse(test_img, (112, 130), (35, 45), 0, 0, 360, (120, 120, 120), -1)
        cv2.ellipse(test_img, (87, 125), (12, 20), 0, 0, 360, (120, 120, 120), -1)
        
        prediction, index = self.classifier.getPrediction(test_img)
        
        if index < len(self.labels) and prediction:
            confidence = max(prediction) if prediction else 0.0
            result = f"Test Result:\n\nPredicted: {self.labels[index]}\nConfidence: {confidence:.3f}\n\nTop 3 predictions:"
            
            # Show top 3 predictions
            if len(prediction) >= 3:
                sorted_indices = np.argsort(prediction)[::-1][:3]
                for i, idx in enumerate(sorted_indices):
                    result += f"\n{i+1}. {self.labels[idx]}: {prediction[idx]:.3f}"
            
            messagebox.showinfo("Test Classification", result)
        else:
            messagebox.showerror("Error", "Invalid prediction result")
    
    def add_space(self):
        """Add space to sentence"""
        if self.sentence and self.sentence[-1] != " ":
            self.sentence.append(" ")
            self.update_sentence_display()
    
    def delete_last(self):
        """Delete last character/word from sentence"""
        if self.sentence:
            self.sentence.pop()
            self.update_sentence_display()
    
    def clear_sentence(self):
        """Clear sentence and predictions"""
        self.sentence = []
        self.predictions = []
        self.prediction_buffer = []
        self.update_sentence_display()
        self.update_predictions_display()
    
    def save_model(self):
        """Save the trained model"""
        if not self.classifier or not hasattr(self.classifier, 'save_model'):
            messagebox.showerror("Error", "No model to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if filename:
            if self.classifier.save_model(filename):
                messagebox.showinfo("Success", f"Model saved to {filename}")
            else:
                messagebox.showerror("Error", "Failed to save model")
    
    def load_model(self):
        """Load a pre-trained model"""
        if not self.classifier or not hasattr(self.classifier, 'load_model'):
            messagebox.showerror("Error", "No classifier available!")
            return
        
        filename = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if filename:
            if self.classifier.load_model(filename):
                messagebox.showinfo("Success", f"Model loaded from {filename}")
                self.model_status.config(text="Model: Loaded & Ready", foreground="green")
            else:
                messagebox.showerror("Error", "Failed to load model")
    
    def smooth_prediction(self, prediction, confidence):
        """Smooth predictions over time to reduce noise"""
        if not self.smoothing_enabled.get():
            return prediction, confidence
        
        current_time = time.time()
        
        # Add to buffer
        self.prediction_buffer.append({
            'prediction': prediction,
            'confidence': confidence,
            'time': current_time
        })
        
        # Keep only recent predictions (last 1 second)
        self.prediction_buffer = [p for p in self.prediction_buffer 
                                if current_time - p['time'] < 1.0]
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Find most common prediction with high confidence
        pred_counts = {}
        for p in self.prediction_buffer:
            if p['confidence'] > self.confidence_threshold.get():
                pred = p['prediction']
                if pred not in pred_counts:
                    pred_counts[pred] = []
                pred_counts[pred].append(p['confidence'])
        
        if not pred_counts:
            return prediction, confidence
        
        # Get prediction with highest average confidence
        best_pred = max(pred_counts.keys(), 
                       key=lambda x: np.mean(pred_counts[x]))
        best_confidence = np.mean(pred_counts[best_pred])
        
        return best_pred, best_confidence
    
    def update_sentence_display(self):
        """Update sentence text widget"""
        sentence_text = "".join(self.sentence)
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(1.0, sentence_text)
        self.sentence_text.see(tk.END)  # Scroll to end
    
    def update_predictions_display(self):
        """Update predictions list"""
        self.predictions_list.delete(0, tk.END)
        for pred in self.predictions[-15:]:  # Show last 15
            text = f"{pred['time']} - {pred['sign']} ({pred['confidence']:.3f})"
            self.predictions_list.insert(tk.END, text)
        if self.predictions:
            self.predictions_list.see(tk.END)  # Scroll to bottom
    
    def start_camera(self):
        """Start camera with better error handling"""
        try:
            # Try different camera indices
            for camera_id in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_id)
                if self.cap.isOpened():
                    break
                self.cap.release()
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open any camera!\n\nPlease check:\n1. Camera is connected\n2. Camera is not used by other apps\n3. Camera permissions are granted")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_camera_running = True
            self.start_btn.config(state="disabled")  
            self.stop_btn.config(state="normal")
            self.status_label.config(text="Camera running - Show hand signs clearly", foreground="green")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera initialization error: {str(e)}")
    
    def stop_camera(self):
        """Stop camera"""
        self.is_camera_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Camera stopped", foreground="orange")
        
        if self.cap:
            self.cap.release()
        
        self.video_label.config(image="", text="Camera stopped")
        self.prediction_label.config(text="None")
        self.confidence_label.config(text="0.000")
    
    def camera_loop(self):
        """Improved camera loop with better processing"""
        frame_count = 0
        
        while self.is_camera_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)  # Mirror for user convenience
                
                # Process every other frame for performance
                if frame_count % 2 == 0:
                    processed_frame, prediction, confidence = self.process_frame(frame)
                else:
                    processed_frame = frame
                    prediction = self.current_prediction
                    confidence = self.current_confidence
                
                # Convert to display format
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.resize((600, 450), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update GUI in main thread
                self.root.after(0, self.update_display, photo, prediction, confidence)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Camera loop error: {e}")
                break
    
    def process_frame(self, frame):
        """Process frame with improved hand detection and recognition"""
        output_frame = frame.copy()
        prediction = None
        confidence = 0.0
        
        try:
            # Add instruction text
            cv2.putText(output_frame, "Show ASL sign clearly", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(output_frame, "Keep hand in center", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect hands
            hands, img_with_hands = self.detector.findHands(frame, draw=False)
            
            if hands and len(hands) > 0:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Expand bounding box for better capture
                padding = 40
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                # Draw bounding box
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 255), 3)
                cv2.putText(output_frame, "Hand Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Extract and process hand region
                hand_region = frame[y:y+h, x:x+w]
                
                if hand_region.size > 0 and hand_region.shape[0] > 50 and hand_region.shape[1] > 50:
                    # Create clean background
                    img_white = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                    
                    # Resize hand region maintaining aspect ratio
                    aspect_ratio = hand_region.shape[1] / hand_region.shape[0]
                    if aspect_ratio > 1:
                        new_w = self.imgSize
                        new_h = int(self.imgSize / aspect_ratio)
                    else:
                        new_h = self.imgSize
                        new_w = int(self.imgSize * aspect_ratio)
                    
                    hand_resized = cv2.resize(hand_region, (new_w, new_h))
                    
                    # Center the hand in the white background
                    y_offset = (self.imgSize - new_h) // 2
                    x_offset = (self.imgSize - new_w) // 2
                    img_white[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = hand_resized
                    
                    # Get prediction
                    pred_result, pred_index = self.classifier.getPrediction(img_white)
                    
                    if pred_index < len(self.labels) and pred_result:
                        raw_confidence = max(pred_result) if pred_result else 0.0
                        raw_prediction = self.labels[pred_index]
                        
                        # Apply smoothing
                        prediction, confidence = self.smooth_prediction(raw_prediction, raw_confidence)
                        
                        # Display prediction info on frame
                        if confidence > self.confidence_threshold.get():
                            color = (0, 255, 0)  # Green for high confidence
                            cv2.putText(output_frame, f"{prediction}", 
                                       (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                            cv2.putText(output_frame, f"Confidence: {confidence:.2f}", 
                                       (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            color = (0, 165, 255)  # Orange for low confidence
                            cv2.putText(output_frame, f"{prediction} (Low Conf)", 
                                       (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
                        # Add to sentence if confidence is high enough
                        current_time = time.time()
                        if (confidence > self.confidence_threshold.get() and 
                            (len(self.sentence) == 0 or 
                             prediction != (self.sentence[-1] if self.sentence[-1] != " " else 
                                          (self.sentence[-2] if len(self.sentence) > 1 else "")) or 
                             current_time - self.last_prediction_time > 3.0)):
                            
                            self.sentence.append(prediction)
                            self.predictions.append({
                                'time': time.strftime("%H:%M:%S"),
                                'sign': prediction,
                                'confidence': confidence
                            })
                            self.last_prediction_time = current_time
                            
                        # Store current prediction
                        self.current_prediction = prediction
                        self.current_confidence = confidence
            else:
                cv2.putText(output_frame, "No hand detected", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            cv2.putText(output_frame, "Processing Error", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_frame, prediction, confidence
    
    def update_display(self, photo, prediction, confidence):
        """Update display elements"""
        # Update video
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
        
        # Update prediction display
        if prediction:
            self.prediction_label.config(text=prediction)
            if confidence > self.confidence_threshold.get():
                self.prediction_label.config(foreground="green")
            else:
                self.prediction_label.config(foreground="orange")
            self.confidence_label.config(text=f"{confidence:.3f}")
        else:
            self.prediction_label.config(text="None", foreground="gray")
            self.confidence_label.config(text="0.000")
        
        # Update sentence and predictions
        self.update_sentence_display()
        self.update_predictions_display()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_camera_running:
            self.stop_camera()
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    """Main function with improved startup"""
    print("Improved ASL Recognition Application")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    print(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Not Available'}")
    print(f"CVZone: {'Available' if CVZONE_AVAILABLE else 'Not Available'}")
    print(f"OpenCV: Available")
    print(f"PIL: Available")
    print(f"NumPy: Available")
    
    if TF_AVAILABLE:
        # Configure TensorFlow for better performance
        try:
            # Limit GPU memory growth if GPU is available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth configured")
            else:
                # Use CPU only for more stable performance
                tf.config.set_visible_devices([], 'GPU')
                print("Using CPU for inference")
                
            # Set mixed precision for better performance
            tf.config.optimizer.set_jit(True)
            print("TensorFlow optimizations enabled")
        except Exception as e:
            print(f"TensorFlow configuration warning: {e}")
    
    print("\nFeatures in this improved version:")
    print("- Complete ASL alphabet (A-Z) + numbers (1-9)")
    print("- More realistic hand pattern training")
    print("- Improved confidence thresholding")
    print("- Prediction smoothing to reduce noise")
    print("- Better hand detection and preprocessing")
    print("- Enhanced GUI with more controls")
    print("- Model save/load functionality")
    print("- Better error handling and user feedback")
    
    print("\nStarting GUI...")
    
    # Create and run GUI
    root = tk.Tk()
    app = ASLRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("GUI started successfully!")
    print("\nUsage Tips:")
    print("1. Ensure good lighting and clear background")
    print("2. Keep your hand centered in the camera view")
    print("3. Hold signs steady for 2-3 seconds")
    print("4. Adjust confidence threshold if needed")
    print("5. Use smoothing for more stable predictions")
    
    root.mainloop()

if __name__ == "__main__":
    main()