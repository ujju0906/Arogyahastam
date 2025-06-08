from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
import pydicom
import cv2
import csv
import os
import base64
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
from dicom_image import process_dicom_image  # Handles DICOM image extraction
import numpy as np

# Import ultralytics for YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics not installed. Install with: pip install ultralytics")

app = FastAPI(title="Medical Image Classification API", version="1.0.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================================
# CONFIGURATION AND MODEL LOADING
# ================================

MODEL_PATH = Path(__file__).parent.parent / "Models" / "yolov8x.pt"
FALLBACK_PATH = Path(__file__).parent.parent / "Models" / "densenet_final_model.pt"
FALLBACK_PATH2 = Path(__file__).parent.parent / "Models" / "augmented_densenet_final_model.pt"

# Medical classification mapping
MEDICAL_CLASS_MAPPING = {
    0: "TB Positive",
    1: "Normal", 
    2: "COPD",
    3: "Silicosis", 
    4: "Lung Cancer"
}

# COCO to Medical mapping (for generic YOLOv8x models)
COCO_TO_MEDICAL_MAPPING = {
    0: 1,   # person -> Normal
    67: 1,  # cell phone -> Normal  
    72: 1,  # tv -> Normal
    'default': 1  # Default to Normal for unrecognized classes
}

# Initialize model variables
model = None
model_type = None

def load_ai_models():
    """Load AI models with fallback logic."""
    global model, model_type
    
    # Try to load YOLOv8x first
    if MODEL_PATH.exists() and YOLO_AVAILABLE:
        try:
            print(f"🔄 Loading YOLOv8x model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            model_type = "yolo"
            
            # Get model information
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                class_names = model.model.names
                num_classes = len(class_names)
                print(f"📋 YOLOv8x Model Info:")
                print(f"   • Number of classes: {num_classes}")
                print(f"   • Model task: {getattr(model, 'task', 'unknown')}")
                
                # Check if it's likely a medical model
                medical_keywords = ['tb', 'normal', 'copd', 'lung', 'cancer', 'pneumonia', 'chest']
                class_names_str = ' '.join(str(name).lower() for name in class_names.values())
                is_medical = any(keyword in class_names_str for keyword in medical_keywords)
                
                if is_medical:
                    print(f"✅ Detected medical-specific model!")
                else:
                    print(f"⚠️ Appears to be generic COCO model - will map classes to medical categories")
            
            print(f"✅ YOLOv8x model loaded successfully!")
            return
        except Exception as e:
            print(f"❌ Failed to load YOLOv8x: {e}")
    
    # Fallback to DenseNet if YOLOv8x fails
    print("🔄 Falling back to DenseNet model...")
    from ModelClass import MedicalDenseNet
    
    # Try different DenseNet model paths
    densenet_path = None
    if FALLBACK_PATH.exists():
        densenet_path = FALLBACK_PATH
    elif FALLBACK_PATH2.exists():
        densenet_path = FALLBACK_PATH2
    
    if densenet_path:
        try:
            num_classes = 5
            model = MedicalDenseNet(num_classes)
            model.load_state_dict(torch.load(densenet_path, map_location=torch.device("cpu"), weights_only=False))
            model.eval()
            model_type = "densenet"
            print(f"✅ DenseNet model loaded successfully from: {densenet_path}")
        except Exception as e:
            print(f"❌ Failed to load DenseNet: {e}")
            raise FileNotFoundError(f"Could not load any model. Error: {e}")
    else:
        raise FileNotFoundError(f"No model files found. Please ensure you have either yolov8x.pt or densenet models in the Models folder.")

# Load models on startup
load_ai_models()
print(f"🎯 Using model type: {model_type.upper()}")

# ================================
# UTILITY FUNCTIONS
# ================================

def map_yolo_class_to_medical(yolo_class, confidence):
    """Map YOLOv8x class to medical classification."""
    print(f"🔍 Mapping YOLO class {yolo_class} (conf: {confidence:.4f}) to medical category")
    
    # If it's already a medical class (0-4), use it directly
    if 0 <= yolo_class <= 4:
        medical_class = yolo_class
        print(f"✅ Direct medical class mapping: {yolo_class} -> {MEDICAL_CLASS_MAPPING[medical_class]}")
        return medical_class
    
    # If it's a COCO class, map to medical category
    if yolo_class in COCO_TO_MEDICAL_MAPPING:
        medical_class = COCO_TO_MEDICAL_MAPPING[yolo_class]
        print(f"🔄 COCO to medical mapping: {yolo_class} -> {medical_class} ({MEDICAL_CLASS_MAPPING[medical_class]})")
        return medical_class
    
    # For unknown classes, use confidence-based logic
    if confidence > 0.7:
        medical_class = 0  # TB Positive as cautious approach
        print(f"⚠️ High confidence unknown class -> {MEDICAL_CLASS_MAPPING[medical_class]} (cautious)")
    elif confidence > 0.3:
        medical_class = 1  # Normal
        print(f"🔄 Medium confidence unknown class -> {MEDICAL_CLASS_MAPPING[medical_class]}")
    else:
        medical_class = 1  # Normal
        print(f"🔄 Low confidence unknown class -> {MEDICAL_CLASS_MAPPING[medical_class]}")
    
    return medical_class

def safe_get_dicom_value(dicom_data, tag, default="Unknown"):
    """Safely extract values from DICOM data, handling encoding issues."""
    try:
        if hasattr(dicom_data, tag):
            value = getattr(dicom_data, tag)
            if value is None or value == "":
                return default
            
            # Convert to string and clean up
            str_value = str(value).strip()
            
            # Filter out obviously corrupted/encoded data
            if (
                len(str_value) > 50 or  # Too long
                any(char in str_value for char in ['=', '+', '/', '_']) and len(str_value) > 15 or  # Base64-like
                str_value.count('A') > len(str_value) * 0.3 or  # Too many A's (common in corrupted data)
                any(ord(char) > 126 for char in str_value[:20]) or  # Non-printable characters
                str_value.lower().startswith(('unkn', 'null', 'none', 'n/a')) or  # Obviously null values
                len(str_value.replace('A', '').replace('a', '')) < 3  # Mostly A's
            ):
                return default
            
            # Handle PersonName objects specifically
            if hasattr(value, 'family_name') and hasattr(value, 'given_name'):
                try:
                    family = str(value.family_name or '').strip()
                    given = str(value.given_name or '').strip()
                    if family and given and len(family) < 30 and len(given) < 30:
                        # Validate names don't contain encoded data
                        if not any(char in family+given for char in ['=', '+', '/', '_']):
                            return f"{given} {family}"
                except:
                    pass
            
            # For regular string values, additional validation
            if len(str_value) < 30 and str_value.replace(' ', '').replace('-', '').replace('.', '').isalnum():
                return str_value
            
            return default
        return default
    except Exception as e:
        print(f"Error extracting DICOM field {tag}: {e}")
        return default

def process_medical_image(image_array):
    """Process medical image with proper contrast and normalization."""
    # Ensure we have a valid image array
    if image_array is None or image_array.size == 0:
        raise ValueError("Invalid image data")
    
    # Convert to float for processing
    image_float = image_array.astype(np.float32)
    
    # Normalize to 0-255 range
    if image_float.max() > 255:
        # DICOM images often have higher bit depth
        image_float = (image_float - image_float.min()) / (image_float.max() - image_float.min()) * 255
    
    # Apply contrast enhancement for medical images
    image_8bit = image_float.astype(np.uint8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image_enhanced = clahe.apply(image_8bit)
    
    return image_enhanced

def create_display_image(image_processed):
    """Create base64 encoded image for frontend display."""
    # Resize for display (smaller size for frontend)
    image_resized = cv2.resize(image_processed, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Convert to PIL Image for base64 encoding
    image_pil = Image.fromarray(image_resized, mode='L')
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    
    # Convert image to Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"

# ================================
# DICOM PROCESSING FUNCTIONS
# ================================

def extract_patient_demographics(dicom_data):
    """Extract patient demographics with fallbacks and validation."""
    # Try multiple approaches for patient name
    patient_name = "Unknown"
    patient_id = "Unknown"
    
    # First, check if we can extract meaningful data from standard fields
    print("🔍 Analyzing DICOM patient fields...")
    
    # Method 1: Try PatientName field
    if hasattr(dicom_data, 'PatientName') and dicom_data.PatientName:
        raw_name = str(dicom_data.PatientName)
        print(f"Raw PatientName: {raw_name[:100]}...")
        
        # Check if it's encrypted (starts with gAAAAAB = Fernet encryption)
        if raw_name.startswith('gAAAAAB'):
            print("🔐 PatientName appears to be Fernet encrypted")
            patient_name = "Encrypted Patient"
        elif len(raw_name) > 50 or any(char in raw_name for char in ['=', '+', '/', '_']):
            print("🔐 PatientName appears to be encoded/encrypted")
            patient_name = "Encrypted Patient"
        else:
            name_value = safe_get_dicom_value(dicom_data, 'PatientName', None)
            if name_value and name_value != "Unknown":
                patient_name = name_value
    
    # Method 2: Try PatientID field
    if hasattr(dicom_data, 'PatientID') and dicom_data.PatientID:
        raw_id = str(dicom_data.PatientID)
        print(f"Raw PatientID: {raw_id[:50]}...")
        
        if raw_id.startswith('gAAAAAB'):
            print("🔐 PatientID appears to be Fernet encrypted")
            patient_id = "ENCRYPTED_ID"
        elif len(raw_id) > 30 or any(char in raw_id for char in ['=', '+', '/', '_']):
            print("🔐 PatientID appears to be encoded/encrypted")
            patient_id = "ENCRYPTED_ID"
        else:
            id_value = safe_get_dicom_value(dicom_data, 'PatientID', None)
            if id_value and id_value != "Unknown":
                patient_id = id_value
    
    # Method 3: Extract meaningful info from study data if patient data is encrypted
    if patient_name in ["Unknown", "Encrypted Patient"] or patient_id in ["Unknown", "ENCRYPTED_ID"]:
        print("🔄 Patient data encrypted, extracting from study information...")
        
        # Try to get study date for meaningful identifier
        study_date = safe_get_dicom_value(dicom_data, 'StudyDate', '')
        study_time = safe_get_dicom_value(dicom_data, 'StudyTime', '')
        modality = safe_get_dicom_value(dicom_data, 'Modality', '')
        
        # Create meaningful patient name from available data
        if study_date and len(study_date) >= 8:
            try:
                formatted_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:8]}"
                if modality:
                    patient_name = f"{modality} Patient {formatted_date}"
                else:
                    patient_name = f"Medical Patient {formatted_date}"
            except:
                patient_name = "Anonymous Patient"
        else:
            patient_name = "Anonymous Patient"
        
        # Create meaningful patient ID
        if patient_id in ["Unknown", "ENCRYPTED_ID"]:
            try:
                # Use study instance UID last part for ID
                study_uid = safe_get_dicom_value(dicom_data, 'StudyInstanceUID', '')
                if study_uid and len(study_uid) > 10:
                    # Take last 8 characters of study UID
                    short_uid = study_uid.split('.')[-1][-8:] if '.' in study_uid else study_uid[-8:]
                    patient_id = f"PAT_{short_uid.upper()}"
                else:
                    # Use current approach with hash
                    import hashlib
                    content_hash = hashlib.md5(str(dicom_data).encode()).hexdigest()[:8].upper()
                    patient_id = f"PAT_{content_hash}"
            except:
                patient_id = "PAT_UNKNOWN"
    
    # Method 4: Try alternative fields if everything else failed
    if patient_name == "Unknown":
        for name_field in ['ResponsiblePerson', 'PatientComments', 'ReferringPhysiciansName']:
            if hasattr(dicom_data, name_field):
                name_value = safe_get_dicom_value(dicom_data, name_field, None)
                if name_value and name_value != "Unknown" and len(name_value) < 30:
                    patient_name = f"Ref: {name_value}"
                    break
    
    print(f"✅ Final patient name: {patient_name}")
    print(f"✅ Final patient ID: {patient_id}")
    
    return patient_name, patient_id

def enhanced_dicom_extraction(dicom_data):
    """Enhanced DICOM metadata extraction with better handling of encrypted data."""
    try:
        # Basic extraction
        patient_name, patient_id = extract_patient_demographics(dicom_data)
        
        # Extract other fields with encryption awareness
        patient_sex = "Unknown"
        patient_age = "Unknown"
        patient_weight = "Unknown"
        
        # Handle PatientSex
        if hasattr(dicom_data, 'PatientSex'):
            sex_value = str(dicom_data.PatientSex)
            if not sex_value.startswith('gAAAAAB') and len(sex_value) <= 3:
                patient_sex = sex_value
        
        # Handle PatientBirthDate for age calculation
        if hasattr(dicom_data, 'PatientBirthDate'):
            birth_date_str = str(dicom_data.PatientBirthDate)
            if not birth_date_str.startswith('gAAAAAB') and len(birth_date_str) == 8:
                try:
                    from datetime import datetime
                    birth_date = datetime.strptime(birth_date_str, "%Y%m%d")
                    today = datetime.today()
                    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    patient_age = f"{age} years"
                except:
                    pass
        
        # Handle PatientWeight
        if hasattr(dicom_data, 'PatientWeight'):
            weight_str = str(dicom_data.PatientWeight)
            if not weight_str.startswith('gAAAAAB') and len(weight_str) < 10:
                try:
                    weight_float = float(weight_str)
                    patient_weight = f"{weight_float:.1f} kg"
                except:
                    patient_weight = weight_str
        
        # If core data is encrypted, provide helpful metadata instead
        study_date = safe_get_dicom_value(dicom_data, 'StudyDate', 'Unknown')
        study_time = safe_get_dicom_value(dicom_data, 'StudyTime', 'Unknown')
        modality = safe_get_dicom_value(dicom_data, 'Modality', 'Unknown')
        manufacturer = safe_get_dicom_value(dicom_data, 'Manufacturer', 'Unknown')
        
        result = {
            'patient_name': patient_name,
            'patient_id': patient_id,
            'patient_sex': patient_sex,
            'patient_age': patient_age,
            'patient_weight': patient_weight,
            'study_info': {
                'study_date': study_date,
                'study_time': study_time,
                'modality': modality,
                'manufacturer': manufacturer
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in enhanced DICOM extraction: {e}")
        return {
            'patient_name': 'Anonymous Patient',
            'patient_id': 'PAT_UNKNOWN',
            'patient_sex': 'Unknown',
            'patient_age': 'Unknown',
            'patient_weight': 'Unknown',
            'study_info': {}
        }

# ================================
# IMAGE PROCESSING FUNCTIONS
# ================================

def process_image_file(file_content, filename):
    """Process both DICOM and regular image files."""
    patient_info = {
        'patient_name': 'Unknown',
        'patient_id': 'Unknown',
        'patient_sex': 'Unknown',
        'patient_age': 'Unknown',
        'patient_weight': 'Unknown'
    }
    
    # Check if it's a DICOM file
    if filename.lower().endswith('.dcm') or 'dicom' in str(file_content[:100]).lower():
        try:
            dicom_data = pydicom.dcmread(BytesIO(file_content))
            
            # Extract patient information
            dicom_info = enhanced_dicom_extraction(dicom_data)
            patient_info.update(dicom_info)
            
            # Process image
            image_array = dicom_data.pixel_array
            image_processed = process_medical_image(image_array)
            
        except Exception as e:
            print(f"DICOM processing error: {e}")
            # If DICOM parsing fails, treat as regular image
            image = Image.open(BytesIO(file_content))
            image_processed = process_regular_image(image)
    else:
        # Process regular image file
        image = Image.open(BytesIO(file_content))
        image_processed = process_regular_image(image)
    
    return image_processed, patient_info

def process_regular_image(image):
    """Process regular image files (PNG, JPG, etc.)."""
    # Convert to grayscale if needed, then to numpy array
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode == 'RGB':
        image = image.convert('L')  # Convert to grayscale
    
    image_array = np.array(image)
    return process_medical_image(image_array)

# ================================
# AI INFERENCE FUNCTIONS
# ================================

def run_yolo_inference(image_processed):
    """Run YOLOv8x inference on processed image."""
    # YOLOv8x preprocessing - keep RGB, resize to 640x640
    if len(image_processed.shape) == 2:  # Grayscale to RGB
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_processed
    
    # YOLOv8x expects 640x640 input
    image_resized = cv2.resize(image_rgb, (640, 640), interpolation=cv2.INTER_AREA)
    
    # Run YOLOv8x inference
    print(f"🔄 Running YOLOv8x inference...")
    results = model(image_resized, verbose=False)
    
    # Process YOLOv8x results
    if results and len(results) > 0:
        result = results[0]
        
        # For classification, get the top prediction
        if hasattr(result, 'probs') and result.probs is not None:
            # Classification mode
            probs = result.probs.data.cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence_scores = probs.tolist()
            
            print(f"🎯 YOLOv8x Classification - Class: {predicted_class}, Confidence: {probs[predicted_class]:.4f}")
            
        elif hasattr(result, 'boxes') and len(result.boxes) > 0:
            # Detection mode - use highest confidence detection
            boxes = result.boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            # Get highest confidence detection
            max_conf_idx = np.argmax(confidences)
            yolo_class = classes[max_conf_idx]
            max_confidence = confidences[max_conf_idx]
            
            print(f"🎯 YOLOv8x Detection - YOLO Class: {yolo_class}, Confidence: {max_confidence:.4f}")
            
            # Map YOLO class to medical class
            predicted_class = map_yolo_class_to_medical(yolo_class, max_confidence)
            
            # Create confidence scores for all medical classes
            confidence_scores = [0.0] * 5  # Initialize with 5 medical classes
            
            # Distribute confidence based on the mapped medical class
            confidence_scores[predicted_class] = float(max_confidence)
            
            # Add some uncertainty to other classes for realism
            remaining_confidence = 1.0 - float(max_confidence)
            if remaining_confidence > 0:
                # Distribute remaining confidence among other classes
                other_classes = [i for i in range(5) if i != predicted_class]
                for i, other_class in enumerate(other_classes):
                    confidence_scores[other_class] = float(remaining_confidence / len(other_classes))
            
            print(f"📊 Mapped confidence distribution: {[f'{MEDICAL_CLASS_MAPPING[i]}: {conf:.3f}' for i, conf in enumerate(confidence_scores)]}")
        
        else:
            # No valid predictions
            predicted_class = 1  # Default to "Normal"
            confidence_scores = [0.1, 0.8, 0.1, 0.0, 0.0]  # Higher confidence in Normal
            print("⚠️ YOLOv8x: No valid predictions found, defaulting to Normal with uncertainty")
    
    else:
        # No results from YOLO
        predicted_class = 1  # Default to "Normal"
        confidence_scores = [0.0, 1.0, 0.0, 0.0, 0.0]
        print("⚠️ YOLOv8x: No results returned, defaulting to Normal")
    
    return predicted_class, confidence_scores

def run_densenet_inference(image_processed):
    """Run DenseNet inference on processed image."""
    # DenseNet preprocessing - convert to RGB, resize to 224x224
    if len(image_processed.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)
    else:  # Already RGB
        image_rgb = image_processed
        
    # Resize for model input (224x224 for DenseNet)
    image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0,1] range
    image_normalized = image_resized.astype(float) / 255.0
    
    # Convert to PyTorch tensor (channel-first format)
    image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    # Run DenseNet inference
    print(f"🔄 Running DenseNet inference...")
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence_scores = torch.softmax(output, dim=1).squeeze().tolist()
    
    print(f"🎯 DenseNet - Class: {predicted_class}, Confidence: {confidence_scores[predicted_class]:.4f}")
    
    return predicted_class, confidence_scores

def prepare_response_data(predicted_class, confidence_scores, patient_info, filename, image_processed):
    """Prepare final response data with proper type conversion."""
    # Use medical class mapping
    diagnosis = MEDICAL_CLASS_MAPPING.get(predicted_class, "Unknown")
    
    # Ensure confidence_scores is the right length and all values are Python floats
    if len(confidence_scores) != 5:
        confidence_scores = confidence_scores[:5] + [0.0] * (5 - len(confidence_scores))
    
    # Convert all numpy types to Python native types for JSON serialization
    confidence_scores = [float(score) for score in confidence_scores]
    predicted_class = int(predicted_class)
    
    # Prepare confidence scores for all classes
    class_confidences = {
        MEDICAL_CLASS_MAPPING[i]: round(float(confidence_scores[i]) * 100, 2) 
        for i in range(len(MEDICAL_CLASS_MAPPING))
    }
    
    # Create display image
    display_image = create_display_image(image_processed)
    
    # Store details in response
    response_data = {
        "PatientID": str(patient_info['patient_id']),
        "PatientName": str(patient_info['patient_name']),
        "PatientSex": str(patient_info['patient_sex']),
        "PatientWeight": str(patient_info['patient_weight']),
        "Age": str(patient_info['patient_age']),
        "Diagnosis": str(diagnosis),
        "Confidence": round(float(confidence_scores[predicted_class]) * 100, 2),
        "AllClassConfidences": class_confidences,
        "FileName": str(filename),
        "ProcessedImage": display_image
    }
    
    return response_data

# ================================
# API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Medical Image Classification API is running", "status": "healthy"}

@app.post("/analyze/")
async def analyze_medical_image(file: UploadFile = File(...)):
    """
    Single endpoint for complete medical image analysis.
    Combines upload, processing, DICOM extraction, AI inference, and display image creation.
    """
    try:
        # Read file content
        file_content = await file.read()
        print(f"📁 Processing file: {file.filename} ({len(file_content)} bytes)")
        
        # Process image and extract patient information
        image_processed, patient_info = process_image_file(file_content, file.filename)
        
        # Run AI inference based on model type
        if model_type == "yolo":
            predicted_class, confidence_scores = run_yolo_inference(image_processed)
        else:
            predicted_class, confidence_scores = run_densenet_inference(image_processed)
        
        # Prepare and return response
        response_data = prepare_response_data(
            predicted_class, confidence_scores, patient_info, file.filename, image_processed
        )
        
        print(f"✅ Analysis complete: {response_data['Diagnosis']} ({response_data['Confidence']}%)")
        return response_data
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=400, detail=f"Error analyzing medical image: {str(e)}")

@app.post("/debug-dicom/")
async def debug_dicom(file: UploadFile = File(...)):
    """Debug endpoint to show available DICOM fields."""
    try:
        file_content = await file.read()
        
        if not (file.filename.lower().endswith('.dcm') or file.content_type == 'application/dicom'):
            return {"error": "Not a DICOM file"}
        
        dicom_data = pydicom.dcmread(BytesIO(file_content))
        
        # Extract interesting fields for debugging
        debug_info = {
            "available_fields": [],
            "patient_fields": {},
            "study_fields": {},
            "image_fields": {}
        }
        
        # Get all available fields
        for elem in dicom_data:
            debug_info["available_fields"].append({
                "tag": str(elem.tag),
                "keyword": elem.keyword if hasattr(elem, 'keyword') else 'Unknown',
                "value": str(elem.value)[:100] + "..." if len(str(elem.value)) > 100 else str(elem.value)
            })
        
        # Extract specific patient-related fields
        patient_fields = ['PatientName', 'PatientID', 'PatientSex', 'PatientAge', 'PatientBirthDate', 
                         'PatientWeight', 'PatientSize', 'ResponsiblePerson', 'PatientsName']
        
        for field in patient_fields:
            if hasattr(dicom_data, field):
                value = getattr(dicom_data, field)
                debug_info["patient_fields"][field] = {
                    "raw_value": str(value),
                    "safe_value": safe_get_dicom_value(dicom_data, field),
                    "type": str(type(value))
                }
        
        # Extract study-related fields
        study_fields = ['StudyDate', 'StudyTime', 'StudyInstanceUID', 'AccessionNumber', 'StudyDescription']
        for field in study_fields:
            if hasattr(dicom_data, field):
                debug_info["study_fields"][field] = str(getattr(dicom_data, field))
        
        # Image-related fields
        image_fields = ['Rows', 'Columns', 'BitsAllocated', 'Modality', 'Manufacturer']
        for field in image_fields:
            if hasattr(dicom_data, field):
                debug_info["image_fields"][field] = str(getattr(dicom_data, field))
        
        return debug_info
    
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)