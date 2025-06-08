# Medical Image Classification System

A comprehensive AI-powered medical image classification system that analyzes DICOM files and medical images to classify various pulmonary conditions including TB Positive, Normal, COPD, Silicosis, and Lung Cancer.

## Features

- **Multi-class Classification**: Detects 5 different conditions:
  - TB Positive
  - Normal
  - COPD
  - Silicosis  
  - Lung Cancer

- **Multiple File Format Support**: 
  - DICOM files (.dcm)
  - Standard image formats (PNG, JPG, JPEG, GIF, BMP)

- **Multi-language Interface**: 
  - English
  - Hindi (हिन्दी)
  - Telugu (తెలుగు)
  - Urdu (اردو)

- **Advanced Features**:
  - AI-powered analysis using DenseNet-121
  - Patient metadata extraction from DICOM files
  - Confidence scores for all classifications
  - Professional medical report generation
  - HIPAA-compliant secure processing

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- All dependencies listed in Backend/requirements.txt

### Installation & Setup

1. **Clone or navigate to the project directory:**
```bash
cd /path/to/MedicalImage
```

2. **Activate the virtual environment:**
```bash
source env/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
cd Backend
pip install -r requirements.txt
```

### Running the Application

#### Method 1: Using the Startup Scripts

**Start Backend:**
```bash
./start_backend.sh
```

**Start Frontend (in a new terminal):**
```bash
./start_frontend.sh
```

#### Method 2: Manual Start

**Backend Server:**
```bash
cd Backend
python app.py
```

**Frontend Server (in a new terminal):**
```bash
cd FrontEnd
python -m http.server 3000
```

### Access the Application

- **Frontend Interface:** http://127.0.0.1:3000/index1.html
- **Backend API:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs

## 🖥️ New Frontend Features

The frontend has been completely redesigned for better functionality:

### **Clean, Modern Interface**
- Professional medical interface using Tailwind CSS
- Responsive design for different screen sizes
- Intuitive drag-and-drop file upload

### **Improved User Workflow**
1. **File Upload**: Drag & drop or browse for DICOM/image files
2. **Real-time Validation**: Instant file type and size validation
3. **Progress Tracking**: Visual progress bar during analysis
4. **Detailed Results**: Comprehensive analysis results with confidence scores

### **Enhanced File Support**
- **DICOM Files**: `.dcm` files with metadata extraction
- **Standard Images**: PNG, JPG, JPEG, GIF, BMP formats
- **File Size Limit**: Up to 50MB per file

### **Results Display**
- **Primary Diagnosis**: Clear display of the main classification
- **Confidence Scores**: Percentage confidence for all 5 conditions:
  - TB Positive
  - Normal
  - COPD
  - Silicosis
  - Lung Cancer
- **Patient Information**: Extracted metadata from DICOM files
- **Visual Progress Bars**: Easy-to-read confidence visualization

### **Additional Features**
- **Download Reports**: Generate and download analysis reports
- **Error Handling**: Comprehensive error messages and validation
- **Reset Functionality**: Easy reset for new analyses
- **Medical Disclaimers**: Important notices about AI assistance

## 🔧 API Endpoints

### **Health Check**
- `GET /` - Returns API status

### **Image Upload**
- `POST /upload/` - Upload image for display
- Returns: `{"image": "base64_encoded_image"}`

### **Prediction**
- `POST /predict/` - Analyze medical image
- Returns comprehensive analysis including:
  - Primary diagnosis
  - Confidence scores for all conditions
  - Patient metadata (from DICOM)
  - File information

## 🧠 AI Model Information

- **Architecture**: DenseNet-121
- **Model Files**: 
  - `densenet_final_model.pt` (primary)
  - `augmented_densenet_final_model.pt` (backup)
- **Input**: 224x224 RGB images
- **Output**: 5 medical conditions with confidence scores

## 📁 Project Structure

```
MedicalImage/
├── Backend/
│   ├── app.py              # FastAPI server
│   ├── ModelClass.py       # AI model wrapper
│   ├── dicom_image.py      # DICOM processing
│   └── requirements.txt    # Python dependencies
├── FrontEnd/
│   ├── index1.html         # Main web interface
│   └── main.js            # Frontend JavaScript
├── Models/
│   ├── densenet_final_model.pt
│   └── augmented_densenet_final_model.pt
├── start_backend.sh        # Backend startup script
├── start_frontend.sh       # Frontend startup script
└── README.md
```

## 🔒 Security & Compliance

- **HIPAA Considerations**: Designed with medical data privacy in mind
- **Local Processing**: All analysis performed locally
- **No Data Storage**: Files processed in memory only
- **Secure Uploads**: Temporary file handling with automatic cleanup

## ⚠️ Important Medical Disclaimer

**This AI system is designed to assist medical professionals and should not be used as the sole basis for medical diagnosis. Always consult qualified healthcare providers for proper medical interpretation and decision-making.**

## 🐛 Troubleshooting

### Backend Issues
```bash
# Check if backend is running
curl http://127.0.0.1:8000/

# Kill existing processes if needed
lsof -ti:8000 | xargs kill -9
```

### Frontend Issues
```bash
# Check if frontend is accessible
curl -I http://127.0.0.1:3000/index1.html

# Kill existing processes if needed
lsof -ti:3000 | xargs kill -9
```

### CORS Issues
The backend includes CORS headers for development. If you encounter CORS errors:
1. Ensure both servers are running
2. Access frontend via http://127.0.0.1:3000 (not localhost)
3. Check browser console for specific error messages

## 🚀 Production Deployment

For production deployment:
1. Use proper ASGI server (uvicorn, gunicorn)
2. Configure proper CORS settings
3. Implement authentication if required
4. Use HTTPS for secure communication
5. Consider containerization with Docker

## 📞 Support

For technical issues or questions about the medical classification system, please ensure you have:
1. Backend server running on port 8000
2. Frontend server running on port 3000
3. Proper virtual environment activation
4. All dependencies installed correctly

## License

This medical image classification system is provided for educational and research purposes. Please ensure proper medical professional oversight when used in clinical settings.

---

**Important**: This AI system is designed to assist medical professionals and should not be used as the sole basis for medical diagnosis. Always consult qualified healthcare providers for proper medical interpretation and decision-making. 