"""
MathPulse AI - FastAPI Backend
Provides ML-powered educational features using Hugging Face
"""
import sys
import os
from datetime import datetime
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from config import config
from models import (
    ChatRequest, ChatResponse,
    StudentData, RiskPredictionResponse,
    LearningPathRequest, LearningPathResponse,
    DailyInsightResponse, BatchRiskRequest,
    FileUploadResponse, CourseMaterialResponse
)
from services.ml_services import (
    AITutorService,
    RiskPredictionService,
    LearningPathService,
    ClassAnalyticsService,
    HuggingFaceService
)
from services.document_parser import DocumentParser, CourseMaterialParser


# Initialize services
ai_tutor = AITutorService()
risk_predictor = RiskPredictionService()
learning_path_service = LearningPathService()
analytics_service = ClassAnalyticsService()
hf_service = HuggingFaceService()
document_parser = DocumentParser(hf_service)
course_parser = CourseMaterialParser(hf_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 MathPulse AI Backend starting up...")
    print(f"📡 Hugging Face API configured: {'✓' if config.HUGGINGFACE_API_TOKEN else '✗ (using fallbacks)'}")
    yield
    print("👋 MathPulse AI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MathPulse AI API",
    description="ML-powered educational backend for MathPulse AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Health Check ============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MathPulse AI Backend",
        "version": "1.0.0",
        "huggingface_configured": bool(config.HUGGINGFACE_API_TOKEN)
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ai_tutor": "ready",
            "risk_prediction": "ready",
            "learning_path": "ready",
            "analytics": "ready"
        }
    }


# ============ AI Tutor Endpoints ============

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_tutor(request: ChatRequest):
    """
    Chat with the AI Math Tutor
    
    Send a message and optionally include conversation history for context.
    """
    try:
        # Convert history to the format expected by the service
        history = []
        if request.conversationHistory:
            history = [
                {"sender": msg.sender, "message": msg.message}
                for msg in request.conversationHistory
            ]
        
        response = await ai_tutor.get_response(request.message, history)
        
        return ChatResponse(
            message=response,
            timestamp=datetime.now().strftime("%I:%M %p")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/chat/simple")
async def simple_chat(message: str):
    """Simple chat endpoint - just send a message string"""
    try:
        response = await ai_tutor.get_response(message)
        return {"message": response, "timestamp": datetime.now().strftime("%I:%M %p")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Risk Prediction Endpoints ============

@app.post("/api/predict-risk", response_model=RiskPredictionResponse)
async def predict_student_risk(student: StudentData):
    """
    Predict risk level for a single student
    
    Analyzes student metrics to determine academic risk level.
    """
    try:
        result = await risk_predictor.predict_risk(student.model_dump())
        return RiskPredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk prediction error: {str(e)}")


@app.post("/api/predict-risk/batch")
async def batch_predict_risk(request: BatchRiskRequest):
    """
    Predict risk levels for multiple students
    
    Returns risk assessments for all students in the batch.
    """
    try:
        results = []
        for student in request.students:
            result = await risk_predictor.predict_risk(student.model_dump())
            results.append({
                "studentId": student.id,
                "studentName": student.name,
                **result
            })
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# ============ Learning Path Endpoints ============

@app.post("/api/learning-path", response_model=LearningPathResponse)
async def generate_learning_path(request: LearningPathRequest):
    """
    Generate a personalized learning path for a student
    
    Creates a 5-step remedial path based on student's weak areas.
    """
    try:
        student_data = {
            "weakestTopic": request.weakestTopic,
            "avgQuizScore": request.avgQuizScore,
            "engagementScore": request.engagementScore
        }
        result = await learning_path_service.generate_learning_path(student_data)
        return LearningPathResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning path generation error: {str(e)}")


@app.get("/api/learning-path/{topic}")
async def get_learning_path_by_topic(topic: str, quiz_score: float = 60, engagement: float = 60):
    """
    Get a learning path for a specific topic
    
    Quick endpoint to generate a path without full student data.
    """
    try:
        student_data = {
            "weakestTopic": topic,
            "avgQuizScore": quiz_score,
            "engagementScore": engagement
        }
        result = await learning_path_service.generate_learning_path(student_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Analytics Endpoints ============

@app.post("/api/analytics/daily-insight", response_model=DailyInsightResponse)
async def get_daily_insight(request: BatchRiskRequest):
    """
    Generate daily AI insight for the class
    
    Analyzes overall class performance and generates actionable insights.
    """
    try:
        students_data = [s.model_dump() for s in request.students]
        result = await analytics_service.generate_daily_insight(students_data)
        return DailyInsightResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """
    Get a summary of available analytics features
    """
    return {
        "features": [
            {
                "name": "Risk Prediction",
                "description": "ML-based student risk assessment",
                "endpoint": "/api/predict-risk"
            },
            {
                "name": "AI Tutor",
                "description": "Conversational math tutoring",
                "endpoint": "/api/chat"
            },
            {
                "name": "Learning Paths",
                "description": "Personalized remedial content generation",
                "endpoint": "/api/learning-path"
            },
            {
                "name": "Daily Insights",
                "description": "Class-wide performance analytics",
                "endpoint": "/api/analytics/daily-insight"
            }
        ]
    }


# ============ File Upload Endpoints ============

@app.post("/api/upload/class-records", response_model=FileUploadResponse)
async def upload_class_records(file: UploadFile = File(...)):
    """
    Upload and parse class records file (CSV, Excel, PDF)
    
    Intelligently detects file format and extracts student data.
    Supports:
    - CSV files with various delimiters
    - Excel files (.xlsx, .xls) with multiple sheets
    - PDF files with tables
    - Word documents (.docx) with tables
    
    The AI will attempt to map columns regardless of naming conventions.
    """
    try:
        # Check file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
        # Check file type
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.pdf', '.docx', '.doc'}
        file_ext = os.path.splitext(file.filename or '')[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Parse the file
        result = await document_parser.parse_file(file.filename or "unknown", contents)
        
        return FileUploadResponse(**result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/upload/course-materials", response_model=CourseMaterialResponse)
async def upload_course_materials(file: UploadFile = File(...)):
    """
    Upload and analyze course materials (syllabus, lesson plans, etc.)
    
    Extracts:
    - Math topics covered
    - Course structure/outline
    - Assessment information
    - Learning objectives
    """
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        file_ext = os.path.splitext(file.filename or '')[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        result = await course_parser.parse_course_material(file.filename or "unknown", contents)
        
        return CourseMaterialResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/upload/supported-formats")
async def get_supported_formats():
    """Get information about supported file formats"""
    return {
        "classRecords": {
            "formats": [
                {"extension": ".csv", "description": "Comma-separated values", "supported": True},
                {"extension": ".xlsx", "description": "Excel workbook", "supported": True},
                {"extension": ".xls", "description": "Legacy Excel format", "supported": True},
                {"extension": ".pdf", "description": "PDF with tables", "supported": True},
                {"extension": ".docx", "description": "Word document", "supported": True}
            ],
            "maxSize": "10MB",
            "notes": [
                "AI will automatically detect column mappings",
                "Supports various naming conventions for columns",
                "Can handle multiple sheets in Excel files",
                "Extracts tables from PDFs automatically"
            ]
        },
        "courseMaterials": {
            "formats": [
                {"extension": ".pdf", "description": "PDF documents", "supported": True},
                {"extension": ".docx", "description": "Word documents", "supported": True},
                {"extension": ".txt", "description": "Plain text", "supported": True}
            ],
            "maxSize": "10MB",
            "notes": [
                "Extracts math topics and curriculum structure",
                "Identifies assessment types",
                "Detects learning objectives"
            ]
        }
    }


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG
    )
