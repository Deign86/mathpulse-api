"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# ============ Chat Models ============

class ChatMessageInput(BaseModel):
    """Single chat message from user"""
    message: str


class ChatHistoryMessage(BaseModel):
    """Message in conversation history"""
    id: str
    sender: str  # "user" or "ai"
    message: str
    timestamp: str


class ChatRequest(BaseModel):
    """Full chat request with optional history"""
    message: str
    conversationHistory: Optional[List[ChatHistoryMessage]] = None


class ChatResponse(BaseModel):
    """AI tutor response"""
    message: str
    timestamp: str


# ============ Risk Prediction Models ============

class StudentData(BaseModel):
    """Student data for risk prediction"""
    id: str
    name: str
    engagementScore: float
    avgQuizScore: float
    weakestTopic: str
    riskLevel: Optional[RiskLevel] = None


class RiskFactor(BaseModel):
    """Individual risk factor"""
    factor: str
    severity: str
    value: str


class RiskPredictionResponse(BaseModel):
    """Risk prediction result"""
    riskLevel: RiskLevel
    confidence: float
    analysis: str
    factors: List[RiskFactor]


# ============ Learning Path Models ============

class LearningStep(BaseModel):
    """Single step in learning path"""
    step: int
    topic: str
    type: str  # "video", "quiz", "exercise"
    duration: str


class LearningPathRequest(BaseModel):
    """Request for learning path generation"""
    studentId: str
    weakestTopic: str
    avgQuizScore: float
    engagementScore: float


class LearningPathResponse(BaseModel):
    """Generated learning path"""
    generatedPath: bool
    steps: List[LearningStep]
    summary: str


# ============ Analytics Models ============

class Trend(BaseModel):
    """Trend metric"""
    metric: str
    value: str
    trend: str  # "up" or "down"


class Recommendation(BaseModel):
    """Actionable recommendation"""
    priority: str
    action: str
    impact: str


class DailyInsightResponse(BaseModel):
    """Daily AI insight response"""
    insight: str
    trends: List[Trend]
    focusTopic: str
    recommendations: List[Recommendation]


class BatchRiskRequest(BaseModel):
    """Request for batch risk analysis"""
    students: List[StudentData]


# ============ File Upload Models ============

class ParsedStudentData(BaseModel):
    """Parsed student from uploaded file"""
    name: str
    engagementScore: float = 50
    avgQuizScore: float = 50
    attendance: Optional[float] = None
    weakestTopic: str = "General Mathematics"
    grades: Optional[dict] = None
    rawData: Optional[dict] = None


class FileUploadResponse(BaseModel):
    """Response from file upload parsing"""
    success: bool
    students: List[ParsedStudentData]
    fileType: str
    columnsDetected: List[str]
    mappingConfidence: float
    warnings: List[str]
    rawPreview: Optional[str] = None
    courseInfo: Optional[dict] = None
    studentCount: int = 0


class CourseMaterialResponse(BaseModel):
    """Response from course material parsing"""
    success: bool
    filename: str
    courseInfo: Optional[dict] = None
    textPreview: Optional[str] = None
    error: Optional[str] = None

