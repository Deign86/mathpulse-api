"""
ML Services for MathPulse AI
Integrates with Hugging Face for AI-powered educational features
"""
import json
import re
import os
import sys
from typing import Optional

from huggingface_hub import InferenceClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class HuggingFaceService:
    """Service for interacting with Hugging Face Inference API using official library"""
    
    def __init__(self):
        self.client = InferenceClient(token=config.HUGGINGFACE_API_TOKEN)
    
    async def query_chat(self, model_id: str, messages: list, max_tokens: int = 300) -> dict:
        """Send a chat completion request to a Hugging Face model"""
        print(f"🔄 Calling HuggingFace Chat API: {model_id}")
        
        try:
            response = self.client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            
            print(f"✅ HuggingFace Response received successfully")
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return {"generated_text": content}
            
            return {"error": "No response generated"}
            
        except Exception as e:
            print(f"❌ HuggingFace Exception: {str(e)}")
            return {"error": str(e)}
    
    async def query_classification(self, model_id: str, text: str, labels: list) -> dict:
        """Send a zero-shot classification request"""
        print(f"🔄 Calling HuggingFace Classification API: {model_id}")
        
        try:
            response = self.client.zero_shot_classification(
                text=text,
                labels=labels,
                model=model_id
            )
            
            print(f"✅ HuggingFace Classification received")
            return response
            
        except Exception as e:
            print(f"❌ HuggingFace Classification Exception: {str(e)}")
            return {"error": str(e)}


class AITutorService:
    """AI Tutor for math education - uses LLM for conversational help"""
    
    def __init__(self):
        self.hf_service = HuggingFaceService()
        self.model_id = config.CHAT_MODEL_ID
        self.system_prompt = """You are MathPulse AI, a friendly and encouraging math tutor for high school students.
Your role is to:
- Help students understand mathematical concepts step by step
- Provide clear explanations using simple language
- Give examples when helpful
- Encourage students and praise their efforts
- Focus on building intuition, not just memorization
- If a student is struggling, break problems into smaller steps

Current topic focus: Calculus and Derivatives
Keep responses concise but helpful (2-3 paragraphs max)."""

    async def get_response(self, user_message: str, conversation_history: list | None = None) -> str:
        """Generate a response from the AI tutor"""
        
        # Build the messages for chat completion
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = "user" if msg.get("sender") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("message", "")})
        
        messages.append({"role": "user", "content": user_message})
        
        # Call the HuggingFace chat API
        result = await self.hf_service.query_chat(self.model_id, messages)
        
        if "error" in result:
            # Fallback response if API fails
            return self._get_fallback_response(user_message)
        
        # Extract generated text
        generated = result.get("generated_text", "")
        if generated:
            return generated
        
        return self._get_fallback_response(user_message)
    
    def _get_fallback_response(self, user_message: str) -> str:
        """Provide a fallback response when API is unavailable"""
        keywords = user_message.lower()
        
        if "derivative" in keywords or "differentiate" in keywords:
            return "The derivative measures how a function changes as its input changes. Think of it as the 'instantaneous rate of change' or the slope of the tangent line at any point. For example, if f(x) = x², then f'(x) = 2x. Would you like me to walk through more examples?"
        
        if "chain rule" in keywords:
            return "The chain rule is used when you have a function inside another function (composition). If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Think of it as 'derivative of the outside × derivative of the inside'. What specific problem are you working on?"
        
        if "integral" in keywords or "integrate" in keywords:
            return "Integration is the reverse of differentiation. While derivatives give us rates of change, integrals help us find areas under curves and accumulate quantities. The basic rule: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C. What would you like to integrate?"
        
        if "limit" in keywords:
            return "Limits describe what value a function approaches as the input gets closer to some value. We write it as lim(x→a) f(x) = L. Limits are the foundation of calculus - they help us define derivatives and integrals precisely. What limit are you trying to solve?"
        
        return "That's a great question! I'm here to help you understand math concepts better. Could you tell me more about what specific part you're finding challenging? Breaking it down into smaller steps often helps."


class RiskPredictionService:
    """Predicts student risk levels using ML-based analysis"""
    
    def __init__(self):
        self.hf_service = HuggingFaceService()
        self.model_id = config.CLASSIFICATION_MODEL_ID
    
    async def predict_risk(self, student_data: dict) -> dict:
        """
        Predict risk level for a student based on their metrics
        
        Args:
            student_data: Dict containing engagementScore, avgQuizScore, etc.
        
        Returns:
            Risk assessment with level and confidence
        """
        # Build a description of the student's performance for classification
        engagement = student_data.get("engagementScore", 50)
        quiz_score = student_data.get("avgQuizScore", 50)
        weakest_topic = student_data.get("weakestTopic", "Unknown")
        
        # Create a text description for zero-shot classification
        student_description = f"""
        Student performance profile:
        - Average quiz score: {quiz_score}%
        - Engagement score: {engagement}%
        - Struggling with: {weakest_topic}
        """
        
        labels = ["high risk of academic failure", "moderate risk needs support", "low risk performing well"]
        
        result = await self.hf_service.query_classification(self.model_id, student_description, labels)
        
        if "error" in result:
            # Fallback to rule-based prediction
            return self._rule_based_prediction(engagement, quiz_score, weakest_topic)
        
        # Parse classification result from huggingface_hub format
        try:
            # The result is a list of ClassificationOutputElement objects
            if result and len(result) > 0:
                top_result = result[0]
                top_label = top_result.label if hasattr(top_result, 'label') else str(top_result)
                confidence = top_result.score if hasattr(top_result, 'score') else 0.5
                
                risk_level = "High" if "high" in top_label else "Medium" if "moderate" in top_label else "Low"
                
                return {
                    "riskLevel": risk_level,
                    "confidence": round(confidence * 100, 1),
                    "analysis": self._generate_analysis(risk_level, engagement, quiz_score, weakest_topic),
                    "factors": self._identify_risk_factors(engagement, quiz_score)
                }
        except Exception as e:
            print(f"⚠️ Classification parse error: {e}")
        
        return self._rule_based_prediction(engagement, quiz_score, weakest_topic)
    
    def _rule_based_prediction(self, engagement: float, quiz_score: float, weakest_topic: str) -> dict:
        """Fallback rule-based risk prediction"""
        combined_score = (engagement + quiz_score) / 2
        
        if combined_score < 55:
            risk_level = "High"
            confidence = 85.0
        elif combined_score < 75:
            risk_level = "Medium"
            confidence = 75.0
        else:
            risk_level = "Low"
            confidence = 80.0
        
        return {
            "riskLevel": risk_level,
            "confidence": confidence,
            "analysis": self._generate_analysis(risk_level, engagement, quiz_score, weakest_topic),
            "factors": self._identify_risk_factors(engagement, quiz_score)
        }
    
    def _generate_analysis(self, risk_level: str, engagement: float, quiz_score: float, topic: str) -> str:
        """Generate human-readable analysis"""
        if risk_level == "High":
            return f"Student shows concerning patterns with {quiz_score}% quiz average and {engagement}% engagement. Priority intervention recommended for {topic}."
        elif risk_level == "Medium":
            return f"Student performing adequately but showing some weakness in {topic}. Consider targeted support to prevent further decline."
        else:
            return f"Student performing well overall. Continue monitoring and encourage advancement in challenging areas."
    
    def _identify_risk_factors(self, engagement: float, quiz_score: float) -> list:
        """Identify specific risk factors"""
        factors = []
        
        if quiz_score < 60:
            factors.append({"factor": "Low quiz performance", "severity": "high", "value": f"{quiz_score}%"})
        elif quiz_score < 75:
            factors.append({"factor": "Below average quiz scores", "severity": "medium", "value": f"{quiz_score}%"})
        
        if engagement < 50:
            factors.append({"factor": "Low engagement", "severity": "high", "value": f"{engagement}%"})
        elif engagement < 70:
            factors.append({"factor": "Moderate engagement", "severity": "medium", "value": f"{engagement}%"})
        
        return factors


class LearningPathService:
    """Generates personalized learning paths based on student needs"""
    
    def __init__(self):
        self.hf_service = HuggingFaceService()
        self.model_id = config.CHAT_MODEL_ID
    
    async def generate_learning_path(self, student_data: dict) -> dict:
        """Generate a personalized learning path for a student"""
        
        weakest_topic = student_data.get("weakestTopic", "Calculus - Derivatives")
        quiz_score = student_data.get("avgQuizScore", 50)
        engagement = student_data.get("engagementScore", 50)
        
        # Try API first, but have good fallback
        try:
            prompt = f"""Create a 5-step learning path for a student struggling with {weakest_topic}. 
Quiz average: {quiz_score}%, Engagement: {engagement}%.
List 5 steps with topic, type (video/quiz/exercise), and duration."""

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 400,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = await self.hf_service.query_model(self.model_id, payload)
            
            if "error" not in result and isinstance(result, list):
                # Parse and use if valid, otherwise fallback
                generated = result[0].get("generated_text", "")
                if len(generated) > 50:
                    return {
                        "generatedPath": True,
                        "steps": self._parse_generated_steps(generated, weakest_topic),
                        "summary": f"Personalized path for {weakest_topic}"
                    }
        except Exception:
            pass
        
        # Return curated fallback path
        return self._get_fallback_path(weakest_topic, quiz_score)
    
    def _parse_generated_steps(self, text: str, topic: str) -> list:
        """Parse generated text into structured steps"""
        # Try to extract steps from generated text, fallback if parsing fails
        steps = []
        
        # Simple parsing - look for numbered items
        lines = text.split('\n')
        step_num = 1
        
        for line in lines:
            if step_num > 5:
                break
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                steps.append({
                    "step": step_num,
                    "topic": line.lstrip('0123456789.-•) '),
                    "type": "video" if step_num <= 2 else "quiz" if step_num == 3 else "exercise",
                    "duration": f"{10 + step_num * 2} min"
                })
                step_num += 1
        
        # If parsing didn't work well, return fallback
        if len(steps) < 3:
            return self._get_fallback_path(topic, 50)["steps"]
        
        return steps
    
    def _get_fallback_path(self, topic: str, quiz_score: float) -> dict:
        """Get a curated fallback learning path"""
        
        # Topic-specific paths
        paths = {
            "derivatives": [
                {"step": 1, "topic": "Visual Introduction to Derivatives", "type": "video", "duration": "12 min"},
                {"step": 2, "topic": "Understanding Rate of Change", "type": "video", "duration": "10 min"},
                {"step": 3, "topic": "Basic Differentiation Rules Quiz", "type": "quiz", "duration": "15 min"},
                {"step": 4, "topic": "Chain Rule Practice", "type": "exercise", "duration": "20 min"},
                {"step": 5, "topic": "Comprehensive Derivatives Assessment", "type": "quiz", "duration": "25 min"}
            ],
            "quadratic": [
                {"step": 1, "topic": "Quadratic Functions Overview", "type": "video", "duration": "10 min"},
                {"step": 2, "topic": "Factoring Techniques", "type": "video", "duration": "12 min"},
                {"step": 3, "topic": "Quadratic Formula Practice", "type": "exercise", "duration": "15 min"},
                {"step": 4, "topic": "Graphing Parabolas", "type": "exercise", "duration": "18 min"},
                {"step": 5, "topic": "Quadratic Equations Mastery Quiz", "type": "quiz", "duration": "20 min"}
            ],
            "trigonometry": [
                {"step": 1, "topic": "Unit Circle Fundamentals", "type": "video", "duration": "15 min"},
                {"step": 2, "topic": "Sine, Cosine, Tangent Explained", "type": "video", "duration": "12 min"},
                {"step": 3, "topic": "Trig Ratios Practice", "type": "exercise", "duration": "15 min"},
                {"step": 4, "topic": "Trig Identities Introduction", "type": "video", "duration": "10 min"},
                {"step": 5, "topic": "Trigonometry Assessment", "type": "quiz", "duration": "20 min"}
            ],
            "default": [
                {"step": 1, "topic": f"Introduction to {topic}", "type": "video", "duration": "12 min"},
                {"step": 2, "topic": "Core Concepts Review", "type": "video", "duration": "10 min"},
                {"step": 3, "topic": "Fundamentals Check Quiz", "type": "quiz", "duration": "15 min"},
                {"step": 4, "topic": "Guided Practice Problems", "type": "exercise", "duration": "20 min"},
                {"step": 5, "topic": "Comprehensive Review", "type": "quiz", "duration": "25 min"}
            ]
        }
        
        # Match topic to path
        topic_lower = topic.lower()
        if "derivative" in topic_lower or "calculus" in topic_lower:
            steps = paths["derivatives"]
        elif "quadratic" in topic_lower or "algebra" in topic_lower:
            steps = paths["quadratic"]
        elif "trig" in topic_lower or "unit circle" in topic_lower:
            steps = paths["trigonometry"]
        else:
            steps = paths["default"]
        
        return {
            "generatedPath": False,
            "steps": steps,
            "summary": f"Remedial learning path for {topic}"
        }


class ClassAnalyticsService:
    """Provides AI-powered class-wide analytics and insights"""
    
    def __init__(self):
        self.hf_service = HuggingFaceService()
    
    async def generate_daily_insight(self, students: list) -> dict:
        """Generate a daily AI insight based on class performance"""
        
        # Calculate class statistics
        total_students = len(students)
        if total_students == 0:
            return {"insight": "No student data available.", "trends": []}
        
        avg_engagement = sum(s.get("engagementScore", 0) for s in students) / total_students
        avg_quiz = sum(s.get("avgQuizScore", 0) for s in students) / total_students
        high_risk = sum(1 for s in students if s.get("riskLevel") == "High")
        
        # Find struggling topics
        topic_counts = {}
        for s in students:
            topic = s.get("weakestTopic", "Unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        most_common_weakness = max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else "Unknown"
        
        # Generate insight
        if high_risk > total_students * 0.3:
            insight = f"⚠️ Alert: {high_risk} students ({round(high_risk/total_students*100)}%) are at high risk. Most are struggling with {most_common_weakness}. Consider a class review session."
        elif avg_engagement < 60:
            insight = f"📊 Class engagement is at {round(avg_engagement)}%. Consider interactive activities to boost participation."
        elif avg_quiz > 80:
            insight = f"🎉 Great news! Class average is {round(avg_quiz)}%. Students are excelling - consider introducing advanced topics."
        else:
            insight = f"📈 Class is performing steadily with {round(avg_quiz)}% average. Focus area: {most_common_weakness} ({topic_counts.get(most_common_weakness, 0)} students struggling)."
        
        trends = [
            {"metric": "Class Average", "value": f"{round(avg_quiz)}%", "trend": "up" if avg_quiz > 70 else "down"},
            {"metric": "Engagement", "value": f"{round(avg_engagement)}%", "trend": "up" if avg_engagement > 65 else "down"},
            {"metric": "At Risk", "value": f"{high_risk} students", "trend": "down" if high_risk < 3 else "up"}
        ]
        
        return {
            "insight": insight,
            "trends": trends,
            "focusTopic": most_common_weakness,
            "recommendations": self._generate_recommendations(avg_quiz, avg_engagement, high_risk)
        }
    
    def _generate_recommendations(self, avg_quiz: float, avg_engagement: float, high_risk: int) -> list:
        """Generate actionable recommendations"""
        recs = []
        
        if high_risk > 2:
            recs.append({
                "priority": "high",
                "action": "Schedule one-on-one sessions with high-risk students",
                "impact": "Direct intervention for struggling students"
            })
        
        if avg_engagement < 60:
            recs.append({
                "priority": "medium",
                "action": "Introduce gamified learning activities",
                "impact": "Boost engagement through interactive content"
            })
        
        if avg_quiz < 70:
            recs.append({
                "priority": "medium",
                "action": "Review foundational concepts as a class",
                "impact": "Strengthen baseline understanding"
            })
        
        return recs
