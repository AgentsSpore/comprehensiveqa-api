"""Document Store with sample data for medical, legal, and research domains"""

from typing import List, Dict


class DocumentStore:
    """In-memory document storage with sample multi-answer datasets"""
    
    def __init__(self):
        self.documents = self._load_sample_documents()
    
    def _load_sample_documents(self) -> List[Dict]:
        """Load sample documents across different domains"""
        
        # Medical domain - Type 2 Diabetes treatments
        medical_docs = [
            {
                "text": "Metformin is the first-line medication for type 2 diabetes, reducing glucose production in the liver and improving insulin sensitivity.",
                "source": "medical_db_001",
                "category": "medical"
            },
            {
                "text": "Lifestyle modifications including regular exercise and Mediterranean diet have shown significant improvements in glycemic control for type 2 diabetes patients.",
                "source": "medical_db_002",
                "category": "medical"
            },
            {
                "text": "GLP-1 receptor agonists like semaglutide provide effective blood sugar control and promote weight loss in type 2 diabetes treatment.",
                "source": "medical_db_003",
                "category": "medical"
            },
            {
                "text": "SGLT2 inhibitors offer cardiovascular and renal benefits beyond glycemic control for type 2 diabetes management.",
                "source": "medical_db_004",
                "category": "medical"
            },
            {
                "text": "Insulin therapy remains essential for advanced type 2 diabetes when oral medications are insufficient.",
                "source": "medical_db_005",
                "category": "medical"
            },
            {
                "text": "DPP-4 inhibitors provide moderate glycemic control with low risk of hypoglycemia in type 2 diabetes treatment.",
                "source": "medical_db_006",
                "category": "medical"
            },
        ]
        
        # Legal domain - Employment discrimination precedents
        legal_docs = [
            {
                "text": "McDonnell Douglas Corp. v. Green established the burden-shifting framework for employment discrimination cases under Title VII.",
                "source": "legal_db_001",
                "category": "legal"
            },
            {
                "text": "Griggs v. Duke Power Co. held that employment practices with disparate impact on protected groups violate Title VII even without discriminatory intent.",
                "source": "legal_db_002",
                "category": "legal"
            },
            {
                "text": "Price Waterhouse v. Hopkins established that mixed-motive discrimination cases require showing discrimination was a motivating factor.",
                "source": "legal_db_003",
                "category": "legal"
            },
            {
                "text": "Oncale v. Sundowner Offshore Services held that same-sex harassment is actionable under Title VII's prohibition of sex discrimination.",
                "source": "legal_db_004",
                "category": "legal"
            },
            {
                "text": "Meritor Savings Bank v. Vinson recognized hostile work environment as a form of actionable sexual harassment.",
                "source": "legal_db_005",
                "category": "legal"
            },
        ]
        
        # Research domain - Machine learning interpretability methods
        research_docs = [
            {
                "text": "LIME (Local Interpretable Model-agnostic Explanations) provides local approximations of model predictions using interpretable models.",
                "source": "research_db_001",
                "category": "research"
            },
            {
                "text": "SHAP (SHapley Additive exPlanations) uses game theory to assign feature importance values ensuring consistency and local accuracy.",
                "source": "research_db_002",
                "category": "research"
            },
            {
                "text": "Integrated Gradients attributes predictions to input features by integrating gradients along a path from baseline to input.",
                "source": "research_db_003",
                "category": "research"
            },
            {
                "text": "Attention visualization methods reveal which input tokens neural networks focus on during prediction.",
                "source": "research_db_004",
                "category": "research"
            },
            {
                "text": "Layer-wise Relevance Propagation (LRP) decomposes predictions backward through network layers to identify relevant features.",
                "source": "research_db_005",
                "category": "research"
            },
            {
                "text": "Counterfactual explanations identify minimal input changes needed to alter model predictions.",
                "source": "research_db_006",
                "category": "research"
            },
        ]
        
        return medical_docs + legal_docs + research_docs
    
    def get_all_documents(self) -> List[Dict]:
        return self.documents
    
    def get_by_category(self, category: str) -> List[Dict]:
        return [doc for doc in self.documents if doc["category"] == category]