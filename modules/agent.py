# ============================================================
# MODULE 1: Intelligent Agent — Healthcare Diagnostic Agent
# Covers: Week 2 (Intelligent Agents) + PEAS Framework
# ============================================================

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime


# ============================================================
# AGENT STATES
# ============================================================

class AgentState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting_symptoms"
    DIAGNOSING = "diagnosing"
    RECOMMENDING = "recommending"
    PLANNING = "planning_treatment"
    DONE = "done"


# ============================================================
# PATIENT PERCEPT
# ============================================================

@dataclass
class PatientPercept:
    """What the agent perceives from the environment."""

    patient_id: str
    symptoms: List[str]
    age: int
    temperature: float
    heart_rate: int
    blood_pressure: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )


# ============================================================
# AGENT MEMORY
# ============================================================

@dataclass
class AgentMemory:
    """Internal model — makes this a model-based agent."""

    patient_history: List[Dict] = field(default_factory=list)
    current_patient: Optional[PatientPercept] = None
    diagnosis_history: List[Dict] = field(default_factory=list)
    action_log: List[str] = field(default_factory=list)


# ============================================================
# HEALTHCARE DIAGNOSTIC AGENT
# ============================================================

class HealthcareDiagnosticAgent:
    """
    PEAS Definition:

    Performance:
        Diagnostic accuracy, patient safety,
        recommendation quality, response time

    Environment:
        Hospital/clinic, patient data, EMR

    Actuators:
        Diagnosis report, treatment plan,
        referral recommendation, alerts

    Sensors:
        Symptom input, vitals, lab results,
        patient history

    Agent Type:
        Model-Based + Goal-Based + Learning
    """

    def __init__(self):
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.performance_score = 0

        # Holds the AI sub-modules
        self._modules = {}

    # ========================================================
    # MODULE REGISTRATION
    # ========================================================

    def register_module(self, name: str, module):
        """Plug in AI sub-modules such as KB, Bayes, ML, etc."""

        self._modules[name] = module

        print(f"  🔌 Module registered: [{name}]")

    # ========================================================
    # PERCEIVE
    # ========================================================

    def perceive(self, percept: PatientPercept):
        """Step 1: Perceive the environment."""

        self.memory.current_patient = percept

        self.memory.patient_history.append(
            {
                "id": percept.patient_id,
                "symptoms": percept.symptoms,
                "time": percept.timestamp
            }
        )

        self.state = AgentState.COLLECTING

        self._log(
            f"Perceived patient {percept.patient_id} "
            f"with {len(percept.symptoms)} symptoms"
        )

        return self

    # ========================================================
    # THINK
    # ========================================================

    def think(self):
        """Step 2: Process and reason."""

        self.state = AgentState.DIAGNOSING

        self._log(
            "Agent thinking: running diagnostic modules..."
        )

        results = {}

        # Run every registered module that has an analyze() method
        for module_name, module in self._modules.items():

            if hasattr(module, "analyze"):

                try:
                    result = module.analyze(
                        self.memory.current_patient
                    )

                    # Make sure result is a dictionary
                    if not isinstance(result, dict):
                        result = {
                            "result": result,
                            "summary": "Module completed"
                        }

                    results[module_name] = result

                    self._log(
                        f"  [{module_name}] → "
                        f"{result.get('summary', 'done')}"
                    )

                except Exception as error:

                    # Prevent one broken module from stopping
                    # the entire diagnostic system
                    results[module_name] = {
                        "error": str(error),
                        "summary": f"Module error: {error}"
                    }

                    self._log(
                        f"  [{module_name}] → ERROR: {error}"
                    )

        self.memory.diagnosis_history.append(results)

        self.state = AgentState.RECOMMENDING

        return results

    # ========================================================
    # ACT
    # ========================================================

    def act(self, diagnosis_results: Dict) -> Dict:
        """Step 3: Generate action and recommendation."""

        self.state = AgentState.PLANNING

        patient = self.memory.current_patient

        # ----------------------------------------------------
        # Collect confidence values from diagnostic modules
        # ----------------------------------------------------

        confidences = []

        for value in diagnosis_results.values():

            if (
                isinstance(value, dict)
                and "confidence" in value
            ):

                try:
                    confidence = float(
                        value["confidence"]
                    )

                    confidences.append(confidence)

                except (TypeError, ValueError):
                    pass

        # Calculate average confidence
        if confidences:
            avg_confidence = (
                sum(confidences) / len(confidences)
            )
        else:
            avg_confidence = 0.5

        # ----------------------------------------------------
        # Determine urgency
        # ----------------------------------------------------

        urgency = self._assess_urgency(
            patient,
            avg_confidence
        )

        # ----------------------------------------------------
        # Generate final report
        # ----------------------------------------------------

        action_report = {
            "patient_id": patient.patient_id,

            "timestamp": patient.timestamp,

            "symptoms": patient.symptoms,

            "diagnosis": self._aggregate_diagnosis(
                diagnosis_results
            ),

            "confidence": round(
                avg_confidence,
                3
            ),

            "urgency": urgency,

            "recommendations":
                self._generate_recommendations(
                    urgency,
                    diagnosis_results
                ),

            "next_action":
                self._decide_next_action(
                    urgency
                )
        }

        # ----------------------------------------------------
        # Update performance score
        # ----------------------------------------------------

        if avg_confidence > 0.7:
            self.performance_score += 10
        else:
            self.performance_score += 5

        self.state = AgentState.DONE

        self._log(
            f"Action generated: {urgency} urgency"
        )

        return action_report

    # ========================================================
    # COMPLETE AGENT CYCLE
    # ========================================================

    def run(self, percept: PatientPercept) -> Dict:
        """Full agent cycle: Perceive → Think → Act."""

        self.perceive(percept)

        results = self.think()

        return self.act(results)

    # ========================================================
    # URGENCY ASSESSMENT
    # ========================================================

    def _assess_urgency(
        self,
        patient,
        confidence
    ):

        if (
            patient.temperature > 39.5
            or patient.heart_rate > 120
        ):
            return "CRITICAL"

        elif (
            patient.temperature > 38.5
            or confidence > 0.8
        ):
            return "HIGH"

        elif patient.temperature > 37.5:
            return "MEDIUM"

        return "LOW"

    # ========================================================
    # DIAGNOSIS AGGREGATION
    # ========================================================

    def _aggregate_diagnosis(self, results):

        diagnoses = []

        for value in results.values():

            if (
                isinstance(value, dict)
                and "diagnosis" in value
            ):

                diagnosis = value.get("diagnosis")

                if diagnosis:
                    diagnoses.append(diagnosis)

        if not diagnoses:
            return "Insufficient data"

        from collections import Counter

        return Counter(
            diagnoses
        ).most_common(1)[0][0]

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        urgency,
        results
    ):

        base = {

            "CRITICAL": [
                "🚨 Immediate emergency consultation required",
                "📞 Alert attending physician now",
                "🏥 Transfer to emergency ward",
                "💊 Administer first-line medications"
            ],

            "HIGH": [
                "⚠️ Schedule urgent appointment within 24 hours",
                "🧪 Order blood panel and cultures",
                "💊 Prescribe symptomatic relief",
                "📋 Monitor vitals every 2 hours"
            ],

            "MEDIUM": [
                "📅 Schedule appointment within 3 days",
                "💊 Over-the-counter treatment advised",
                "🌡️ Monitor temperature twice daily",
                "💧 Increase fluid intake"
            ],

            "LOW": [
                "🏠 Home rest recommended",
                "💧 Stay hydrated",
                "📱 Follow up if symptoms worsen",
                "📋 General wellness monitoring"
            ]
        }

        return base.get(
            urgency,
            base["LOW"]
        )

    # ========================================================
    # NEXT ACTION
    # ========================================================

    def _decide_next_action(self, urgency):

        actions = {

            "CRITICAL":
                "EMERGENCY_REFERRAL",

            "HIGH":
                "URGENT_APPOINTMENT",

            "MEDIUM":
                "SCHEDULE_FOLLOWUP",

            "LOW":
                "MONITOR_AT_HOME"
        }

        return actions.get(
            urgency,
            "MONITOR_AT_HOME"
        )

    # ========================================================
    # ACTION LOG
    # ========================================================

    def _log(self, message):

        entry = (
            f"[{self.state.value}] "
            f"{message}"
        )

        self.memory.action_log.append(entry)

    # ========================================================
    # PRINT LOG
    # ========================================================

    def print_log(self):

        print("\n📋 Agent Action Log:")
        print("─" * 50)

        for entry in self.memory.action_log:

            print(f"  {entry}")

    # ========================================================
    # PERFORMANCE
    # ========================================================

    def get_performance(self):

        return {

            "total_patients":
                len(self.memory.patient_history),

            "performance_score":
                self.performance_score,

            "diagnoses_made":
                len(self.memory.diagnosis_history)
        }