# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import warnings
import json

warnings.filterwarnings("ignore")


# ============================================================
# IMPORT MODULES
# ============================================================

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner


# ============================================================
# OPTIONAL NEURAL NETWORK
# ============================================================

# TensorFlow is currently not working correctly in your
# virtual environment. Therefore, the neural network is
# optional so the rest of the system can still run.

try:
    from modules.neural_network import NeuralDiagnosticModel
    NEURAL_NETWORK_AVAILABLE = True
except Exception as e:
    NeuralDiagnosticModel = None
    NEURAL_NETWORK_AVAILABLE = False
    NEURAL_NETWORK_ERROR = str(e)


# ============================================================
# ANSI COLORS
# ============================================================

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


# ============================================================
# BANNER
# ============================================================

def banner():

    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║
║         Introduction to AI — Capstone Project            ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║
╚══════════════════════════════════════════════════════════╝
{C.END}
""")


# ============================================================
# SECTION
# ============================================================

def section(title):

    print(
        f"\n{C.BOLD}{C.YELLOW}"
        + "═" * 60
        + f"{C.END}"
    )

    print(
        f"{C.BOLD}{C.YELLOW}  {title}{C.END}"
    )

    print(
        f"{C.BOLD}{C.YELLOW}"
        + "═" * 60
        + f"{C.END}"
    )


# ============================================================
# BUILD SYSTEM
# ============================================================

def build_system():

    section("🔧 Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\nInitializing modules...")

    # --------------------------------------------------------
    # Core AI modules
    # --------------------------------------------------------

    modules = {

        "KnowledgeBase":
            MedicalKnowledgeBase(),

        "BayesianNet":
            SimpleBayesianDiagnostics(),

        "MLClassifier":
            MLDiagnosticClassifier(),

        "FuzzyController":
            FuzzySeverityAssessor(),

        "TreatmentPlanner":
            TreatmentPlanner()
    }

    # --------------------------------------------------------
    # Register core modules
    # --------------------------------------------------------

    for name, module in modules.items():

        agent.register_module(
            name,
            module
        )

        print(
            f"{C.GREEN}✓ {name} registered{C.END}"
        )

    # --------------------------------------------------------
    # Register Neural Network only if available
    # --------------------------------------------------------

    if NEURAL_NETWORK_AVAILABLE:

        neural_network = NeuralDiagnosticModel()

        agent.register_module(
            "NeuralNetwork",
            neural_network
        )

        print(
            f"{C.GREEN}✓ NeuralNetwork registered{C.END}"
        )

    else:

        print(
            f"{C.YELLOW}"
            "⚠ NeuralNetwork skipped — TensorFlow is not available."
            f"{C.END}"
        )

    print(
        f"\n{C.GREEN}"
        "All available modules registered successfully"
        f"{C.END}"
    )

    return agent


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(report):

    section("🏥 DIAGNOSTIC REPORT")

    for key, value in report.items():

        print(
            f"\n{C.BOLD}{key.upper()}:{C.END}"
        )

        if isinstance(value, dict):

            print(
                json.dumps(
                    value,
                    indent=4,
                    default=str
                )
            )

        elif isinstance(value, list):

            print(
                json.dumps(
                    value,
                    indent=4,
                    default=str
                )
            )

        else:

            print(value)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # Banner
    # --------------------------------------------------------

    banner()

    # --------------------------------------------------------
    # Build AI system
    # --------------------------------------------------------

    agent = build_system()

    # --------------------------------------------------------
    # Patient Information
    # --------------------------------------------------------

    section("👤 Patient Information")

    patient = PatientPercept(

        patient_id="P001",

        symptoms=[
            "fever",
            "cough",
            "fatigue",
            "loss_of_smell"
        ],

        age=25,

        temperature=38.7,

        heart_rate=95,

        blood_pressure="120/80"
    )

    print(
        "Patient:",
        patient.patient_id
    )

    print(
        "Symptoms:",
        patient.symptoms
    )

    print(
        "Temperature:",
        patient.temperature
    )

    print(
        "Heart Rate:",
        patient.heart_rate
    )

    print(
        "Blood Pressure:",
        patient.blood_pressure
    )

    # --------------------------------------------------------
    # Run Diagnostic Agent
    # --------------------------------------------------------

    section("🤖 Running Diagnostic Agent")

    try:

        report = agent.run(patient)

        display_results(report)

        # ----------------------------------------------------
        # Agent Log
        # ----------------------------------------------------

        if hasattr(agent, "print_log"):

            agent.print_log()

        print(
            f"\n{C.GREEN}"
            "✅ Diagnostic completed successfully"
            f"{C.END}"
        )

    except Exception as e:

        print(
            f"\n{C.RED}"
            "❌ Error while running diagnostic:"
            f"{C.END}"
        )

        print(e)

        raise


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()