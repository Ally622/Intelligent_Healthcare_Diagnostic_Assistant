# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  

# Import all modules  
from modules import agent
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules...")  
    modules = {  
        'KnowledgeBase': MedicalKnowledgeBase(),  
        'BayesianNet':   SimpleBayesianDiagnostics(),  
        'MLClassifier':  MLDiagnosticClassifier(),  
        'NeuralNetwork': NeuralDiagnosticModel(),  
        'FuzzySeverity': FuzzySeverityAssessor(),
    }
      # Register each module with the agent
    for name, module in modules.items():
        agent.register_module(name, module)
        print(f"✓ {name} loaded")

    print("\nAll modules successfully registered.\n")

    return agent
def main():
    banner()

    agent = build_system()

    print("System initialized successfully!")
    print("\n===== PATIENT INFORMATION =====")

    patient_id = input("Patient ID: ")
    age = int(input("Age: "))
    temperature = float(input("Temperature (°C): "))
    heart_rate = int(input("Heart Rate: "))
    blood_pressure = input("Blood Pressure: ")

    symptoms = input(
        "Symptoms (comma separated): "
    ).split(",")

    symptoms = [s.strip().lower() for s in symptoms]
    patient = PatientPercept(
        patient_id=patient_id,
        symptoms=symptoms,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure
    )
    report = agent.run(patient)
    print("\n===== DIAGNOSIS REPORT =====")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\n===== MODULE RESULTS =====")

    module_results = agent.memory.diagnosis_history[-1]

    for module_name, result in module_results.items():
        print(f"\n--- {module_name} ---")

        if isinstance(result, dict):
            for k, v in result.items():
                print(f"{k}: {v}")

if __name__ == "__main__":
    main()