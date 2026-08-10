import streamlit as st
import pandas as pd
import os

# ==========================================================
# IMPORT AI MODULES
# ==========================================================

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net import SimpleBayesianDiagnostics
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner import TreatmentPlanner


# ==========================================================
# PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="Healthcare Diagnostic Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Intelligent Healthcare Diagnostic Assistant")
st.write("Hello! Please enter the patient's information and select the relevant symptoms to get started.")


# ==========================================================
# LOAD SYMPTOMS
# ==========================================================

@st.cache_data
def load_symptoms():

    symptom_set = set()

    # ------------------------------------------------------
    # Possible CSV locations
    # ------------------------------------------------------

    files = [
        "data/symptoms.csv",
        "data/Training.csv",
        "data/training.csv",
        "Training.csv",
        "training.csv"
    ]

    for file in files:

        if not os.path.exists(file):
            continue

        try:

            df = pd.read_csv(file)

            # ------------------------------------------------
            # Case 1: Training dataset
            # Symptoms are normally column names
            # ------------------------------------------------

            for column in df.columns:

                column_clean = str(column).strip().lower()

                if column_clean not in [
                    "prognosis",
                    "disease",
                    "diagnosis"
                ]:

                    # Ignore completely empty columns
                    if not df[column].isna().all():
                        symptom_set.add(str(column).strip())

            # ------------------------------------------------
            # Case 2: symptoms.csv containing symptom values
            # ------------------------------------------------

            for column in df.columns:

                column_clean = str(column).strip().lower()

                if (
                    "symptom" in column_clean
                    or "name" in column_clean
                ):

                    for value in df[column].dropna():

                        value = str(value).strip()

                        if value:
                            symptom_set.add(value)

        except Exception:
            continue

    return sorted(symptom_set)


all_symptoms = load_symptoms()


# ==========================================================
# BUILD AI AGENT
# ==========================================================

@st.cache_resource
def build_agent():

    agent = HealthcareDiagnosticAgent()

    agent.register_module(
        "KnowledgeBase",
        MedicalKnowledgeBase()
    )

    agent.register_module(
        "BayesianNet",
        SimpleBayesianDiagnostics()
    )

    agent.register_module(
        "MLClassifier",
        MLDiagnosticClassifier()
    )

    agent.register_module(
        "NeuralNetwork",
        NeuralDiagnosticModel()
    )

    agent.register_module(
        "FuzzySeverity",
        FuzzySeverityAssessor()
    )

    agent.register_module(
        "TreatmentPlanner",
        TreatmentPlanner()
    )

    return agent


agent = build_agent()


# ==========================================================
# PATIENT INFORMATION
# ==========================================================

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    patient_id = st.text_input(
        "Patient ID",
        value="p001"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=30.0,
        max_value=45.0,
        value=37.0,
        step=0.1
    )


with col2:

    heart_rate = st.number_input(
        "Heart Rate",
        min_value=30,
        max_value=220,
        value=80
    )

    blood_pressure = st.text_input(
        "Blood Pressure",
        value="120/80"
    )


# ==========================================================
# SYMPTOMS
# ==========================================================

st.header("Symptoms")

selected = st.multiselect(
    "🔎 Search and select symptoms",
    options=all_symptoms,
    placeholder="Type to search symptoms..."
)

# ==========================================================
# SHOW SELECTED SYMPTOMS
# ==========================================================

if selected:

    st.write("### Selected Symptoms")

    for symptom in selected:
        st.write(
            f"✅ {symptom.replace('_', ' ').title()}"
        )

# ==========================================================
# RUN DIAGNOSIS
# ==========================================================

if st.button(
    "🚀 Run Diagnosis",
    type="primary"
):

    if not patient_id.strip():

        st.warning("Please enter a Patient ID.")

    elif not selected:

        st.warning(
            "Please select at least one symptom."
        )

    else:

        # --------------------------------------------------
        # Create Patient Percept
        # --------------------------------------------------

        patient = PatientPercept(

            patient_id=patient_id,

            symptoms=selected,

            age=age,

            temperature=temperature,

            heart_rate=heart_rate,

            blood_pressure=blood_pressure
        )

        # --------------------------------------------------
        # Run COMPLETE AI SYSTEM
        # --------------------------------------------------

        with st.spinner(
            "Running Knowledge Base, Bayesian, ML, Neural Network and Fuzzy analysis..."
        ):

            report = agent.run(patient)


        # ==================================================
        # DIAGNOSIS REPORT
        # ==================================================

        st.success("Diagnosis Complete")

        st.header("🏥 Diagnosis Report")


        # --------------------------------------------------
        # Extract report information
        # --------------------------------------------------

        diagnosis = report.get(
            "diagnosis",
            "Unknown"
        )

        confidence = report.get(
            "confidence",
            0
        )

        urgency = report.get(
            "urgency",
            "UNKNOWN"
        )

        recommendations = report.get(
            "recommendations",
            []
        )

        next_action = report.get(
            "next_action",
            "No action specified"
        )


        # ==================================================
        # MAIN RESULTS
        # ==================================================

        result_col1, result_col2, result_col3 = st.columns(3)


        with result_col1:

            st.metric(
                "Diagnosis",
                str(diagnosis).replace("_", " ").title()
            )


        with result_col2:

            st.metric(
                "Confidence",
                f"{float(confidence) * 100:.1f}%"
            )


        with result_col3:

            if str(urgency).upper() == "CRITICAL":

                st.error(
                    f"🚨 {urgency}"
                )

            elif str(urgency).upper() == "HIGH":

                st.warning(
                    f"⚠️ {urgency}"
                )

            else:

                st.info(
                    f"ℹ️ {urgency}"
                )


        # ==================================================
        # URGENCY / CRITICAL STATUS
        # ==================================================

        st.subheader("🚨 Patient Status")


        if str(urgency).upper() == "CRITICAL":

            st.error(
                "🚨 CRITICAL: Immediate medical attention is required."
            )

        elif str(urgency).upper() == "HIGH":

            st.warning(
                "⚠️ HIGH RISK: Prompt medical attention is recommended."
            )

        elif str(urgency).upper() == "MODERATE":

            st.warning(
                "⚠️ MODERATE: Medical evaluation is recommended."
            )

        else:

            st.success(
                "✅ LOW RISK based on the current assessment."
            )


        # ==================================================
        # RECOMMENDATIONS
        # ==================================================

        st.subheader("💊 Recommendations")


        if recommendations:

            for recommendation in recommendations:

                st.write(
                    f"• {recommendation}"
                )

        else:

            st.write(
                "No recommendations were returned."
            )


        # ==================================================
        # NEXT ACTION
        # ==================================================

        st.subheader("➡️ Recommended Next Action")

        st.info(
            str(next_action).replace(
                "_",
                " "
            )
        )


        # ==================================================
        # PATIENT SUMMARY
        # ==================================================

        st.subheader("📋 Patient Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        with summary_col1:
            st.write(f"**Patient ID:** {patient_id}")

        with summary_col2:
            st.write(f"**Age:** {age}")

        with summary_col3:
            st.write(f"**Temperature:** {temperature} °C")

        with summary_col4:
            st.write(f"**Heart Rate:** {heart_rate} bpm")

        st.write(
            f"**Blood Pressure:** {blood_pressure}"
        )

        st.write(
            "**Symptoms:** "
            + ", ".join(
                symptom.replace("_", " ")
                for symptom in selected
            )
        )


        # ==================================================
        # INDIVIDUAL MODULE RESULTS
        # ==================================================

        st.header("🧠 Individual AI Module Results")


        # Get latest results from agent memory
        try:

            results = agent.memory.diagnosis_history[-1]

        except Exception:

            results = {}


        for module, output in results.items():

            with st.expander(
                f"🔹 {module}"
            ):

                if isinstance(output, dict):

                    st.json(output)

                else:

                    st.write(output)


        # ==================================================
        # COMPLETE RAW REPORT
        # ==================================================

        with st.expander(
            "📄 Complete Diagnosis Report"
        ):

            st.json(report)