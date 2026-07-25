from modules.knowledge_base import MedicalKnowledgeBase

kb = MedicalKnowledgeBase()

# Add patient symptoms
symptoms = [
    "fever",
    "cough",
    "fatigue",
    "loss_of_smell"
]

kb.load_patient_symptoms(symptoms)

print("\n--- FORWARD CHAINING RESULTS ---")
results = kb.forward_chain(verbose=True)
print(results)

print("\n--- BACKWARD CHAINING TEST ---")
proved, confidence = kb.backward_chain("covid19_suspected")
print("COVID suspected:", proved)
print("Confidence:", confidence)