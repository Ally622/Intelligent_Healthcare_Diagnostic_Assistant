from modules.knowledge_base import MedicalKnowledgeBase


# Patient input object required by analyze()
class PatientPercept:
    def __init__(self, symptoms, temperature, heart_rate):
        self.symptoms = symptoms
        self.temperature = temperature
        self.heart_rate = heart_rate


# Create knowledge base
kb = MedicalKnowledgeBase()


# Example patient symptoms
patient = PatientPercept(
    symptoms=[
        "fever",
        "cough",
        "fatigue",
        "loss_of_smell"
    ],
    temperature=38.5,
    heart_rate=95
)


# Run diagnosis analysis
result = kb.analyze(patient)


# Display results
print("\n--- ANALYSIS RESULTS ---")
print("Summary:", result["summary"])
print("Diagnosis:", result["diagnosis"])
print("Confidence:", result["confidence"])

print("\n--- ALL INFERRED RESULTS ---")
for disease, confidence in result["all_inferred"].items():
    print(f"{disease}: {confidence}")


# Explanation
print("\n--- EXPLANATION ---")
print(kb.get_explanation(result["diagnosis"]))
print("\n--- FORWARD CHAINING TEST ---")
forward_results = kb.forward_chain(verbose=True)
print(forward_results)


print("\n--- BACKWARD CHAINING TEST ---")
proved, confidence = kb.backward_chain("covid19_suspected")

print("COVID suspected:", proved)
print("Confidence:", confidence)