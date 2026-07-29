from modules.knowledge_base import MedicalKnowledgeBase


def analyze(symptoms):
    """
    Test Knowledge-Based AI analysis.
    Demonstrates facts, inference engine,
    forward chaining and backward chaining.
    """

    kb = MedicalKnowledgeBase()

    # Load patient symptoms as facts
    kb.load_patient_symptoms(symptoms)

    print("\n--- PATIENT FACTS ---")
    print(symptoms)

    print("\n--- FORWARD CHAINING RESULTS ---")
    results = kb.forward_chain(verbose=True)
    print(results)

    print("\n--- BACKWARD CHAINING TEST ---")
    goal = "covid19_suspected"

    proved, confidence = kb.backward_chain(goal)

    print("Goal:", goal)
    print("Proved:", proved)
    print("Confidence:", confidence)

    return results


if __name__ == "__main__":

    patient_symptoms = [
        "fever",
        "cough",
        "fatigue",
        "loss_of_smell"
    ]

    analyze(patient_symptoms)