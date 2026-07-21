# ============================================================
# MODULE 3: Bayesian Network — Probabilistic Diagnosis
# Team Member 3
#
# AI Concepts:
# - Bayesian Networks
# - Naïve Bayes
# - Conditional Probability
# - Posterior Probability
# - Probabilistic Reasoning
#
# Purpose:
# This module estimates the probability of diseases based on
# observed patient symptoms using the Naïve Bayes algorithm.
# ============================================================

import numpy as np
from typing import Dict, List


class SimpleBayesianDiagnostics:
    """
    Bayesian diagnostic model using pre-defined
    prior probabilities and likelihood tables.
    """

    def __init__(self):

        # Small probability used for unknown symptoms
        # Prevents log(0) errors.
        self.min_probability = 0.01

        # ---------------------------------------------------
        # Prior Probabilities P(Disease)
        # Represents how common each disease is before
        # observing any symptoms.
        # ---------------------------------------------------

        self.priors = {
            "flu": 0.15,
            "covid19": 0.08,
            "dengue": 0.05,
            "cardiac": 0.04,
            "diabetes": 0.10,
            "common_cold": 0.30,
            "healthy": 0.28,
        }

        # ---------------------------------------------------
        # Likelihood Table
        # P(Symptom | Disease)
        # ---------------------------------------------------

        self.likelihoods = {

            "flu": {
                "fever": 0.90,
                "cough": 0.85,
                "fatigue": 0.88,
                "headache": 0.70,
                "body_aches": 0.80,
                "loss_of_smell": 0.20,
                "chest_pain": 0.05,
                "rash": 0.05,
                "joint_pain": 0.40,
            },

            "covid19": {
                "fever": 0.88,
                "cough": 0.80,
                "fatigue": 0.90,
                "loss_of_smell": 0.85,
                "headache": 0.65,
                "body_aches": 0.60,
                "chest_pain": 0.20,
                "rash": 0.05,
                "joint_pain": 0.20,
            },

            "dengue": {
                "fever": 0.98,
                "rash": 0.75,
                "joint_pain": 0.85,
                "headache": 0.90,
                "fatigue": 0.80,
                "cough": 0.15,
                "loss_of_smell": 0.05,
                "chest_pain": 0.05,
                "body_aches": 0.88,
            },

            "cardiac": {
                "chest_pain": 0.92,
                "shortness_of_breath": 0.88,
                "fatigue": 0.70,
                "sweating": 0.75,
                "fever": 0.10,
                "cough": 0.15,
                "rash": 0.02,
                "joint_pain": 0.10,
                "headache": 0.30,
            },

            "diabetes": {
                "fatigue": 0.82,
                "frequent_urination": 0.95,
                "excessive_thirst": 0.92,
                "blurred_vision": 0.70,
                "fever": 0.10,
                "cough": 0.05,
                "rash": 0.08,
                "headache": 0.40,
                "joint_pain": 0.20,
            },

            "common_cold": {
                "cough": 0.90,
                "fever": 0.50,
                "headache": 0.60,
                "fatigue": 0.55,
                "body_aches": 0.50,
                "loss_of_smell": 0.30,
                "rash": 0.02,
                "chest_pain": 0.05,
                "joint_pain": 0.15,
            },

            "healthy": {
                "fever": 0.02,
                "cough": 0.05,
                "fatigue": 0.10,
                "headache": 0.08,
                "rash": 0.01,
                "chest_pain": 0.01,
                "joint_pain": 0.05,
                "loss_of_smell": 0.01,
                "body_aches": 0.05,
            }

        }

    # ---------------------------------------------------
    # Helper Function
    # ---------------------------------------------------

    def clean_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Standardizes symptom names.
        Example:
        'Loss of Smell' -> 'loss_of_smell'
        """
        return [
            symptom.lower().strip().replace(" ", "_")
            for symptom in symptoms
        ]

    # ---------------------------------------------------
    # Posterior Probability Computation
    # ---------------------------------------------------

    def compute_posterior(
        self,
        symptoms: List[str]
    ) -> Dict[str, float]:
        """
        Computes posterior probabilities using
        Naïve Bayes.

        P(Disease | Symptoms)
        ∝
        P(Disease) × Π P(Symptom | Disease)
        """

        # If no symptoms are provided,
        # simply return the prior probabilities.
        if not symptoms:
            return self.priors.copy()

        symptoms = self.clean_symptoms(symptoms)

        log_scores = {}

        # Calculate log probability for each disease
        for disease, prior in self.priors.items():

            log_probability = np.log(prior)

            for symptom in symptoms:

                likelihood = self.likelihoods[disease].get(
                    symptom,
                    self.min_probability
                )

                log_probability += np.log(likelihood)

            log_scores[disease] = log_probability

        # Convert log scores back to probabilities
        max_log = max(log_scores.values())

        probabilities = {
            disease: np.exp(score - max_log)
            for disease, score in log_scores.items()
        }

        total = sum(probabilities.values())

        posterior = {
            disease: round(prob / total, 4)
            for disease, prob in probabilities.items()
        }

        return posterior

    # ---------------------------------------------------
    # Main Interface
    # ---------------------------------------------------

    def analyze(self, percept) -> Dict:
        """
        Receives a patient percept from the agent
        and returns the most probable diagnosis.
        """

        posterior = self.compute_posterior(
            percept.symptoms
        )

        ranked = sorted(
            posterior.items(),
            key=lambda item: item[1],
            reverse=True
        )

        diagnosis = ranked[0][0]
        confidence = ranked[0][1]

        return {

            "summary":
                f"Most probable diagnosis: {diagnosis} ({confidence:.2%})",

            "diagnosis":
                diagnosis,

            "confidence":
                confidence,

            "all_posteriors":
                posterior,

            "ranked_diagnoses":
                ranked[:5]

        }

    # ---------------------------------------------------
    # Explanation Function
    # ---------------------------------------------------

    def explain(
        self,
        disease: str,
        symptoms: List[str]
    ) -> str:
        """
        Explains how the probability
        for a disease was calculated.
        """

        symptoms = self.clean_symptoms(symptoms)

        lines = []

        lines.append("Bayesian Diagnostic Explanation")
        lines.append("----------------------------------")
        lines.append(f"Disease: {disease}")
        lines.append("")
        lines.append(
            f"Prior Probability: P({disease}) = {self.priors[disease]}"
        )
        lines.append("")
        lines.append("Evidence:")

        for symptom in symptoms:

            probability = self.likelihoods[disease].get(
                symptom,
                self.min_probability
            )

            lines.append(
                f"  P({symptom}|{disease}) = {probability:.2f}"
            )

        return "\n".join(lines)


# ---------------------------------------------------
# Standalone Testing
# ---------------------------------------------------

if __name__ == "__main__":

    bayes = SimpleBayesianDiagnostics()

    patient_symptoms = [
        "fever",
        "cough",
        "fatigue",
        "loss of smell"
    ]

    results = bayes.compute_posterior(patient_symptoms)

    print("\nPosterior Probabilities\n")

    for disease, probability in sorted(
        results.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{disease:<20} {probability:.2%}")

    print("\n")

    print(
        bayes.explain(
            "covid19",
            patient_symptoms
        )
    )