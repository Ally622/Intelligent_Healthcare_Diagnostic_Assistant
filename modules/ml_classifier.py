# ============================================================
# MODULE 4: ML Classifier — Supervised Diagnosis
# Team Member 3
#
# AI Concepts:
# - Supervised Learning
# - Decision Trees
# - Random Forest
# - Gradient Boosting
# - Classification
#
# Purpose:
# Uses machine learning algorithms to predict diseases
# from patient symptoms and selects the best-performing
# model automatically.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


class MLDiagnosticClassifier:
    """
    Machine Learning Diagnostic Module.

    Trains multiple supervised learning models
    and automatically selects the best-performing
    classifier.
    """

    # ---------------------------------------------------
    # Symptoms used as model features
    # ---------------------------------------------------

    SYMPTOM_FEATURES = [

        "fever",
        "cough",
        "fatigue",
        "headache",
        "body_aches",
        "loss_of_smell",
        "chest_pain",
        "rash",
        "joint_pain",
        "shortness_of_breath",
        "sweating",
        "frequent_urination",
        "excessive_thirst",
        "blurred_vision",
        "night_sweats",
        "weight_loss",
        "stiff_neck",
        "light_sensitivity"

    ]

    # ---------------------------------------------------
    # Disease Labels
    # ---------------------------------------------------

    DISEASE_LABELS = [

        "flu",
        "covid19",
        "dengue",
        "cardiac_event",
        "diabetes",
        "common_cold",
        "tuberculosis",
        "meningitis"

    ]

    # ---------------------------------------------------

    def __init__(self):

        self.models = {

            "Decision Tree":

                DecisionTreeClassifier(
                    criterion="entropy",
                    max_depth=8,
                    random_state=42
                ),

            "Random Forest":

                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),

            "Gradient Boosting":

                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                )

        }

        self.best_model = None
        self.best_model_name = None

        self.label_encoder = LabelEncoder()

        self.is_trained = False

    # ---------------------------------------------------
    # Generate Synthetic Dataset
    # ---------------------------------------------------

    def generate_synthetic_data(
        self,
        samples: int = 2000
    ) -> pd.DataFrame:

        np.random.seed(42)

        disease_profiles = {

            "flu": {
                "fever": 0.90,
                "cough": 0.85,
                "fatigue": 0.88,
                "headache": 0.70,
                "body_aches": 0.80,
                "loss_of_smell": 0.20
            },

            "covid19": {
                "fever": 0.88,
                "cough": 0.80,
                "fatigue": 0.90,
                "loss_of_smell": 0.85,
                "headache": 0.65,
                "body_aches": 0.60
            },

            "dengue": {
                "fever": 0.98,
                "rash": 0.75,
                "joint_pain": 0.85,
                "headache": 0.90,
                "fatigue": 0.80,
                "body_aches": 0.88
            },

            "cardiac_event": {
                "chest_pain": 0.92,
                "shortness_of_breath": 0.88,
                "fatigue": 0.70,
                "sweating": 0.75
            },

            "diabetes": {
                "fatigue": 0.82,
                "frequent_urination": 0.95,
                "excessive_thirst": 0.92,
                "blurred_vision": 0.70,
                "weight_loss": 0.50
            },

            "common_cold": {
                "cough": 0.90,
                "fever": 0.50,
                "headache": 0.60,
                "fatigue": 0.55,
                "body_aches": 0.50
            },

            "tuberculosis": {
                "cough": 0.95,
                "weight_loss": 0.85,
                "night_sweats": 0.80,
                "fatigue": 0.88,
                "fever": 0.70
            },

            "meningitis": {
                "headache": 0.95,
                "stiff_neck": 0.90,
                "fever": 0.92,
                "light_sensitivity": 0.85,
                "fatigue": 0.80
            }

        }

        records = []

        samples_per_disease = samples // len(disease_profiles)

        for disease, profile in disease_profiles.items():

            for _ in range(samples_per_disease):

                patient = {
                    feature: 0
                    for feature in self.SYMPTOM_FEATURES
                }

                for symptom, probability in profile.items():

                    patient[symptom] = int(
                        np.random.rand() < probability
                    )

                # Add 5% random noise
                for feature in self.SYMPTOM_FEATURES:

                    if (
                        patient[feature] == 0
                        and np.random.rand() < 0.05
                    ):
                        patient[feature] = 1

                patient["disease"] = disease

                records.append(patient)

        dataframe = pd.DataFrame(records)

        dataframe = dataframe.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        return dataframe

    # ---------------------------------------------------
    # Train Models
    # ---------------------------------------------------

    def train(
        self,
        verbose: bool = True
    ) -> Dict:

        dataset = self.generate_synthetic_data()

        X = dataset[self.SYMPTOM_FEATURES]

        y = self.label_encoder.fit_transform(
            dataset["disease"]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        self.X_test = X_test
        self.y_test = y_test

        results = {}

        best_accuracy = 0
        if verbose:
            print("=" * 60)
            print("Training Machine Learning Models")
            print("=" * 60)

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="accuracy"
            )

            results[name] = {

                "accuracy": accuracy,

                "cv_mean": cv_scores.mean(),

                "cv_std": cv_scores.std()

            }

            if verbose:

                print(f"\n{name}")

                print(f"Test Accuracy : {accuracy:.4f}")

                print(
                    f"Cross Validation : "
                    f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )

            if accuracy > best_accuracy:

                best_accuracy = accuracy

                self.best_model = model

                self.best_model_name = name

        self.is_trained = True

        if verbose:

            print("\nBest Model")

            print(f"{self.best_model_name}")

            print(f"Accuracy : {best_accuracy:.4f}")

        return results

    # ---------------------------------------------------
    # Predict Disease
    # ---------------------------------------------------

    def predict(
        self,
        symptoms: List[str]
    ) -> Dict:

        if not self.is_trained:

            self.train(verbose=False)

        symptoms = [
            symptom.lower().replace(" ", "_")
            for symptom in symptoms
        ]

        feature_vector = np.array([

            [

                1 if feature in symptoms else 0

                for feature in self.SYMPTOM_FEATURES

            ]

        ])

        prediction = self.best_model.predict(
            feature_vector
        )[0]

        probabilities = self.best_model.predict_proba(
            feature_vector
        )[0]

        disease = self.label_encoder.inverse_transform(
            [prediction]
        )[0]

        classes = self.label_encoder.inverse_transform(

            np.arange(len(probabilities))

        )

        ranked = sorted(

            zip(classes, probabilities),

            key=lambda item: item[1],

            reverse=True

        )

        return {

            "diagnosis": disease,

            "confidence": round(

                float(probabilities[prediction]),

                4

            ),

            "top_predictions": ranked[:5],

            "model_used": self.best_model_name,

            "feature_vector": feature_vector.tolist()[0]

        }

    # ---------------------------------------------------
    # Agent Interface
    # ---------------------------------------------------

    def analyze(
        self,
        percept
    ) -> Dict:

        result = self.predict(

            percept.symptoms

        )

        result["summary"] = (

            f"{result['model_used']} predicts "

            f"{result['diagnosis']} "

            f"({result['confidence']:.2%})"

        )

        return result

    # ---------------------------------------------------
    # Plot Evaluation
    # ---------------------------------------------------

    def plot_evaluation(self):

        if not self.is_trained:

            self.train(verbose=False)

        predictions = self.best_model.predict(

            self.X_test

        )

        print("\nClassification Report\n")

        print(

            classification_report(

                self.y_test,

                predictions,

                target_names=self.label_encoder.classes_

            )

        )

        confusion = confusion_matrix(

            self.y_test,

            predictions

        )

        plt.figure(figsize=(8, 7))

        display = ConfusionMatrixDisplay(

            confusion_matrix=confusion,

            display_labels=self.label_encoder.classes_

        )

        display.plot(

            cmap="Blues",

            values_format="d"

        )

        plt.title(

            f"Confusion Matrix ({self.best_model_name})"

        )

        plt.tight_layout()

        plt.savefig(

            "confusion_matrix.png",

            dpi=300

        )

        if hasattr(

            self.best_model,

            "feature_importances_"

        ):

            importance = self.best_model.feature_importances_

            indices = np.argsort(

                importance

            )[::-1]

            plt.figure(figsize=(10, 6))

            plt.bar(

                range(len(self.SYMPTOM_FEATURES)),

                importance[indices]

            )

            plt.xticks(

                range(len(self.SYMPTOM_FEATURES)),

                np.array(

                    self.SYMPTOM_FEATURES

                )[indices],

                rotation=90

            )

            plt.title(

                f"Feature Importance ({self.best_model_name})"

            )

            plt.tight_layout()

            plt.savefig(

                "feature_importance.png",

                dpi=300

            )

        plt.show()


# ---------------------------------------------------
# Standalone Testing
# ---------------------------------------------------

if __name__ == "__main__":

    classifier = MLDiagnosticClassifier()

    classifier.train()

    prediction = classifier.predict(

        [

            "fever",

            "cough",

            "fatigue",

            "loss of smell"

        ]

    )

    print("\nPrediction Result\n")

    print(prediction)

    classifier.plot_evaluation()