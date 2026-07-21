# ============================================================
# MODULE 4: Machine Learning Classifier
# Covers: Week 9 (Supervised Learning)
# Uses Kaggle Medical Dataset
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


class MLDiagnosticClassifier:
    """
    Machine Learning Diagnostic Classifier

    Algorithms:
    - Decision Tree
    - Random Forest
    - Gradient Boosting

    Uses the Kaggle medical symptom dataset.
    """

    def __init__(self):

        self.models = {

            "Decision Tree": DecisionTreeClassifier(
                criterion="entropy",
                max_depth=8,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        }

        self.label_encoder = LabelEncoder()

        self.best_model = None
        self.best_model_name = None

        self.is_trained = False

        self.symptom_features = []

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        df = pd.read_csv("data/patient_records.csv")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Replace missing values with empty strings
        df = df.fillna("")

        # Rename disease column
        df.rename(columns={"Disease": "disease"}, inplace=True)

        return df

    # ======================================================
    # Data Preprocessing
    # ======================================================

    def preprocess_data(self, df):

        symptom_columns = [
            col for col in df.columns
            if col.startswith("Symptom")
        ]

        symptoms = set()

        # Collect every unique symptom
        for col in symptom_columns:

            values = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
            )

            values = values[values != ""]

            symptoms.update(values.tolist())

        self.symptom_features = sorted(symptoms)

        processed = []

        for _, row in df.iterrows():

            patient = {feature: 0 for feature in self.symptom_features}

            for col in symptom_columns:

                symptom = str(row[col]).strip().lower()

                symptom = symptom.replace(" ", "_")

                if symptom in patient:

                    patient[symptom] = 1

            patient["disease"] = row["disease"]

            processed.append(patient)

        processed_df = pd.DataFrame(processed)

        return processed_df

    # ======================================================
    # Train Models
    # ======================================================

    def train(self, verbose=True):

        raw_df = self.load_dataset()

        df = self.preprocess_data(raw_df)

        X = df[self.symptom_features]

        y = self.label_encoder.fit_transform(df["disease"])

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        self.X_test = X_test

        self.y_test = y_test

        best_accuracy = 0

        results = {}

        if verbose:

            print("=" * 60)

            print("Training Machine Learning Models")

            print("=" * 60)

        for name, model in self.models.items():

            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)

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

            print(self.best_model_name)

            print(f"Accuracy : {best_accuracy:.4f}")

        return results
        # ======================================================
    # Predict Disease
    # ======================================================

    def predict(self, symptoms: List[str]) -> Dict:

        if not self.is_trained:
            self.train(verbose=False)

        symptoms = [
            s.strip().lower().replace(" ", "_")
            for s in symptoms
        ]

        feature_vector = np.zeros(len(self.symptom_features))

        for i, feature in enumerate(self.symptom_features):
            if feature in symptoms:
                feature_vector[i] = 1

        prediction = self.best_model.predict([feature_vector])[0]

        probabilities = self.best_model.predict_proba([feature_vector])[0]

        disease = self.label_encoder.inverse_transform([prediction])[0]

        classes = self.label_encoder.inverse_transform(
            np.arange(len(probabilities))
        )

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        return {

            "diagnosis": disease,

            "confidence": round(float(max(probabilities)), 4),

            "top_predictions": ranked[:5],

            "model_used": self.best_model_name,

            "feature_vector": feature_vector.astype(int).tolist()

        }

    # ======================================================
    # Agent Interface
    # ======================================================

    def analyze(self, percept):

        result = self.predict(percept.symptoms)

        result["summary"] = (

            f"{result['model_used']} predicts "

            f"{result['diagnosis']} "

            f"({result['confidence']:.2%})"

        )

        return result

    # ======================================================
    # Evaluation
    # ======================================================

    def plot_evaluation(self):

        if not self.is_trained:
            self.train(verbose=False)

        predictions = self.best_model.predict(self.X_test)

        print("\nClassification Report\n")

        print(

            classification_report(

                self.y_test,

                predictions,

                target_names=self.label_encoder.classes_

            )

        )

        cm = confusion_matrix(

            self.y_test,

            predictions

        )

        plt.figure(figsize=(10,8))

        plt.imshow(cm, cmap="Blues")

        plt.title("Confusion Matrix")

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.colorbar()

        ticks = np.arange(len(self.label_encoder.classes_))

        plt.xticks(

            ticks,

            self.label_encoder.classes_,

            rotation=90,

            fontsize=8

        )

        plt.yticks(

            ticks,

            self.label_encoder.classes_,

            fontsize=8

        )

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                plt.text(

                    j,

                    i,

                    cm[i, j],

                    ha="center",

                    va="center",

                    fontsize=6

                )

        plt.tight_layout()

        plt.savefig(

            "confusion_matrix.png",

            dpi=300

        )

        if hasattr(self.best_model, "feature_importances_"):

            importance = self.best_model.feature_importances_

            indices = np.argsort(importance)[::-1]

            top = 20

            plt.figure(figsize=(12,6))

            plt.bar(

                range(top),

                importance[indices[:top]]

            )

            plt.xticks(

                range(top),

                np.array(self.symptom_features)[indices[:top]],

                rotation=90

            )

            plt.title(

                f"Top {top} Important Symptoms ({self.best_model_name})"

            )

            plt.tight_layout()

            plt.savefig(

                "feature_importance.png",

                dpi=300

            )

        plt.show()


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":

    classifier = MLDiagnosticClassifier()

    classifier.train()

    print("\nPrediction Result\n")

    result = classifier.predict([

        "itching",

        "skin_rash",

        "nodal_skin_eruptions"

    ])

    print(result)

    classifier.plot_evaluation()