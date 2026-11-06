
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ---------------- CONFIGURATION ----------------

EMBEDDING_FILE = "models/embeddings.npz"
MODEL_DIR = "models"

CLASSIFIER_FILE = os.path.join(
    MODEL_DIR,
    "svm_classifier.pkl"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

# ------------------------------------------------


def train_classifier():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading embeddings...")

    data = np.load(EMBEDDING_FILE)

    X = data["embeddings"]
    y = data["labels"]

    print("\nDataset information:")
    print("Embeddings:", X.shape)
    print("Labels:", y.shape)

    # Identify registered students
    students, counts = np.unique(y, return_counts=True)

    print("\nRegistered students:")

    for student, count in zip(students, counts):
        print(f"{student}: {count} images")

    # Check number of classes
    if len(students) < 2:
        print("\nError: At least two students are required.")
        return

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nDataset split:")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Initialize SVM
    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr"
    )

    print("\nTraining SVM classifier...")

    clf.fit(X_train, y_train)

    print("Training completed!")

    # Predict test samples
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== MODEL EVALUATION ==========")

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=students,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=students
    )

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=students
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("ArcFace + SVM Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(MODEL_DIR, "confusion_matrix.png")
    )

    plt.show()

    # Save trained classifier
    joblib.dump(
        clf,
        CLASSIFIER_FILE
    )

    print("\nClassifier saved successfully!")
    print("Location:", CLASSIFIER_FILE)


if __name__ == "__main__":
    train_classifier()
