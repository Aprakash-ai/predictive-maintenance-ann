from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)


def evaluate_model(model, X_test, y_test):

    # evaluation model
    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"\nTest Accuracy : {test_accuracy:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    #prediction
    y_prob = model.predict(X_test, verbose=0)

    y_pred = (y_prob >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nPrediction Accuracy : {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Failure",
                "Failure"
            ]
        )
    )

    return y_pred