import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_processing import data_processing
from SVM import SVM


def main():
    # ------------ Preprocessing ----------

    # Extracting features and targets from the data set and converting to
    # numpy arrays
    X, y = data_processing()

    # ------- K-Fold Cross-Validation ----------

    # Initializing the model
    model = SVM()

    # Defining a K-Fold cross-validator
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Performing cross-validation and getting accuracy for each fold
    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    print("Cross-validation accuracies for each fold:", scores)

    # Printing the average accuracy across all folds
    print(f"Average cross-validation accuracy: {scores.mean():.4f}")

    # ------------ Initializing the model -----------

    # Final evaluation on a separate test set
    # Separating the training and test sets
    X_train, X_test = X[:380], X[380:]
    y_train, y_test = y[:380], y[380:]

    final_model = SVM()

    # Training the model on the training sets
    final_model.fit(X_train, y_train)

    # Making predictions on the training set
    predictions_train = final_model.predict(X_train)

    # Calculating and printing the accuracy on the train set
    accuracy = np.mean(predictions_train == y_train)
    print(f"Train set accuracy: {accuracy:.4f}")

    # Calculating and printing the average Hinge loss of the train set
    hinge_loss_value = final_model._hinge_loss(X_train, y_train)
    print(f"Average Hinge loss for the train set: {hinge_loss_value:.4f}")

    # Making and printing the predictions on the test set
    predictions_test = final_model.predict(X_test)
    label_predictions = final_model.labels(predictions_test)
    print("Predictions:", label_predictions)

    # Calculating and printing the accuracy on the test set
    accuracy = np.mean(predictions_test == y_test)
    print(f"Test set accuracy: {accuracy:.4f}")

    # Calculating and printing the average Hinge lossof the test set
    hinge_loss_value = final_model._hinge_loss(X_test, y_test)
    print(f"Average Hinge loss for the test set: {hinge_loss_value:.4f}")


if __name__ == "__main__":
    main()
