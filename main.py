import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_processing import data_processing
from SVM import SVM


def main():
    # Extracting features and targets from the data set
    X, y = data_processing()

    # Converting to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy()

    # K-Fold Cross-Validation 
    # Initializing the model
    model = SVM()

    # Defining a K-Fold cross-validator
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Performing cross-validation and getting accuracy for each fold
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print("Cross-validation accuracies for each fold:", scores)

    # Printing the average accuracy across all folds
    print(f"Average cross-validation accuracy: {scores.mean():.4f}")

    # Final evaluation on a separate test set
    # Separating the training and test sets
    X_train, X_test = X[:380], X[380:]
    y_train, y_test = y[:380], y[380:]

    # Initializing the model
    final_model = SVM()

    # Training the model on the training sets
    final_model.fit(X_train, y_train)

    # Making and printing the predictions
    predictions = final_model.predict(X_test)
    label_predictions = final_model.labels(predictions)
    print("Predictions:", label_predictions)

    # Calculating and printing the accuracy on the test set
    accuracy = np.mean(predictions == y_test)
    print(f"Test set accuracy: {accuracy:.4f}")

    # Calculating and printing the average Hinge loss
    hinge_loss_value = final_model._hinge_loss(
        X_test, y_test
    )
    print(f"Average Hinge loss: {hinge_loss_value:.4f}")

if __name__ == "__main__":
    main()
