from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from data_processing import extracting_features_and_target, preprocessing_pipeline
from SVM import SVM


def main():
    # Extracting features and targets from the data set
    X_raw, y = extracting_features_and_target()

    # K-Fold Cross-Validation
    # Initializing the pipline and the model
    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline(n_components=0.95)),
            ("SVM_model", SVM()),
        ]
    )

    # Defining a k-fold cross-validator
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Performing cross-validation and getting accuracy for each fold
    scores = cross_val_score(pipeline, X_raw, y, cv=kf, scoring="accuracy")
    print("Cross-validation accuracies for each fold:", scores)

    # Printing the average accuracy across all folds
    print(f"Average cross-validation accuracy: {scores.mean():.4f}")

    # Final evaluation on a separate test set

    # Separating the training and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing the training data
    test_preprocessing_pipeline = preprocessing_pipeline(n_components=0.95)
    X_train = test_preprocessing_pipeline.fit_transform(X_train_raw)
    X_test = test_preprocessing_pipeline.transform(X_test_raw)

    # Printing the number of features before and after PCA
    print(
        f"Number of features before PCA: {X_train_raw.shape[1]}."
        f"Number of features after PCA: {X_train.shape[1]}"
    )

    # Initializing the SVM model
    final_model = SVM()

    # Training the model on the training sets
    final_model.fit(X_train, y_train)

    # Making predictions on the training set
    predictions_train = final_model.predict(X_train)

    # Calculating and printing the balanced accuracy on the train set
    train_balanced_accuracy = balanced_accuracy_score(y_train, predictions_train)
    print(f"Train set balanced accuracy: {train_balanced_accuracy:.4f}")

    # Calculating and printing the average Hinge loss of the train set
    train_hinge_loss_value = final_model._hinge_loss(X_train, y_train)
    print(f"Average Hinge loss for the train set: {train_hinge_loss_value:.4f}")

    # Making and printing the predictions on the test set
    predictions_test = final_model.predict(X_test)
    label_predictions = final_model.labels(predictions_test)
    print("Predictions:", label_predictions)

    # Calculating and printing the balanced accuracy on the test set
    test_balanced_accuracy = balanced_accuracy_score(y_test, predictions_test)
    print(f"Test set balanced accuracy: {test_balanced_accuracy:.4f}")

    # Calculating and printing the average Hinge lossof the test set
    test_hinge_loss_value = final_model._hinge_loss(X_test, y_test)
    print(f"Average Hinge loss for the test set: {test_hinge_loss_value:.4f}")

    # Creating and printing the confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions_test)
    print("Confusion Matrix:")
    print(conf_matrix)


# Plotting all the graphs
if __name__ == "__main__":
    main()
