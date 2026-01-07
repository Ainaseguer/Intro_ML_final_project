from data_processing import data_processing
from SVM import SVM


def main():
    # Extracting features and targets from the data set
    X, y = data_processing()

    # Separating the training sets
    X_train = X[:380]
    y_train = y[:380]

    # Separating the test sets
    X_test = X[380:]
    y_test = y[380:]

    # Initializing the model
    model = SVM()

    # Training the model on the training sets
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Making and printing the predictions
    predictions = model.predict(X_test.to_numpy())
    print("Predictions:", predictions)

    # Calculating and printing the average Hinge loss
    hinge_loss_value = model.calculate_loss(
        X_test.to_numpy(), y_test.to_numpy(), "hinge"
    )
    print(f"Average Hinge loss: {hinge_loss_value:.4f}")

    # Calculating and printing BCE
    bce_loss_value = model.calculate_loss(X_test.to_numpy(), y_test.to_numpy(), "bce")
    print(f"Average binary cross entrophy: {bce_loss_value:.4f}")


if __name__ == "__main__":
    main()
