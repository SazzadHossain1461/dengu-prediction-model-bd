import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(filepath='F:\My_Projects\dengu-pred-model-ml\dataset.csv'):
    """
    Loads the dataset, performs exploratory data analysis,
    and preprocesses it for modeling.
    """
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please make sure the dataset.csv file is in the same directory as this script.")
        return None, None, None, None

    print("--- Data Head ---")
    print(df.head())
    print("\n--- Data Info ---")
    df.info()
    print("\n--- Data Description ---")
    print(df.describe())

    # --- 2. Exploratory Data Analysis (EDA) ---
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Outcome', data=df)
    plt.title('Distribution of Dengue Outcome (0 = Negative, 1 = Positive)')
    plt.savefig('outcome_distribution.png')
    print("\nSaved 'outcome_distribution.png' to show target variable balance.")

    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='Age', hue='Outcome', kde=True, bins=30)
    plt.title('Age Distribution by Dengue Outcome')
    plt.savefig('age_distribution.png')
    print("Saved 'age_distribution.png' to show age distribution.")

    # --- 3. Define Features (X) and Target (y) ---
    if 'Outcome' not in df.columns:
        print("Error: 'Outcome' column not found in the dataset.")
        return None, None, None, None
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # --- 4. Define Preprocessing Steps ---
    
    # Identify numerical and categorical features
    # 'NS1', 'IgG', 'IgM' are binary/numerical, 'Age' is continuous
    numerical_features = ['Age', 'NS1', 'IgG', 'IgM']
    
    # 'Gender', 'Area', 'AreaType', 'HouseType', 'District' are categorical
    categorical_features = ['Gender', 'Area', 'AreaType', 'HouseType', 'District']

    # Create preprocessing pipelines
    # Scale numerical features
    numerical_transformer = StandardScaler()

    # One-hot encode categorical features
    # handle_unknown='ignore' will set new categories in test data to all zeros
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Use ColumnTransformer to apply transformers to the correct columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns (though we've used all)
    )

    # --- 5. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply preprocessing
    # Fit on training data and transform both train and test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for the model input shape
    # This is slightly complex but good for understanding the input layer
    try:
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numerical_features + list(cat_feature_names)
        print(f"\nTotal number of features after preprocessing: {len(feature_names)}")
        
        # Convert sparse matrix to dense array for TensorFlow
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
        
    except AttributeError: # Handle older sklearn versions
        print("Could not get feature names, proceeding...")
        # Convert sparse matrix to dense array for TensorFlow
        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()
            X_test_processed = X_test_processed.toarray()

    return X_train_processed, X_test_processed, y_train, y_test

def build_model(input_shape):
    """
    Builds a Sequential Keras model for binary classification.
    [Image of a simple neural network architecture]
    """
    model = tf.keras.models.Sequential([
        # Input layer: Must match the number of features
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # Hidden layer 1
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3), # Dropout for regularization
        
        # Hidden layer 2
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer: 1 neuron for binary classification, sigmoid activation
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- Model Summary ---")
    model.summary()
    return model

def plot_history(history):
    """
    Plots training & validation accuracy and loss.
    """
    # Plot Accuracy
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('model_training_history.png')
    print("Saved 'model_training_history.png' to show training progress.")

def main():
    """
    Main function to run the full pipeline.
    """
    # 1. Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('dataset.csv')
    
    if X_train is None:
        return

    # 2. Build the model
    input_shape = X_train.shape[1] # Number of features
    model = build_model(input_shape)

    # 3. Train the model
    print("\n--- Starting Model Training ---")
    # Add an EarlyStopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2, # Use part of training data for validation
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    print("--- Model Training Finished ---")

    # 4. Plot training history
    plot_history(history)

    # 5. Evaluate the model on the test set
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 6. Get detailed metrics
    # Get probability predictions
    y_pred_probs = model.predict(X_test)
    # Convert probabilities to binary classes (0 or 1)
    y_pred = (y_pred_probs > 0.5).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Saved 'confusion_matrix.png' for detailed performance review.")
    
    # Show all plots at the end
    print("\nDisplaying plots... Close the plot windows to exit.")
    plt.show()

if __name__ == "__main__":
    main()