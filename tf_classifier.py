import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from preprocess import preprocess_trading_data

class StockClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # First block
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Second block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Third block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fourth block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Fifth block
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=64):
        # Create callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6
        )
        
        # Train the model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history

    def plot_training_history(self):
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        self.model.save(filepath)
        
    def load_model(self, filepath):
        self.model = models.load_model(filepath)
        
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X):
        probs = self.model.predict(X)
        return np.column_stack((1 - probs, probs))

def main():
    # Load and preprocess data
    csv_path = "training.csv"
    X, y, scaler = preprocess_trading_data(csv_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    classifier = StockClassifier()
    classifier.build_model(input_shape=(X.shape[1],))
    
    # Train the model
    history = classifier.train(X_train, y_train, epochs=100)
    
    # Plot training history
    classifier.plot_training_history()
    
    # Evaluate the model
    test_loss, test_accuracy, test_auc = classifier.model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Save the model
    classifier.save_model('tf_stock_classifier.h5')
    
    # Make sample predictions
    sample_predictions = classifier.predict(X_test[:5])
    sample_probabilities = classifier.predict_proba(X_test[:5])
    
    print("\nSample predictions:", sample_predictions)
    print("Sample probabilities:", sample_probabilities)
    print("Actual values:", y_test[:5].values)

if __name__ == "__main__":
    main() 