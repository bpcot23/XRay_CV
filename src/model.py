import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


class EfficientNet:

    def __init__(self, config):
        self.target_condition = config["file_processing"]["target_condition"]

        self.image_height = config["file_processing"]["image_height"]
        self.image_width = config["file_processing"]["image_width"]
        self.metrics_output_path = config["file_processing"]["metrics_output_path"]

        self.dropout_rate = config["model_building"]["dropout_rate"]
        self.activation_fn = config["model_building"]["activation_fn"]
        self.loss_fn = config["model_building"]["loss_fn"]
        self.num_epochs = config["model_building"]["num_epochs"]
        self.fine_tune_at = config["model_building"]["fine_tune_at"]
        self.decision_threshold = config["model_building"]["decision_threshold"]
        self.fine_tune_learning_rate = config["model_building"]["fine_tune_learning_rate"]

        self.output_csv_path = config["file_processing"]["output_csv_path"]
        self.model = None

    
    def build_model(self):
        """
        Initialize the EfficientNetB3 model, establish the model's layers, and compile the model.
        
        Args:
            None
        
        Returns:
            None
        """

        model_base = EfficientNetB3(weights = 'imagenet', include_top = False, input_shape = (self.image_height, self.image_width, 3))

        # Freeze all layers initially
        model_base.trainable = False 

        # Add model layers
        model = models.Sequential([
                model_base,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(self.dropout_rate),
                layers.Dense(1, 
                             activation = self.activation_fn,
                             kernel_regularizer=l2(0.01))
                ])
        
        # Compile the model
        model.compile(optimizer = tf.keras.optimizers.Adam(),
                      loss = self.loss_fn,
                      metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                      )

        self.model = model
        
        # Comment in to visualize the blocks and layers of the model
        # with open("model_layers.txt", "w") as f:
        #     for i, layer in enumerate(model_base.layers):
        #         f.write(f"Layer {i}: {layer.name}")

    
    def unfreeze_layers(self):
        """
        This function allows the unfreezing of layers within the model for fine-tuning.
        This is currently only established for unfreezing from layer "x" to the end of the model,
        not for unfreezing just a middle subset of layers from "x" to "y".

        Args:
            None
        
        Returns:
            None
        """
        # Unfreeze layers starting at the fine_tune_at index
        model_base = self.model.layers[0] 
        for layer in model_base.layers[:self.fine_tune_at]:
            layer.trainable = False
        for layer in model_base.layers[self.fine_tune_at:]:
            layer.trainable = True

        # Recompile the model with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = float(self.fine_tune_learning_rate)),
            loss = self.loss_fn,
            metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )


    def train_model(self, train_dataset, validation_dataset):
        """
        Fits the model using specified train and validation datasets
        
        Args:
            train_dataset (tf.Dataset): The training dataset used for fitting the model. Comprised of images and gold labels
            validation_dataset (tf.Dataset): The validation dataset used for verifying the model's trained weights
        
        Returns:
            history (tf.keras.callbacks.History): A history object that contains training metrics across epochs
        """

        # Weight classes differently to offset the imbalance in classes
        train_labels = np.concatenate([y for x, y in train_dataset], axis=0)
        class_weights = compute_class_weight(
                                            class_weight='balanced',
                                            classes=np.unique(train_labels),
                                            y=train_labels
                                            )
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}

        model = self.model

        # Fit the model to the dataset
        history = model.fit(train_dataset,
                            epochs = self.num_epochs,
                            validation_data = validation_dataset,
                            class_weight = class_weights
                            )
        
        # Visualize model's accuracy and loss functions across epochs
        self.plot_training_history(history)

        self.model = model
        return history
    
   
    def generate_predictions(self, test_dataset):
        """
        Generates predictions on the test dataset using the fitted model

        Args:
            test_dataset (tf.Dataset): The training dataset used for fitting the model. Comprised of images and gold labels
        
        Returns:
            predictions_dataset (pd.DataFrame): A dataframe comprised of gold labels and predicted labels for each test set image
        """
        # Generate binary classification predictions
        y_pred = self.model.predict(test_dataset)
        y_pred_binary = (y_pred > self.decision_threshold).astype(int).flatten()
        print(y_pred)

        # Extract gold labels from dataset
        y_gold = np.concatenate([y for x, y in test_dataset], axis=0)

        # Calculate metrics
        accuracy = tf.keras.metrics.Accuracy()(y_gold, y_pred_binary).numpy()
        precision = tf.keras.metrics.Precision()(y_gold, y_pred_binary).numpy()
        recall = tf.keras.metrics.Recall()(y_gold, y_pred_binary).numpy()
        auc = tf.keras.metrics.AUC()(y_gold, y_pred.flatten()).numpy()

        # Save Model prediction metrics to a text file
        with open(self.metrics_output_path, "w") as f:
            f.write(f"Model Evaluation Metrics:\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")

        # Visualize predictions with a confusion matrix
        cm = confusion_matrix(y_gold, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absent", "Present"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # Save predictions to CSV
        predictions_dataset = pd.DataFrame({
            'Gold Label': y_gold,
            'Predicted Label': y_pred_binary
        })
        predictions_dataset.to_csv(self.output_csv_path, index=False)

        return predictions_dataset


    def plot_training_history(self, history):
        """
        Uses matplotlib to visualize the relationship between the training and validation data on both
        accuracy and loss. This helps to visualize overfitting and other training issues.  

        Args:
            history (tf.keras.callbacks.History): A history object that contains training metrics across epochs
        
        Returns:
            None
        """
        # Extract training history metrics
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot accuracy
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Show plots
        plt.tight_layout()
        plt.show()
