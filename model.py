import numpy as np
import cv2
from PIL import Image
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CropDiseaseModel:
    """
    Crop Disease Detection Model Training and Inference Class
    """
    
    def __init__(self, img_height=224, img_width=224, num_classes=33):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        
        # Disease classes mapping
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    def create_model(self):
        """
        Create CNN model architecture for crop disease detection
        """
        # Use transfer learning with EfficientNetB0
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess single image for prediction
        """
        try:
            # Load and preprocess image
            image = tf.keras.utils.load_img(
                image_path, 
                target_size=(self.img_height, self.img_width)
            )
            image_array = tf.keras.utils.img_to_array(image)
            image_array = tf.expand_dims(image_array, 0)  # Create batch dimension
            image_array = image_array / 255.0  # Normalize
            
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32):
        """
        Create data generators for training
        """
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """
        Train the model with callbacks
        """
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_crop_disease_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=20):
        """
        Fine-tune the model by unfreezing some layers
        """
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Continue training
        history_fine = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            verbose=1
        )
        
        return history_fine
    
    def predict_disease(self, image_path):
        """
        Predict disease from image path
        """
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        if predicted_class < len(self.class_names):
            disease_name = self.class_names[predicted_class]
        else:
            disease_name = "Unknown"
        
        return {
            'disease': disease_name,
            'confidence': float(confidence),
            'class_id': int(predicted_class),
            'all_predictions': predictions[0].tolist()
        }
    
    def evaluate_model(self, test_generator):
        """
        Evaluate model performance
        """
        if self.model is None:
            print("Model not loaded.")
            return None
        
        # Get predictions
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        
        # Classification report
        class_labels = list(test_generator.class_indices.keys())
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        return report, cm
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize model
    crop_model = CropDiseaseModel()
    
    # Create model architecture
    model = crop_model.create_model()
    print("Model created successfully!")
    print(f"Model summary: {model.summary()}")
    
    # For training (uncomment if you have dataset):
    # train_generator, val_generator = crop_model.create_data_generators(
    #     train_dir="path/to/train",
    #     val_dir="path/to/validation"
    # )
    # history = crop_model.train_model(train_generator, val_generator)
    # crop_model.save_model("crop_disease_model.h5")
    
    # For prediction (after training):
    # result = crop_model.predict_disease("path/to/image.jpg")
    # print(f"Prediction: {result}")
