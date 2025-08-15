import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from random import sample

# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CLASSES = 4
BATCH_SIZE = 20
EPOCHS = 20

# Class mapping
CLASS_MAPPING = {
    'Drone1': 0,
    'Drone2': 1,
    'Drone3': 2,
    'No_Drone': 3
}

def image_generator(folder, batch_size=32):
    """Improved generator with proper error handling and batching"""
    while True:
        image_paths = []
        labels = []
        
        # Collect all valid image paths and labels
        for class_name, class_idx in CLASS_MAPPING.items():
            class_dir = os.path.join(folder, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        # Shuffle together
        combined = list(zip(image_paths, labels))
        np.random.shuffle(combined)
        image_paths, labels = zip(*combined) if combined else ([], [])
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_images = []
            batch_labels = []
            
            for img_path, label in zip(image_paths[i:i+batch_size], labels[i:i+batch_size]):
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=-1)
                    batch_images.append(img_array)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                    continue
            
            if batch_images:
                yield np.array(batch_images), to_categorical(batch_labels, NUM_CLASSES)

def create_model(input_shape, num_classes):
    """Create and compile the CNN model with mixed precision support"""
    model = models.Sequential([
        layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(16, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax', dtype='float32')  # Ensure float32 for softmax
    ])

    # Enable mixed precision if GPU available
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def show_predictions(model, test_images_dir, num_samples=48, figsize=(20, 10)):
    """Visualize model predictions with true vs predicted labels"""
    all_images = []
    for class_name in CLASS_MAPPING.keys():
        class_dir = os.path.join(test_images_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in images:
                img_path = os.path.join(class_dir, img_file)
                try:
                    img = Image.open(img_path).convert('L')
                    img_array = np.array(img)
                    all_images.append((img_array, class_name, img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

    if not all_images:
        print("No images found in test directory!")
        return

    num_samples = min(num_samples, len(all_images))
    selected = sample(all_images, num_samples)

    cols = min(4, num_samples)
    rows = math.ceil(num_samples / cols)

    plt.figure(figsize=figsize)
    for idx, (img_array, true_label, img_path) in enumerate(selected, 1):
        img_norm = np.expand_dims(img_array.astype('float32') / 255.0, axis=(0, -1))
        pred = model.predict(img_norm, verbose=0)[0]
        pred_class = list(CLASS_MAPPING.keys())[np.argmax(pred)]
        confidence = np.max(pred)

        plt.subplot(rows, cols, idx)
        plt.imshow(img_array, cmap='gray')
        color = 'green' if pred_class == true_label else 'red'
        plt.title(f"{os.path.basename(img_path)}\nTrue: {true_label}\nPred: {pred_class} ({confidence:.2f})",
                  color=color, fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', bbox_inches='tight', dpi=150)
    print(f"Saved {num_samples} predictions to predictions.png")

def validate_data_directories():
    """Check that all required directories and files exist"""
    for split in ['train_images', 'val_images', 'test_images']:
        if not os.path.exists(split):
            raise ValueError(f"Missing directory: {split}")
        for cls in CLASS_MAPPING:
            cls_path = os.path.join(split, cls)
            if not os.path.exists(cls_path):
                raise ValueError(f"Missing class directory: {cls_path}")
            if not os.listdir(cls_path):
                raise ValueError(f"Empty class directory: {cls_path}")

def print_class_distribution():
    """Print distribution of classes across splits"""
    print("\nClass distribution:")
    for split in ['train_images', 'val_images', 'test_images']:
        print(f"\n{split}:")
        for cls in CLASS_MAPPING:
            count = len(os.listdir(os.path.join(split, cls)))
            print(f"{cls}: {count} samples")

def main():
    # Validate data directories
    validate_data_directories()
    print_class_distribution()

    # Calculate dataset sizes
    train_image_count = sum(len(os.listdir(os.path.join('train_images', cls))) 
                      for cls in CLASS_MAPPING.keys())
    print(f"\nTraining Dataset Info:")
    print(f"Total images: {train_image_count}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch: {math.ceil(train_image_count / BATCH_SIZE)}\n")

    # Create datasets using generators
    train_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator('train_images', BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator('val_images', BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(
        lambda: image_generator('test_images', BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # Calculate steps using ceiling to ensure all data is used
    train_steps = math.ceil(sum(len(os.listdir(os.path.join('train_images', cls))) 
                 for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
    val_steps = math.ceil(sum(len(os.listdir(os.path.join('val_images', cls))) 
               for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
    test_steps = math.ceil(sum(len(os.listdir(os.path.join('test_images', cls))) 
                for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)

    # Build and train model
    model = create_model((IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=[reduce_lr, model_checkpoint]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Additional evaluation: No_Drone vs Drones
    print("\nNo_Drone vs Drones Evaluation:")
    y_true, y_pred = [], []
    for images, labels in test_dataset.take(test_steps):
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    # Convert to binary: No_Drone (3) vs Drones (0-2)
    y_true_binary = [1 if x == 3 else 0 for x in y_true]
    y_pred_binary = [1 if x == 3 else 0 for x in y_pred]
    
    print(f"Accuracy: {accuracy_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Precision (No_Drone): {precision_score(y_true_binary, y_pred_binary):.4f}")
    print(f"Recall (No_Drone): {recall_score(y_true_binary, y_pred_binary):.4f}")

    # Save final model
    model.save('flying_object_classifier_final.keras')
    return model

if __name__ == '__main__':
    model = main()
    show_predictions(model, 'test_images')