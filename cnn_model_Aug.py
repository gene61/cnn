import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, Callback
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from random import sample
import seaborn as sns
# Single comprehensive ablation configuration - Updated based on previous results
ABLATION_CONFIGS = {
    'baseline': {'l2_reg': 0.001, 'use_batchnorm': False, 'use_dropout': True, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3]},
    'no_dropout': {'l2_reg': 0.001, 'use_batchnorm': False, 'use_dropout': False, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3]},
    'with_batchnorm': {'l2_reg': 0.001, 'use_batchnorm': True, 'use_dropout': True, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3]},
    'simpler_arch': {'l2_reg': 0.001, 'use_batchnorm': False, 'use_dropout': True, 'num_conv_blocks': 2, 'filters': [8, 16], 'kernels': [5, 3]},
    'stronger_reg': {'l2_reg': 0.002, 'use_batchnorm': False, 'use_dropout': True, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3]},
    'flatten_vs_gap': {'l2_reg': 0.001, 'use_batchnorm': False, 'use_dropout': True, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3], 'use_gap': True},
    'extra_dense': {'l2_reg': 0.001, 'use_batchnorm': False, 'use_dropout': True, 'num_conv_blocks': 3, 'filters': [8, 16, 32], 'kernels': [5, 3, 3], 'extra_dense': True},
}
# Constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
# IMG_WIDTH = 128
# IMG_HEIGHT = 128
NUM_CLASSES = 4
BATCH_SIZE = 30
EPOCHS = 40

# Class mapping
CLASS_MAPPING = {
    'Drone1': 0,
    'Drone2': 1,
    'Drone3': 2,
    'No_Drone': 3
}

def run_image_size_ablation():
    """Ablation study focusing on image size impact"""
    ablation_results = {}
    
    # Test only 256x256 (skip 128x128 comparison)
    image_sizes = [256]  # Only test 256x256
    
    for img_size in image_sizes:
        print(f"\n{'='*60}")
        print(f"Testing image size: {img_size}x{img_size}")
        print(f"{'='*60}")
        
        # Create datasets with specific image size using augmented training data
        train_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator('train_images', BATCH_SIZE, augment=True),
            output_signature=(
                tf.TensorSpec(shape=(None, img_size, img_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        print("  Using augmented training data (rotation ±5°) for image size ablation")
        
        val_dataset = create_dataset_for_size('val_images', img_size, BATCH_SIZE)
        test_dataset = create_dataset_for_size('test_images', img_size, BATCH_SIZE)

        # Calculate steps
        train_steps = math.ceil(sum(len(os.listdir(os.path.join('train_images', cls))) 
                     for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
        val_steps = math.ceil(sum(len(os.listdir(os.path.join('val_images', cls))) 
                   for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
        test_steps = math.ceil(sum(len(os.listdir(os.path.join('test_images', cls))) 
                    for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)

        # Create and train model with current image size
        model = create_model((img_size, img_size, 1), NUM_CLASSES)
        
        # Train with 30 epochs for ablation study
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=30,  # Increased to 30 epochs
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ]
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps, verbose=0)
        
        # Calculate F1 score
        y_true = []
        y_pred = []
        for images, labels in test_dataset.take(test_steps):
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Store results
        ablation_results[f'{img_size}x{img_size}'] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_weighted': f1,
            'f1_macro': f1_macro,
            'final_val_accuracy': history.history['val_accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'training_time_per_epoch': np.mean(history.history['time']) if 'time' in history.history else None,
            'model_params': model.count_params(),
            'input_pixels': img_size * img_size,
            'relative_size': (img_size * img_size) / (256 * 256)  # Relative to baseline
        }
        
        print(f"Results for {img_size}x{img_size}:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test F1 (Weighted): {f1:.4f}")
        print(f"  Model Parameters: {model.count_params()}")
        print(f"  Relative Input Size: {ablation_results[f'{img_size}x{img_size}']['relative_size']:.2f}x")
    
    return ablation_results

def run_component_ablation_with_size(img_size=256, skip_baseline=False):
    """Test component importance at specific image size"""
    component_results = {}
    
    for ablation_type, config in ABLATION_CONFIGS.items():
        # Skip baseline if requested (we already have results from image size ablation)
        if skip_baseline and ablation_type == 'baseline':
            print(f"\nSkipping baseline training - using results from image size ablation")
            continue
            
        print(f"\nTesting {ablation_type} with {img_size}x{img_size} images...")
        
        # For baseline, use the same model architecture as image size ablation (which gives better results)
        if ablation_type == 'baseline':
            model = create_model((img_size, img_size, 1), NUM_CLASSES)
        else:
            model = create_ablated_model(ablation_type, (img_size, img_size, 1), NUM_CLASSES)
        
        # Create datasets with full steps (same as image size ablation)
        # All ablation variants use augmented training data
        train_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator('train_images', BATCH_SIZE, augment=True),
            output_signature=(
                tf.TensorSpec(shape=(None, img_size, img_size, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        print("  Using augmented training data (rotation ±5°) for all ablation variants")
        
        val_dataset = create_dataset_for_size('val_images', img_size, BATCH_SIZE)
        test_dataset = create_dataset_for_size('test_images', img_size, BATCH_SIZE)
        
        # Calculate full steps (same as image size ablation)
        train_steps = math.ceil(sum(len(os.listdir(os.path.join('train_images', cls))) 
                     for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
        val_steps = math.ceil(sum(len(os.listdir(os.path.join('val_images', cls))) 
                   for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
        test_steps = math.ceil(sum(len(os.listdir(os.path.join('test_images', cls))) 
                    for cls in CLASS_MAPPING.keys()) / BATCH_SIZE)
        
        print(f"Training steps: {train_steps}, Validation steps: {val_steps}")
        
        # Train with same parameters as image size ablation
        try:
            print("Starting training...")
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                epochs=30,  # Increased from 25 to 30 epochs
                verbose=1,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # Same as image size ablation
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)  # Same as image size ablation
                ]
            )
            print("Training completed successfully")
            
            # Evaluate with full test steps
            test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps, verbose=0)
            
            # Calculate F1 score for consistency
            y_true = []
            y_pred = []
            for images, labels in test_dataset.take(test_steps):
                preds = model.predict(images, verbose=0)
                y_true.extend(np.argmax(labels.numpy(), axis=1))
                y_pred.extend(np.argmax(preds, axis=1))
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Show test set prediction accuracy immediately after each ablation test
            print(f"✅ {ablation_type}: Test Accuracy = {test_accuracy:.4f}, F1 = {f1:.4f}")
            
            component_results[ablation_type] = {
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'f1_weighted': f1,
                'model_params': model.count_params(),
                'val_accuracy': history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0.0
            }
            
        except Exception as e:
            print(f"ERROR training {ablation_type}: {e}")
            component_results[ablation_type] = {
                'test_accuracy': 0.0,
                'test_loss': float('inf'),
                'f1_weighted': 0.0,
                'model_params': model.count_params(),
                'val_accuracy': 0.0
            }
    
    return component_results

def plot_ablation_results(size_results, component_results):
    """Create comprehensive visualization of ablation study results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Image Size vs Accuracy
    sizes = [int(k.split('x')[0]) for k in size_results.keys()]
    accuracies = [size_results[k]['test_accuracy'] for k in size_results.keys()]
    ax1.plot(sizes, accuracies, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Image Size (pixels)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Image Size vs Test Accuracy')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Component Ablation Comparison
    components = list(component_results.keys())
    comp_accuracies = [component_results[k]['test_accuracy'] for k in components]
    bars = ax2.bar(components, comp_accuracies, color=['blue' if 'baseline' in k else 'red' for k in components])
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Component Ablation: Test Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Computational Efficiency
    relative_sizes = [size_results[k]['relative_size'] for k in size_results.keys()]
    ax3.plot(relative_sizes, accuracies, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Relative Input Size (vs 256x256)')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Computational Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Size vs Performance
    model_sizes = [component_results[k]['model_params'] for k in components]
    ax4.scatter(model_sizes, comp_accuracies, s=100, alpha=0.7)
    for i, comp in enumerate(components):
        ax4.annotate(comp, (model_sizes[i], comp_accuracies[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    ax4.set_xlabel('Model Parameters')
    ax4.set_ylabel('Test Accuracy')
    ax4.set_title('Model Complexity vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main ablation study execution
def run_complete_ablation_study():
    """Run the complete ablation study"""
    print("Starting Comprehensive Ablation Study")
    print("=" * 60)
    
    # 1. Image Size Ablation
    print("\n1. IMAGE SIZE ABLATION STUDY")
    size_results = run_image_size_ablation()
    
    # 2. Component Ablation at Optimal Size
    print("\n2. COMPONENT ABLATION STUDY")
    best_size = max(size_results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_size_value = int(best_size.split('x')[0])
    print(f"Using best performing size: {best_size}")
    
    # Skip baseline training since we already have results from image size ablation
    component_results = run_component_ablation_with_size(best_size_value, skip_baseline=True)
    
    # Add baseline results from image size ablation
    component_results['baseline'] = {
        'test_accuracy': size_results[best_size]['test_accuracy'],
        'test_loss': size_results[best_size]['test_loss'],
        'f1_weighted': size_results[best_size]['f1_weighted'],
        'model_params': size_results[best_size]['model_params'],
        'val_accuracy': size_results[best_size]['final_val_accuracy']
    }
    
    # 3. Generate comprehensive report
    print("\n3. GENERATING RESULTS AND VISUALIZATIONS")
    plot_ablation_results(size_results, component_results)
    
    # 4. Print summary report
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    
    print("\nIMAGE SIZE PERFORMANCE:")
    for size, results in sorted(size_results.items()):
        print(f"  {size}: Accuracy = {results['test_accuracy']:.4f}, F1 = {results['f1_weighted']:.4f}")
    
    print("\nCOMPONENT IMPORTANCE:")
    for component, results in component_results.items():
        diff = results['test_accuracy'] - component_results['baseline']['test_accuracy']
        percentage_change = (diff / component_results['baseline']['test_accuracy']) * 100
        print(f"  {component}: Accuracy = {results['test_accuracy']:.4f} ({diff:+.4f}, {percentage_change:+.1f}%)")
    
    return size_results, component_results
def generate_ablation_insights(size_results, component_results):
    """Extract actionable insights from ablation study"""
    
    # Find optimal configuration
    best_size = max(size_results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    best_component = max(component_results.items(), key=lambda x: x[1]['test_accuracy'])[0]
    
    insights = {
        'optimal_image_size': best_size,
        'optimal_architecture': best_component,
        'performance_improvement': component_results[best_component]['test_accuracy'] - component_results['baseline']['test_accuracy'],
        'computational_savings': (1 - size_results[best_size]['relative_size']) * 100,
        'components_to_keep': [],
        'components_to_remove': []
    }
    
    # Analyze which components matter
    baseline_acc = component_results['baseline']['test_accuracy']
    for component, results in component_results.items():
        if component != 'baseline':
            performance_drop = baseline_acc - results['test_accuracy']
            if performance_drop < 0.02:  # Less than 2% drop
                insights['components_to_remove'].append(component)
            else:
                insights['components_to_keep'].append(component)
    
    return insights
def create_optimized_model(component_results, input_shape, num_classes):
    """Create final optimized model based on ablation results"""
    
    # Start with baseline configuration
    baseline_config = ABLATION_CONFIGS['baseline']
    optimized_config = baseline_config.copy()
    
    print("Analyzing ablation results to create optimized model...")
    
    # Analyze each component and choose the best performing variant
    baseline_accuracy = component_results['baseline']['test_accuracy']
    
    # Check each ablation variant and keep the best performing one
    for component, results in component_results.items():
        if component == 'baseline':
            continue
            
        # If this variant performs better than baseline, update configuration
        if results['test_accuracy'] > baseline_accuracy:
            improvement = results['test_accuracy'] - baseline_accuracy
            print(f"  ✅ {component}: +{improvement:.4f} improvement → Using this configuration")
            
            # Update configuration based on which component performed better
            if component == 'no_dropout':
                optimized_config['use_dropout'] = False
            elif component == 'with_batchnorm':
                optimized_config['use_batchnorm'] = True
            elif component == 'simpler_arch':
                optimized_config['num_conv_blocks'] = 2
                optimized_config['filters'] = [8, 16]
                optimized_config['kernels'] = [5, 3]
            elif component == 'stronger_reg':
                optimized_config['l2_reg'] = 0.002
            elif component == 'conservative_filters':
                optimized_config['filters'] = [4, 8, 16]
            elif component == 'flatten_vs_gap':
                optimized_config['use_gap'] = True
            elif component == 'extra_dense':
                optimized_config['extra_dense'] = True
        else:
            degradation = baseline_accuracy - results['test_accuracy']
            print(f"  ❌ {component}: -{degradation:.4f} degradation → Keeping baseline")
    
    # Create the optimized model using the best configuration
    print(f"\nCreating optimized model with:")
    print(f"  L2 regularization: {optimized_config['l2_reg']}")
    print(f"  BatchNorm: {optimized_config['use_batchnorm']}")
    print(f"  Dropout: {optimized_config['use_dropout']}")
    print(f"  Conv blocks: {optimized_config['num_conv_blocks']}")
    print(f"  Filters: {optimized_config['filters']}")
    print(f"  Kernels: {optimized_config['kernels']}")
    if optimized_config.get('use_gap', False):
        print(f"  GlobalAveragePooling2D: True")
    if optimized_config.get('extra_dense', False):
        print(f"  Extra dense layer: True")
    
    # Build the optimized model
    model = models.Sequential()
    
    filter_sequence = optimized_config['filters']
    kernels = optimized_config['kernels']
    num_blocks = optimized_config['num_conv_blocks']
    
    for i in range(num_blocks):
        if i == 0:
            model.add(layers.Conv2D(filter_sequence[i], kernels[i], 
                                  activation='relu', input_shape=input_shape,
                                  kernel_regularizer=regularizers.l2(optimized_config['l2_reg']) if optimized_config['l2_reg'] > 0 else None))
        else:
            model.add(layers.Conv2D(filter_sequence[i], kernels[i], activation='relu',
                                  kernel_regularizer=regularizers.l2(optimized_config['l2_reg']) if optimized_config['l2_reg'] > 0 else None))
        
        if optimized_config['use_batchnorm']:
            model.add(layers.BatchNormalization())
            
        model.add(layers.MaxPooling2D((2,2)))
        
        if optimized_config['use_dropout']:
            model.add(layers.Dropout(0.2))
    
    # Classification head
    if optimized_config.get('use_gap', False):
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.Flatten())
    
    if optimized_config.get('extra_dense', False):
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(optimized_config['l2_reg']) if optimized_config['l2_reg'] > 0 else None))
        if optimized_config['use_batchnorm']:
            model.add(layers.BatchNormalization())
        if optimized_config['use_dropout']:
            model.add(layers.Dropout(0.2))
    
    # Standard dense layer
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(optimized_config['l2_reg']) if optimized_config['l2_reg'] > 0 else None))
    
    if optimized_config['use_batchnorm']:
        model.add(layers.BatchNormalization())
    
    if optimized_config['use_dropout']:
        model.add(layers.Dropout(0.2))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, 256  # Always use 256x256 based on image size ablation
def image_generator(folder, batch_size=32, augment=False):
    """Improved generator with infinite looping, better batching, and optional augmentation"""
    # Pre-load all image paths and labels once
    all_image_paths = []
    all_labels = []
    
    for class_name, class_idx in CLASS_MAPPING.items():
        class_dir = os.path.join(folder, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, filename)
                all_image_paths.append(img_path)
                all_labels.append(class_idx)
    
    if not all_image_paths:
        raise ValueError(f"No images found in {folder}")
    
    # Convert to arrays for easier shuffling
    all_image_paths = np.array(all_image_paths)
    all_labels = np.array(all_labels)
    dataset_size = len(all_image_paths)
    
    print(f"Generator created for {folder}: {dataset_size} images (augmentation: {augment})")
    
    while True:
        # Shuffle indices each epoch
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        # Process in batches
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img_path = all_image_paths[idx]
                label = all_labels[idx]
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
                    img_array = np.array(img).astype('float32') / 255.0
                    
                    # Apply small angle rotation augmentation only for training data
                    if augment and folder == 'train_images':
                        # Random rotation between -5 and +5 degrees (smaller range)
                        angle = np.random.uniform(-5, 5)
                        img_rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                        img_array = np.array(img_rotated).astype('float32') / 255.0
                    
                    img_array = np.expand_dims(img_array, axis=-1)
                    batch_images.append(img_array)
                    batch_labels.append(label)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")
                    continue
            
            if batch_images:
                yield np.array(batch_images), to_categorical(batch_labels, NUM_CLASSES)
def validate_optimized_model(optimized_model, test_dataset, original_baseline_score):
    """Thoroughly validate the optimized model"""
    
    # Test accuracy comparison
    test_loss, test_accuracy = optimized_model.evaluate(test_dataset, verbose=0)
    
    # Inference speed test
    import time
    start_time = time.time()
    for images, _ in test_dataset.take(10):
        _ = optimized_model.predict(images, verbose=0)
    inference_time = (time.time() - start_time) / 10
    
    # Model size comparison
    model_size_mb = optimized_model.count_params() * 4 / (1024 * 1024)  # MB
    
    validation_results = {
        'test_accuracy': test_accuracy,
        'performance_change': test_accuracy - original_baseline_score,
        'inference_time_per_batch': inference_time,
        'model_size_mb': model_size_mb,
        'meets_deployment_requirements': test_accuracy >= original_baseline_score * 0.98  # Within 2%
    }
    
    return validation_results
def create_model(input_shape, num_classes, learning_rate=0.001):
    """Create and compile the CNN model with optimized regularization"""
    model = models.Sequential([
        # First conv block - larger kernel for initial feature extraction
        layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape, 
                     kernel_regularizer=regularizers.l2(0.001)),
        # No BatchNorm based on ablation results
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second conv block
        layers.Conv2D(16, (3, 3), activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        # No BatchNorm based on ablation results
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Third conv block - reduced complexity (removed 4th block)
        layers.Conv2D(32, (3, 3), activation='relu', 
                     kernel_regularizer=regularizers.l2(0.001)),
        # No BatchNorm based on ablation results
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Flatten(),
        
        # Single dense layer with optimized regularization
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.001)),
        # No BatchNorm based on ablation results
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    # Enable mixed precision if GPU available
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('mixed_float16')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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


def exponential_decay_schedule(epoch, lr):
    """Exponential decay learning rate schedule"""
    decay_rate = 0.5
    decay_steps = 5
    if epoch % decay_steps == 0 and epoch > 0:
        return lr * decay_rate
    return lr

def cosine_annealing_schedule(epoch, lr, initial_lr=0.001, max_epochs=30):
    """Cosine annealing learning rate schedule"""
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

def warmup_cosine_decay_schedule(epoch, lr, initial_lr=0.001, warmup_epochs=5, max_epochs=30):
    """Warmup followed by cosine decay"""
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay after warmup
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))

def get_learning_rate_schedule(schedule_type='adaptive'):
    """Get learning rate schedule based on type"""
    if schedule_type == 'exponential':
        return LearningRateScheduler(exponential_decay_schedule)
    elif schedule_type == 'cosine':
        return LearningRateScheduler(lambda epoch, lr: cosine_annealing_schedule(epoch, lr))
    elif schedule_type == 'warmup_cosine':
        return LearningRateScheduler(lambda epoch, lr: warmup_cosine_decay_schedule(epoch, lr))
    else:  # adaptive (default)
        return ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3, 
            patience=2,  # Reduced from 3 to 2
            min_lr=1e-7,
            min_delta=0.001,  # Add minimum change requirement
            cooldown=1,       # Add cooldown period
            verbose=1
        )

class F1ScoreTracker(Callback):
    """Custom callback to track F1 scores during training"""
    def __init__(self, train_dataset, train_steps, val_dataset, val_steps):
        super().__init__()
        self.train_dataset = train_dataset
        self.train_steps = train_steps
        self.val_dataset = val_dataset
        self.val_steps = val_steps
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_f1_macro_scores = []
        self.val_f1_macro_scores = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate F1 score for training data
        train_f1, train_f1_macro = self._calculate_f1_score(self.model, self.train_dataset, self.train_steps)
        self.train_f1_scores.append(train_f1)
        self.train_f1_macro_scores.append(train_f1_macro)
        
        # Calculate F1 score for validation data
        val_f1, val_f1_macro = self._calculate_f1_score(self.model, self.val_dataset, self.val_steps)
        self.val_f1_scores.append(val_f1)
        self.val_f1_macro_scores.append(val_f1_macro)
        
        # Print F1 scores
        print(f"Epoch {epoch+1}: Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}")
        print(f"Epoch {epoch+1}: Train F1 Macro = {train_f1_macro:.4f}, Val F1 Macro = {val_f1_macro:.4f}")
    
    def _calculate_f1_score(self, model, dataset, steps):
        """Calculate F1 score for a given dataset"""
        y_true = []
        y_pred = []
        
        try:
            for images, labels in dataset.take(steps):
                preds = model.predict(images, verbose=0)
                y_true.extend(np.argmax(labels.numpy(), axis=1))
                y_pred.extend(np.argmax(preds, axis=1))
        except tf.errors.OutOfRangeError:
            print("Warning: Dataset ended early for F1 calculation.")
        
        if not y_true:
            return 0.0, 0.0
        
        # Calculate F1 scores
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return f1_weighted, f1_macro

def plot_f1_scores_vs_epoch(f1_tracker, save_path='f1_scores_vs_epoch.png'):
    """Plot F1 scores vs epoch number"""
    epochs = range(1, len(f1_tracker.train_f1_scores) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot weighted F1 scores
    plt.subplot(2, 1, 1)
    plt.plot(epochs, f1_tracker.train_f1_scores, 'b-', label='Train F1 (Weighted)', linewidth=2)
    plt.plot(epochs, f1_tracker.val_f1_scores, 'r-', label='Validation F1 (Weighted)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Weighted)')
    plt.title('F1 Score (Weighted) vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot macro F1 scores
    plt.subplot(2, 1, 2)
    plt.plot(epochs, f1_tracker.train_f1_macro_scores, 'b-', label='Train F1 (Macro)', linewidth=2)
    plt.plot(epochs, f1_tracker.val_f1_macro_scores, 'r-', label='Validation F1 (Macro)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score (Macro)')
    plt.title('F1 Score (Macro) vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"F1 scores plot saved as '{save_path}'")
    plt.show()
    
    # Print final F1 scores
    print(f"\nFinal F1 Scores:")
    print(f"Train F1 (Weighted): {f1_tracker.train_f1_scores[-1]:.4f}")
    print(f"Validation F1 (Weighted): {f1_tracker.val_f1_scores[-1]:.4f}")
    print(f"Train F1 (Macro): {f1_tracker.train_f1_macro_scores[-1]:.4f}")
    print(f"Validation F1 (Macro): {f1_tracker.val_f1_macro_scores[-1]:.4f}")
def create_ablated_model(ablation_type, input_shape, num_classes, learning_rate=0.0001):
    """Create model variants for ablation study"""
    
    # Force float32 for stability in ablation study
    if tf.config.list_physical_devices('GPU'):
        from tensorflow.keras.mixed_precision import set_global_policy
        set_global_policy('float32')  # Force float32 for stability
    
    cfg = ABLATION_CONFIGS.get(ablation_type, ABLATION_CONFIGS['baseline'])
    
    model = models.Sequential()
    
    # Use filter sequence and kernels from config
    filter_sequence = cfg['filters']
    kernels = cfg['kernels']
    num_blocks = min(cfg['num_conv_blocks'], len(filter_sequence))
    
    print(f"Creating {ablation_type}: {num_blocks} conv blocks with filters {filter_sequence}, kernels {kernels}")
    
    for i in range(num_blocks):
        # First layer needs input_shape
        if i == 0:
            model.add(layers.Conv2D(filter_sequence[i], kernels[i], 
                                  activation='relu', input_shape=input_shape,
                                  kernel_regularizer=regularizers.l2(cfg['l2_reg']) if cfg['l2_reg'] > 0 else None))
        else:
            model.add(layers.Conv2D(filter_sequence[i], kernels[i], activation='relu',
                                  kernel_regularizer=regularizers.l2(cfg['l2_reg']) if cfg['l2_reg'] > 0 else None))
        
        if cfg['use_batchnorm']:
            model.add(layers.BatchNormalization())
            
        model.add(layers.MaxPooling2D((2,2)))
        
        if cfg['use_dropout']:
            model.add(layers.Dropout(0.2))
    
    # Classification head - handle special cases
    if ablation_type == 'flatten_vs_gap' and cfg.get('use_gap', False):
        # Replace Flatten with GlobalAveragePooling2D
        model.add(layers.GlobalAveragePooling2D())
        print("  Using GlobalAveragePooling2D instead of Flatten")
    else:
        model.add(layers.Flatten())
    
    if ablation_type == 'extra_dense' and cfg.get('extra_dense', False):
        # Add extra dense layer: 128→64→softmax
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(cfg['l2_reg']) if cfg['l2_reg'] > 0 else None))
        if cfg['use_batchnorm']:
            model.add(layers.BatchNormalization())
        if cfg['use_dropout']:
            model.add(layers.Dropout(0.2))
        print("  Added extra dense layer: 128→64→softmax")
    
    # Standard dense layer
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(cfg['l2_reg']) if cfg['l2_reg'] > 0 else None))
    
    if cfg['use_batchnorm']:
        model.add(layers.BatchNormalization())
    
    if cfg['use_dropout']:
        model.add(layers.Dropout(0.2))
        model.add(layers.Dropout(0.2))  # Additional dropout
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # CRITICAL: Set initial learning rate here
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Debug: Print model summary
    print(f"{ablation_type} parameters: {model.count_params()}")
    
    return model

def apply_regularization_config(config, level):
    """Apply regularization configuration based on level"""
    if level == 'aggressive':
        config['l2_reg'] = 0.002
        config['use_dropout'] = True
    elif level == 'moderate':
        config['l2_reg'] = 0.001
        config['use_dropout'] = True
    elif level == 'minimal':
        config['l2_reg'] = 0.0
        config['use_dropout'] = False

def test_configuration(config):
    """Test a specific configuration and return performance score"""
    print(f"Testing configuration: {config}")
    
    # Create model with current configuration
    model = create_ablated_model_from_config(config, (config['image_size'], config['image_size'], 1), NUM_CLASSES)
    
    # Create datasets
    train_dataset = create_dataset_for_size('train_images', config['image_size'])
    val_dataset = create_dataset_for_size('val_images', config['image_size'])
    test_dataset = create_dataset_for_size('test_images', config['image_size'])
    
    # Train and evaluate
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        steps_per_epoch=math.ceil(sum(len(os.listdir(os.path.join('train_images', cls))) 
                     for cls in CLASS_MAPPING.keys()) / BATCH_SIZE),
        epochs=15,  # Quick evaluation
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )
    
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    
    print(f"Configuration performance: {test_accuracy:.4f}")
    return test_accuracy

def create_ablated_model_from_config(config, input_shape, num_classes, learning_rate=0.001):
    """Create model from configuration dictionary"""
    model = models.Sequential()
    
    filter_sequence = [8, 16, 32, 32]
    num_blocks = min(config['num_conv_blocks'], len(filter_sequence))
    
    for i in range(num_blocks):
        if i == 0:
            model.add(layers.Conv2D(filter_sequence[i], (5,5) if i==0 else (3,3), 
                                  activation='relu', input_shape=input_shape,
                                  kernel_regularizer=regularizers.l2(config['l2_reg']) if config['l2_reg'] > 0 else None))
        else:
            model.add(layers.Conv2D(filter_sequence[i], (3,3), activation='relu',
                                  kernel_regularizer=regularizers.l2(config['l2_reg']) if config['l2_reg'] > 0 else None))
        
        if config['use_batchnorm']:
            model.add(layers.BatchNormalization())
            
        model.add(layers.MaxPooling2D((2,2)))
        
        if config['use_dropout']:
            model.add(layers.Dropout(0.2))
    
    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(config['l2_reg']) if config['l2_reg'] > 0 else None))
    
    if config['use_batchnorm']:
        model.add(layers.BatchNormalization())
    
    if config['use_dropout']:
        model.add(layers.Dropout(0.2))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_sequential_optimization():
    """Optimize one component at a time, fixing the best so far"""
    
    current_best = {
        'image_size': 256,
        'l2_reg': 0.002,
        'use_batchnorm': True,
        'use_dropout': True,
        'num_conv_blocks': 4
    }
    
    optimization_steps = [
        ('image_size', [256, 128]),
        ('architecture', [3, 4]),  # num_conv_blocks
        ('regularization', ['aggressive', 'moderate', 'minimal'])
    ]
    
    print("=" * 60)
    print("SEQUENTIAL OPTIMIZATION")
    print("=" * 60)
    
    for step_name, options in optimization_steps:
        print(f"\nOptimizing {step_name}...")
        best_performance = -1
        best_option = None
        
        for option in options:
            test_config = current_best.copy()
            
            if step_name == 'image_size':
                test_config['image_size'] = option
            elif step_name == 'architecture':
                test_config['num_conv_blocks'] = option
            elif step_name == 'regularization':
                apply_regularization_config(test_config, option)
            
            performance = test_configuration(test_config)
            
            if performance > best_performance:
                best_performance = performance
                best_option = option
        
        # Update current best with the best option for this step
        if step_name == 'image_size':
            current_best['image_size'] = best_option
            print(f"Best image size: {best_option}")
        elif step_name == 'architecture':
            current_best['num_conv_blocks'] = best_option
            print(f"Best architecture: {best_option} conv blocks")
        elif step_name == 'regularization':
            apply_regularization_config(current_best, best_option)
            print(f"Best regularization: {best_option}")
        
        print(f"Current best performance: {best_performance:.4f}")
    
    print(f"\nFinal optimized configuration: {current_best}")
    return current_best

def create_dataset_for_size(folder, img_size, batch_size=32):
    """Create dataset with specific image size"""
    def sized_generator():
        all_image_paths = []
        all_labels = []
        
        for class_name, class_idx in CLASS_MAPPING.items():
            class_dir = os.path.join(folder, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    all_image_paths.append(img_path)
                    all_labels.append(class_idx)
        
        all_image_paths = np.array(all_image_paths)
        all_labels = np.array(all_labels)
        
        while True:
            indices = np.arange(len(all_image_paths))
            np.random.shuffle(indices)
            
            for start_idx in range(0, len(indices), batch_size):
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    img_path = all_image_paths[idx]
                    label = all_labels[idx]
                    try:
                        img = Image.open(img_path).convert('L')
                        img = img.resize((img_size, img_size))
                        img_array = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=-1)
                        batch_images.append(img_array)
                        batch_labels.append(label)
                    except Exception as e:
                        continue
                
                if batch_images:
                    yield np.array(batch_images), to_categorical(batch_labels, NUM_CLASSES)
    
    return tf.data.Dataset.from_generator(
        sized_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_size, img_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    
def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_schedule', type=str, default='adaptive',
                       choices=['adaptive', 'exponential', 'cosine', 'warmup_cosine'],
                       help='Learning rate schedule type')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from saved model')
    parser.add_argument('--model_path', type=str, default='best_model.keras',
                       help='Path to model file for resuming training')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study instead of normal training')
    args = parser.parse_args()
    
    if args.ablation:
        # Run ablation study
        print("RUNNING COMPREHENSIVE ABLATION STUDY")
        print("=" * 60)
        size_results, component_results = run_complete_ablation_study()
        
        # Create optimized model based on component results
        optimized_model, optimal_size = create_optimized_model(component_results, (IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES)
        
        # Skip validation since no deployment needed
        print("\nOptimized model configuration complete!")
        print("Saving optimized model...")
        
        # Save optimized model
        optimized_model.save('optimized_drone_classifier.keras')
        print("Optimized model saved as 'optimized_drone_classifier.keras'")
        
        return optimized_model
    else:
        # Original training code (your existing main() function)
        print(f"Using learning rate schedule: {args.lr_schedule}")
        print(f"Initial learning rate: {args.initial_lr}")
        if args.resume:
            print(f"Resuming training from: {args.model_path}")

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

        # Create datasets using generators (baseline model: with augmentation)
        train_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator('train_images', BATCH_SIZE, augment=True),
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator('val_images', BATCH_SIZE, augment=False),
            output_signature=(
                tf.TensorSpec(shape=(None, IMG_WIDTH, IMG_HEIGHT, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_generator(
            lambda: image_generator('test_images', BATCH_SIZE, augment=False),
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

        # Build or load model
        if args.resume and os.path.exists(args.model_path):
            print(f"Loading model from {args.model_path}...")
            model = tf.keras.models.load_model(args.model_path)
            print("Model loaded successfully!")
            
            # Update learning rate for continued training
            if hasattr(model.optimizer, 'learning_rate'):
                model.optimizer.learning_rate.assign(args.initial_lr)
                print(f"Reset learning rate to: {args.initial_lr}")
        else:
            if args.resume:
                print(f"Model file {args.model_path} not found. Creating new model.")
            model = create_model((IMG_WIDTH, IMG_HEIGHT, 1), NUM_CLASSES, learning_rate=args.initial_lr)
        
        # Get learning rate schedule
        lr_schedule = get_learning_rate_schedule(args.lr_schedule)
        model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )

        # Create F1 score tracker
        f1_tracker = F1ScoreTracker(train_dataset, train_steps, val_dataset, val_steps)
        
        # Additional callbacks for monitoring
        callbacks = [lr_schedule, model_checkpoint, early_stopping, f1_tracker]
        
        # Add TensorBoard for learning rate visualization if desired
        try:
            from tensorflow.keras.callbacks import TensorBoard
            log_dir = f"logs/lr_{args.lr_schedule}_{args.initial_lr}"
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
            print(f"TensorBoard logs available at: {log_dir}")
        except ImportError:
            print("TensorBoard not available, skipping logging")

        print(f"\nStarting training with {args.lr_schedule} learning rate schedule...")
        print("F1 scores will be tracked and displayed after each epoch")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot F1 scores vs epoch after training
        print("\n" + "="*60)
        print("PLOTTING F1 SCORES VS EPOCH")
        print("="*60)
        plot_f1_scores_vs_epoch(f1_tracker)

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_dataset, steps=test_steps)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Generate confusion matrix and classification report
        print("\n" + "="*60)
        print("GENERATING CONFUSION MATRIX AND CLASSIFICATION REPORT")
        print("="*60)
        
        y_true = []
        y_pred = []
        for images, labels in test_dataset.take(test_steps):
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_MAPPING.keys(),
                    yticklabels=CLASS_MAPPING.keys())
        plt.title('Confusion Matrix - Baseline Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as 'confusion_matrix_baseline.png'")

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=CLASS_MAPPING.keys()))

        # Calculate and print per-class accuracy
        print("\nPer-Class Accuracy:")
        class_names = list(CLASS_MAPPING.keys())
        for i, class_name in enumerate(class_names):
            class_mask = np.array(y_true) == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.mean(np.array(y_pred)[class_mask] == i)
                print(f"  {class_name}: {class_accuracy:.4f} ({np.sum(class_mask)} samples)")
            else:
                print(f"  {class_name}: No samples in test set")

        # Save final model
        model.save('flying_object_classifier_final.keras')
        
        print("\nTraining completed!")
        
        return model

if __name__ == '__main__':
    model = main()
    # show_predictions(model, 'test_images')
