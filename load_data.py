import tensorflow as tf
import numpy as np

# ================================
# Utility Functions
# ================================

def normalize_images(images, mean, std):
    """
    Normalize images using given mean and standard deviation.
    """
    return (images - mean) / std

def random_crop_and_flip(image):
    """
    Apply random cropping and horizontal flipping for data augmentation.
    """
    # Pad to 40x40
    image = tf.image.resize_with_pad(image, 40, 40)
    # Random crop back to 32x32
    image = tf.image.random_crop(image, size=[32, 32, 3])
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    return image

def augment_training_data(X_train):
    """
    Apply data augmentation to training data: random crop and flip.
    """
    augmented_train = [
        random_crop_and_flip(img).numpy() for img in X_train
    ]
    return np.stack(augmented_train, axis=0)

# ================================
# CIFAR-10 Loader
# ================================

def load_cifar10():
    """
    Load and preprocess the CIFAR-10 dataset.
    Includes normalization and data augmentation for training data.
    """
    mean = np.array([125.3, 123.0, 113.9], dtype=np.float32)
    std = np.array([63.0, 62.1, 66.7], dtype=np.float32)

    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize data
    X_train = normalize_images(X_train.astype(np.float32), mean, std)
    X_test = normalize_images(X_test.astype(np.float32), mean, std)
    
    # Apply augmentation to training data
    X_train = augment_training_data(X_train)

    return X_train, y_train.reshape(-1), X_test, y_test.reshape(-1)

# ================================
# CIFAR-100 Loader
# ================================

def load_cifar100():
    """
    Load and preprocess the CIFAR-100 dataset.
    Includes normalization and data augmentation for training data.
    """
    mean = np.array([129.3, 124.1, 112.4], dtype=np.float32)
    std = np.array([68.2, 65.4, 70.4], dtype=np.float32)

    # Load CIFAR-100 dataset with fine labels
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')

    # Normalize data
    X_train = normalize_images(X_train.astype(np.float32), mean, std)
    X_test = normalize_images(X_test.astype(np.float32), mean, std)

    # Apply augmentation to training data
    X_train = augment_training_data(X_train)

    return X_train, y_train.reshape(-1), X_test, y_test.reshape(-1)

# ================================
# Dataset Loader Wrapper
# ================================

def load_data(dataset_name):
    """
    Load CIFAR-10 or CIFAR-100 dataset.
    """
    if dataset_name == 'cifar-10':
        return load_cifar10()
    elif dataset_name == 'cifar-100':
        return load_cifar100()
    else:
        raise ValueError("Dataset not supported. Use 'cifar-10' or 'cifar-100'.")