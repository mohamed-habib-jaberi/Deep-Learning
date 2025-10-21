# 🧠 CNN with Transfer Learning — Feature Extraction vs Fine-Tuning

This project explains how to apply **Transfer Learning** using a **Convolutional Neural Network (CNN)** for image classification.

We explore two main strategies:

- **Feature Extraction**  
- **Fine-Tuning**

---

## 📘 What is Transfer Learning?

**Transfer Learning** means reusing a **pre-trained model** (for example, EfficientNet, ResNet, or MobileNet) that has already learned useful visual features from a large dataset such as *ImageNet*, and adapting it to a **new, smaller dataset**.

**Example:**  
Classifying 10 food categories using a limited number of images.

---

## ⚙️ 1. Feature Extraction

In **Feature Extraction**, we use the pre-trained model as a **fixed feature extractor**.  
All convolutional layers are **frozen** (not trainable), and we add a new classification head on top.

### 🧩 Example Code

```python
import tensorflow as tf

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze all convolutional layers

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
