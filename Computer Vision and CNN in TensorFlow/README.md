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
```

### ✅ Advantages of Feature Extraction

| Advantage | Description |
|------------|-------------|
| ⚡ **Fast training** | Training is quick because most layers are frozen. |
| 🧠 **Efficient for small datasets** | Works well even with limited data. |
| 🧩 **Leverages pre-trained knowledge** | Uses general features (edges, textures, shapes) learned from ImageNet. |
| 📉 **Reduces overfitting risk** | Fewer trainable parameters lower overfitting chances. |

### ⚠️ Limitations
- The model may not learn very specific patterns in your dataset.  
- Features remain generic and not fully adapted to your data.

---

## 🔧 2. Fine-Tuning

After initial training with **Feature Extraction**, we **unfreeze some of the top layers** of the base model and **train them again** with a smaller learning rate.  

This allows the model to **adjust its pre-trained weights** to better fit the new dataset and learn more specific features related to your task.

### 🧩 Example Code

```python
# Unfreeze the top layers of the base model
base_model.trainable = True

# Keep most layers frozen
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compile with a low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
