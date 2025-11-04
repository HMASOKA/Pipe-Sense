# 🧠 PipeSense: AI-Powered Lead Pipe Detection & Home Safety Chatbot

PipeSense is an AI-driven public-health tool that helps users identify potential **lead contamination** in their homes.  
It combines a **TensorFlow CNN model**, a **FastAPI backend**, a **Dialogflow chatbot**, and a **Hugging Face Space** for visual inference.

---

## 🌍 Project Overview

PipeSense provides two ways to detect lead exposure risk:
1. **Address-based detection** – Users type an address, and the system checks a MySQL database for known lead pipe records.  
2. **Image-based detection** – Users upload a photo of their home’s pipes, and a CNN model predicts whether the pipe is **Lead**, **Non-Lead**, or **Other**.

---

## ⚙️ Pipeline Architecture

[ User ]
│
▼
[ Dialogflow Chatbot ]
│
▼
[ FastAPI Webhook Service ]
│ ├── Text lookup → MySQL database
│ └── Image upload → Hugging Face Space (CNN inference)
▼
[ TensorFlow CNN Model hosted on Hugging Face Spaces ]
│
▼
[ Response: Lead / Non-Lead / Not Found ]


---

## 💾 Data Pipeline

| Component | Purpose |
|------------|----------|
| **Address Database** (`sakila.address`, `sampled`) | Contains address and pipe material (Lead / Non-Lead) data |
| **Training Dataset** | Folder-based image dataset (`train/Lead`, `train/Non-Lead`, `train/Other`) |
| **Data Loader** | Uses `tf.keras.utils.image_dataset_from_directory` for batching and preprocessing |
| **Dataset Split** | `train`, `validation`, and `test` created via `.take()` and `.skip()` |

---

## 🧠 Model Development

**Framework:** TensorFlow / Keras  
**Model Type:** Convolutional Neural Network (CNN)

### Model Architecture
```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

Compilation & Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=30)
model.save("lead_pipe_cnn.h5")

💬 Chatbot Integration (Dialogflow)

Intent: Check Home Safety – Get Address

User Input	System Behavior
“Check 320 Brest Avenue”	FastAPI checks MySQL and returns lead status
“Is my home safe?”	Dialogflow triggers same intent
“I want to check a pipe image”	Bot responds with Hugging Face upload link

Example Responses
⚠️ Your area '320 Brest Avenue' has been reported with Lead pipes.
✅ '145-53 7 Avenue' is recorded as Non-Lead.
❓ No match found for 'Oak Street'.

📈 Outcome

✅ Fully functional AI chatbot for home safety queries
✅ Image-based lead pipe detection with CNN
✅ Cloud-deployed public interface via Hugging Face
✅ Low-cost, open-source, and reproducible pipeline

🧑‍💻 Author

Happymore Masoka
🎓 M.Sc. Data Science | Pace University
📧 happymoremasoka@gmail.com

🔗 Hugging Face Space – PipeSense

🏁 License

This project is released under the MIT License.
You are free to use, modify, and distribute with proper attribution.
