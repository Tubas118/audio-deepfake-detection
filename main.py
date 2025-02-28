# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: audio-deepfake-detection
#     language: python
#     name: python3
# ---

## IMPORTANT: The export of "main.py" is only intended to allow easier review of changes before committing code and pushing to source control.
## The file is not used within the notebook.
import jupytext
jupytext.write(jupytext.read('./main.ipynb'), 'main.py')

# + colab={"base_uri": "https://localhost:8080/"} id="S9PlhJVOJNh5" outputId="a10320b8-268e-4276-d4c0-9b98288a87b7"
import MyConfig
    
config = MyConfig.MyConfig("config.yml")
print(f"sample rate: {config.sampleRate}")

if (config.vendor == "google-colab"):
    from google.colab import drive
    drive.mount(config.dataPath)

print(f"Active path: '{config.dataPath}' using '{config.vendor}'.")

# + id="sDbNKAWTPaV6"
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# + id="bxvClGw3Pc0B"
# Define paths and parameters
DATASET_PATH = config.trainDatasetPath
LABEL_FILE_PATH = config.trainLabelFilePath
NUM_CLASSES = config.numClasses
SAMPLE_RATE = config.sampleRate
DURATION = config.duration
N_MELS = config.numMels

readFileName = config.fullPath(LABEL_FILE_PATH)



# + id="UyeBVeXLSNQn"
labels = {}

with open(readFileName, 'r') as label_file:
    lines = label_file.readlines()

for line in lines:
    parts = line.strip().split()
    file_name = parts[1]
    label = 1 if parts[-1] == "bonafide" else 0
    labels[file_name] = label

X = []
y = []

max_time_steps = 109  # Define the maximum time steps for your model
datasetRoot = config.fullPath(DATASET_PATH)

for file_name, label in labels.items():
    file_path = os.path.join(datasetRoot, file_name + ".flac")

    # Load audio file using librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    X.append(mel_spectrogram)
    y.append(label)

# + id="kRaT9u43SWwZ"
X = np.array(X)
y = np.array(y)

X,y

# + id="hiuB3-tKYtaY"
y_encoded = to_categorical(y, NUM_CLASSES)


# + id="Qpd51oPQYzVX"
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y_encoded[:split_index], y_encoded[split_index:]

print(f"split_index: {split_index}")
print(f"X_train:     {X_train}")
print(f"X_val:       {X_val}")
print(f"y_train:     {y_train}")
print(f"y_val:       {y_val}")

# + id="GFWARwGbY1pX"
# Define CNN model architecture
input_shape = (N_MELS, X_train.shape[2], 1)  # Input shape for CNN (height, width, channels)
model_input = Input(shape=input_shape)

# + id="oWcZ9HdlY4EJ"
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(model_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
model_output = Dense(NUM_CLASSES, activation='softmax')(x)

# + id="Oi5J0oiGY6yw"
model = Model(inputs=model_input, outputs=model_output)

# + id="iTo-iOaYY9Ns"
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# + colab={"base_uri": "https://localhost:8080/"} id="3Cw81pgfY_kc" outputId="4f47d238-fa77-46f1-c7e0-b1bfb27da228"
# Train the Model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# + id="wovU4AJmZsjg"
# saving the model
model.save(config.modelName)

# + [markdown] id="lYbmD_fxRpVy"
# --------
# ## Visualisation

# + id="UO0AeZHnRpV0"
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# + id="xu9t_YBSRpV0"
# Load the model and preprocess test data (similar to training data preprocessing)
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

# + id="ds0qYDyVRpV1"
import MyConfig
    
config = MyConfig.MyConfig("config.yml")

# Define paths and parameters
TEST_DATASET_PATH = config.testDatasetPath
MODEL_PATH = config.modelName
SAMPLE_RATE = config.sampleRate
DURATION = config.duration
N_MELS = config.numMels
MAX_TIME_STEPS = config.maxTimeSteps

# + id="gSJ-2NRuRpV1"
# Load the saved model
model = load_model(MODEL_PATH)

# + id="VkurRue7RpV1" outputId="b5639724-6dc0-46b5-8fc0-9abcd2c385a4"
# Load and preprocess test data using librosa
X_test = []

test_files = os.listdir(TEST_DATASET_PATH)
for file_name in test_files:
    file_path = os.path.join(TEST_DATASET_PATH, file_name)

    # Load audio file using librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Extract Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < MAX_TIME_STEPS:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, MAX_TIME_STEPS - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :MAX_TIME_STEPS]

    X_test.append(mel_spectrogram)

# Convert list to numpy array
X_test = np.array(X_test)

# Predict using the loaded model
y_pred = model.predict(X_test)

# Convert probabilities to predicted classes
y_pred_classes = np.argmax(y_pred, axis=1)

y_pred

# + id="bOxGNkgwRpV2" outputId="d338933d-1e4e-4708-9820-15266b9a8baa"
# Get True Labels

# Path to the ASVspoof 2019 protocol file
PROTOCOL_FILE_PATH = "test_eval.txt"

# Dictionary to store true labels for each file
true_labels = {}

# Read the protocol file
with open(PROTOCOL_FILE_PATH, 'rb') as protocol_file:
    lines = protocol_file.read().decode('utf-8').splitlines()
    print(lines)

for line in lines:
    line = line.strip()  # Strip leading/trailing whitespace
    parts = line.split()
    if len(parts) > 1:  # Check if line has enough parts to extract label
        file_name = parts[0]
        label = parts[-1]  # Last part contains the label
        true_labels[file_name] = label

# Now 'true_labels' contains the true labels for each file
true_labels

# + id="zf4ItnseRpV3" outputId="0ef7b428-d11c-4848-90d9-7fbd4a112109"
y_true = np.array([1 if label == "bonafide" else 0 for label in true_labels.values()]) # y_true are the true labels for each file
y_true

# + id="oTlEMsUNRpV6" outputId="83538031-a3f3-42a2-cd3e-72ab875ac82b"
# CONFUSION MATRIX

cm = confusion_matrix(y_true, y_pred_classes)

# Display the confusion matrix
classes = ["spoof", "bonafide"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# + id="j_KmS4xCRpV6" outputId="a5892b27-a6be-4ce4-ab15-8d4ee0e219a3"
# ROC Curve

from sklearn.metrics import roc_curve, auc

# Predict using the loaded model
y_pred = model.predict(X_test)

# Get the predicted probabilities for the positive class
y_pred_prob = y_pred[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# + id="3NxmBscPRpV7"
# Precision-Recall Curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# + id="M_hxp0l6RpV8" outputId="9af9f5e1-ece6-446e-89c3-3770d83b6f05"
# Compute precision-recall curve and average precision score
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
avg_precision = average_precision_score(y_true, y_pred_prob)

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='darkorange', lw=2, label='Avg. Precision = %0.2f' % avg_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# + id="W33XXm38RpV8"
# Calibration Curve
from sklearn.calibration import calibration_curve

# + id="R2Day10ZRpV9" outputId="87a5ced3-1336-49e0-9f73-9b66e4d83bbf"
# Compute calibration curve
prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10)

# Plot calibration curve
plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc="lower right")
plt.show()

# + id="OZwEyXU6RpV9" outputId="552ec693-ea7b-459a-abe8-79c6588bb9b8"
# Plot bar chart of class distribution

import seaborn as sns
import matplotlib.pyplot as plt


LABELS = ['spoof', 'bonafide']

plt.figure(figsize=(6, 4))
sns.countplot(x=y_true, palette="Set2", hue=X, legend=False)
plt.xticks(ticks=[0, 1], labels=LABELS)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

# + id="yK7vymddRpV-" outputId="b76e39d0-1d32-491d-c21b-8df2f9318345"
# Visualising Mel Spectrogram

import os
import librosa.display

# Folder containing .flac audio files
folder_path = "TestEvaluation"

# Get a list of all .flac files in the folder
flac_files = [file for file in os.listdir(folder_path) if file.endswith(".flac")]

# Define the hop length
HOP_LENGTH = 512  # Adjust this value based on your needs

# Loop through each .flac file and visualize its Mel spectrogram
for flac_file in flac_files:
    audio_file_path = os.path.join(folder_path, flac_file)

    # Load the audio file using librosa
    audio, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE, duration=DURATION)

    # Calculate the Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {flac_file}')
    plt.show()


# + colab={"base_uri": "https://localhost:8080/", "height": 976} id="eDP0wapvRpV_" outputId="18d2eae6-cd3f-4ca2-ec6a-b013153dd81d"
from keras.utils import plot_model

plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
