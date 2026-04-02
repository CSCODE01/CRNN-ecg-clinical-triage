import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = '/kaggle/input/datasets/cherguisofiane/ptb-xl-ecg-cleaned/'
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
BATCH_SIZE = 64

def load_and_resplit_data():
    print("Loading Data Tensors from Kaggle...")
    X_train_old = np.load(os.path.join(DATA_PATH, 'X_train.npy'))
    y_train_old = np.load(os.path.join(DATA_PATH, 'y_train.npy'))
    X_val_old = np.load(os.path.join(DATA_PATH, 'X_val.npy'))
    y_val_old = np.load(os.path.join(DATA_PATH, 'y_val.npy'))
    X_test_old = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
    y_test_old = np.load(os.path.join(DATA_PATH, 'y_test.npy'))

    X_all = np.vstack((X_train_old, X_val_old, X_test_old))
    y_all = np.vstack((y_train_old, y_val_old, y_test_old))
    total_patients = X_all.shape[0]

    print(f"Re-splitting {total_patients} patients into exactly 80% | 10% | 10% ...")

    ten_percent_count = int(total_patients * 0.10)

    X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all, test_size=ten_percent_count, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=ten_percent_count, random_state=42)

    print("Split Successful!")
    print(f"Training (80%) : {X_train.shape[0]} patients")
    print(f"Validation (10%) : {X_val.shape[0]} patients")
    print(f"Testing (10%) : {X_test.shape[0]} patients")

    return X_train, X_val, X_test, y_train, y_val, y_test

def residual_block(x, filters, kernel_size=5):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def build_crnn_model(input_shape=(1000, 12), num_classes=5):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, filters=128)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = residual_block(x, filters=256)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(multi_label=True, name='auc')])
    return model

def train_and_evaluate_single(X_train, X_val, X_test, y_train, y_val, y_test):
    print(f"\n{'=' * 50}\nSTARTING CRNN TRAINING (80-10-10)\n{'=' * 50}")
    tf.keras.backend.clear_session()

    model = build_crnn_model()

    early_stop = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=35,
              batch_size=BATCH_SIZE,
              callbacks=[early_stop, reduce_lr], verbose=1)

    print("\nUnlocking the 10% Test Vault for final unbiased predictions...")
    final_probs = model.predict(X_test, verbose=0)
    y_pred_binary = (final_probs >= 0.50).astype(int)

    print("\n" + "=" * 65)
    print("--- CRNN CLINICAL EVALUATION REPORT ---")
    print("=" * 65)

    norm_index = 3
    true_healthy = y_test[:, norm_index]
    pred_healthy = y_pred_binary[:, norm_index]
    true_sick = 1 - true_healthy
    pred_sick = 1 - pred_healthy

    pathology_indices = [0, 1, 2, 4]
    true_pathologies = y_test[:, pathology_indices]
    pred_pathologies = y_pred_binary[:, pathology_indices]

    true_overlap = (np.sum(true_pathologies, axis=1) >= 2).astype(int)
    pred_overlap = (np.sum(pred_pathologies, axis=1) >= 2).astype(int)

    print("\n[Level 1] Triage Level (Sick vs. Healthy):")
    print(f"Overall Triage Accuracy                 : {accuracy_score(true_sick, pred_sick) * 100:.2f}%")
    print(f"Precision (Confidence in Sick Alert)    : {precision_score(true_sick, pred_sick) * 100:.2f}%")
    print(f"Recall/Sensitivity (Catching Sick)      : {recall_score(true_sick, pred_sick) * 100:.2f}%")
    print(f"Specificity (Clearing Healthy Patients) : {recall_score(true_healthy, pred_healthy) * 100:.2f}%")

    print("\n[Level 2] Specific Pathologies (Disease Detection):")
    print(classification_report(y_test, y_pred_binary, target_names=CLASSES, zero_division=0))

    print("\n[Level 3] Co-morbidities (Overlapping Diseases):")
    print("(Measuring the model's ability to detect a patient with two or more simultaneous diseases)")
    print(f"Overall Co-morbidity Accuracy           : {accuracy_score(true_overlap, pred_overlap) * 100:.2f}%")
    print(f"Precision (Confidence in Overlap)       : {precision_score(true_overlap, pred_overlap, zero_division=0) * 100:.2f}%")
    print(f"Recall (Catching Complex Cases)         : {recall_score(true_overlap, pred_overlap, zero_division=0) * 100:.2f}%")
    print("=" * 65)

    print("\nGenerating Confusion Matrices for the thesis...")
    mcm = multilabel_confusion_matrix(y_test, y_pred_binary)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, ax in enumerate(axes):
        sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'], ax=ax, annot_kws={"size": 14})
        ax.set_title(f'{CLASSES[i]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('crnn_matrices_80_10_10.png', dpi=300, bbox_inches='tight')
    print("Matrices saved successfully as 'crnn_matrices_80_10_10.png'!")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_resplit_data()
    train_and_evaluate_single(X_train, X_val, X_test, y_train, y_val, y_test)