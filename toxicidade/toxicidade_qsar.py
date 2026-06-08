# -- coding: utf-8 --
"""
3_Toxicidade.py

Refatoração completa para atender aos Requisitos do Projeto Prático de Deep Learning:
QSAR baseado em imagens moleculares (CNN + Keras)
"""

import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ==========================================
# REQUISITO TÉCNICO 3: Fixar semente aleatória
# ==========================================
SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("Versão do TensorFlow:", tf.__version__)
print("GPU Disponível:", tf.config.list_physical_devices('GPU'))

# ==========================================
# Célula 2: Carregamento de Dados (Imagens)
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = 'Projeto2/dataset_Toxicidade/train/'
val_dir   = 'Projeto2/dataset_Toxicidade/val/'
test_dir  = 'Projeto2/dataset_Toxicidade/test/'

print("Carregando datasets de imagens de Toxicidade...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary')
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary')
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False)

# === GRANDE CORREÇÃO DA NORMALIZAÇÃO ===
# A normalização (1./255) do código antigo estava zerando as ativações do EfficientNet!
# O EfficientNet exige imagens no formato [0, 255] porque ele tem a normalização embutida dentro dele.
# Portanto, a camada de Rescaling vai ser colocada APENAS dentro do modelo Baseline.
print("Calculando pesos de classe dinamicamente para compensar desbalanceamento...")
train_labels = np.concatenate([y.numpy() for x, y in train_dataset], axis=0).flatten()
neg = len(train_labels[train_labels == 0])
pos = len(train_labels[train_labels == 1])
total = neg + pos
class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}
print(f"Pesos de Classe -> Não Tóxico (0): {class_weights[0]:.2f} | Tóxico (1): {class_weights[1]:.2f}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print("Dados do Toxicidade carregados com sucesso!")

# ==========================================
# Célula 3: Callbacks (Requisito 5)
# ==========================================
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# === MONITORAMENTO POR AUC EM VEZ DE LOSS ===
# Em datasets muito desbalanceados, val_loss engana o EarlyStopping. Vamos salvar o modelo pela AUC!
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', mode='max', patience=15, restore_best_weights=True
)

checkpoint_baseline = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_baseline_toxicidade.weights.h5', monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True)

checkpoint_finetuning = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_finetuning_toxicidade.weights.h5', monitor='val_auc', mode='max', save_best_only=True, save_weights_only=True)


# ==========================================
# Célula 4: Construção do Modelo Baseline
# ==========================================
def build_baseline_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Normalização específica do Baseline (Requisito 4: "justificar por que pré-treinado não usa")
        layers.Rescaling(1./255), 
        layers.RandomFlip("horizontal"), # Distorções levíssimas para evitar overfit
        layers.RandomRotation(0.05),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.45),

        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])
    return model

baseline_model = build_baseline_cnn()
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\n=== Treinamento do Modelo Baseline (Toxicidade) ===")
history_baseline = baseline_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=60,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint_baseline, time_callback]
)

baseline_epoch_times = time_callback.epoch_times
baseline_total_time = sum(baseline_epoch_times)

# ==========================================
# Célula 5: Modelo Fine-Tuning (EfficientNetB0)
# ==========================================
def build_finetuned_model(input_shape=(224, 224, 3)):
    # EfficientNetB0 aceita [0, 255] diretamente.
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.RandomFlip("horizontal"), 
        layers.RandomRotation(0.05),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return base_model, model

base_model, finetuning_model = build_finetuned_model()

finetuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

time_callback_ft = TimeHistory()

print("\n=== Estágio 1 (Warmup) do Fine-Tuning ===")
finetuning_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    class_weight=class_weights,
    callbacks=[time_callback_ft]
)

print("\n=== Estágio 2 (Fine-Tuning Ativo) ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

finetuning_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

history_finetuning = finetuning_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=45,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint_finetuning, time_callback_ft]
)

ft_epoch_times = time_callback_ft.epoch_times
ft_total_time = sum(ft_epoch_times)

# ==========================================
# Avaliação Final e Resultados
# ==========================================
baseline_model.load_weights('best_baseline_toxicidade.weights.h5')
finetuning_model.load_weights('best_finetuning_toxicidade.weights.h5')

loss_base, acc_base, auc_base = baseline_model.evaluate(test_dataset, verbose=0)
loss_ft, acc_ft, auc_ft = finetuning_model.evaluate(test_dataset, verbose=0)

y_true = np.concatenate([y for x, y in test_dataset], axis=0)
pred_base = baseline_model.predict(test_dataset, verbose=0)
pred_ft = finetuning_model.predict(test_dataset, verbose=0)

cm_base = confusion_matrix(y_true, (pred_base > 0.5).astype(int))
cm_ft = confusion_matrix(y_true, (pred_ft > 0.5).astype(int))

print("\n=== RESULTADOS BASELINE (Toxicidade) ===")
print(f"Acurácia Teste: {acc_base:.4f} | AUC-ROC Teste: {auc_base:.4f}")
print("Matriz de Confusão:\n", cm_base)

print("\n=== RESULTADOS FINE-TUNING (Toxicidade) ===")
print(f"Acurácia Teste: {acc_ft:.4f} | AUC-ROC Teste: {auc_ft:.4f}")
print("Matriz de Confusão:\n", cm_ft)

def plot_learning_curves(history, title):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    auc = history.history['auc']
    val_auc = history.history['val_auc']
    epochs_range = range(len(acc))

    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Treino')
    plt.plot(epochs_range, val_acc, label='Validação')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Acurácia')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Treino')
    plt.plot(epochs_range, val_loss, label='Validação')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Perda (Loss)')
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, auc, label='Treino')
    plt.plot(epochs_range, val_auc, label='Validação')
    plt.legend(loc='lower right')
    plt.title(f'{title} - AUC-ROC')
    
    plt.show()

plot_learning_curves(history_baseline, "CNN Baseline")
plot_learning_curves(history_finetuning, "Fine-Tuning EfficientNetB0")

acc_train_base = history_baseline.history['accuracy'][-1]
acc_train_ft = history_finetuning.history['accuracy'][-1]

data = {
    'Dataset': ['Toxicidade', 'Toxicidade'],
    'Modelo': ['Baseline (Do Zero)', 'Fine-tuning (EfficientNetB0)'],
    'Acurácia Treino': [f"{acc_train_base:.4f}", f"{acc_train_ft:.4f}"],
    'Acurácia Teste': [f"{acc_base:.4f}", f"{acc_ft:.4f}"],
    'AUC-ROC (Teste)': [f"{auc_base:.4f}", f"{auc_ft:.4f}"],
    'Tempo Médio/Época (s)': [f"{np.mean(baseline_epoch_times):.2f}", f"{np.mean(ft_epoch_times):.2f}"],
    'Tempo Total (s)': [f"{baseline_total_time:.2f}", f"{ft_total_time:.2f}"]
}

df_comparativo = pd.DataFrame(data)
print("\n", df_comparativo.to_string(index=False))