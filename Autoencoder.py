from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.layers import BatchNormalization

# Bildung eines Inquartilsbereichs um Ausreißer aus dem Trainingsdatenset zu entfernen
def removeOutliers(train_data):
    Q1 = train_data.quantile(0.25)
    Q3 = train_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return train_data[~((train_data < lower_bound) | (train_data > upper_bound)).any(axis=1)]

# Laden der Trainingsdaten und Testdaten und Entfernen der label Spalte im Testset für das Skalieren
df_train = pd.read_csv('healthy_hrv_metrics.csv')
df_no_outliers = removeOutliers(df_train)
df_test = pd.read_csv('testSet.csv')
if 'label' in df_test.columns:
    df_test.drop('label', axis=1, inplace=True)

# Skalierung der Trainingsdaten und Testdaten auf gleiche Weise
scaler = StandardScaler()
X_train = scaler.fit_transform(df_no_outliers)
X_test = scaler.transform(df_test)

# Aufteilung des Trainingsdatensets in Trainings- und Validierungsdatensets
X_train_split, X_validation_split = train_test_split(X_train, test_size=0.2, random_state=42)

# Definition des Autoencoder-Modells
input_size = X_train.shape[1]
input_layer = Input(shape=(input_size,))
encoder = Dense(256, activation='relu')(input_layer)
encoder = BatchNormalization()(encoder)
encoder = Dense(128, activation='relu')(encoder)
encoder = Dense(64, activation='relu')(encoder)
encoder = BatchNormalization()(encoder)
encoder = Dense(32, activation='relu')(encoder)
encoded = Dense(16, activation='relu')(encoder)
encoded = BatchNormalization()(encoded)
decoder = Dense(32, activation='relu')(encoded)
decoder = Dense(64, activation='relu')(decoder)
decoder = BatchNormalization()(decoder)
decoder = Dense(128, activation='relu')(decoder)
decoded = Dense(256, activation='relu')(decoder)
decoded = BatchNormalization()(decoded)
output_layer = Dense(input_size, activation='relu')(decoded)
autoencoder = Model(input_layer, output_layer)

# Kompilieren und Trainieren des Modells
autoencoder.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
autoencoder.fit(X_train, X_train, 
                epochs=100, 
                batch_size=128, 
                validation_split=0.2,
                callbacks=[early_stopping])

# Anzeigen der Trainings- und Validierungsverluste in Abhängigkeit der Epochen
history = autoencoder.history.history
train_loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'b', label='Trainingdaten Verlust')
plt.plot(epochs, val_loss, 'r', label='Validierungdaten Verlust')
plt.title('Trainingdaten and Validierungdaten Verlust')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.show()

# Berechnung des Rekonstruktionsfehlers auf dem Trainingsdatensatz
X_train_split_pred = autoencoder.predict(X_train_split)
train_split_loss = np.mean(np.abs(X_train_split - X_train_split_pred), axis=1)

# Berechnung des Rekonstruktionsfehlers auf den geteilten Testdaten
X_validation_split_pred = autoencoder.predict(X_validation_split)
validation_split_loss = np.mean(np.abs(X_validation_split - X_validation_split_pred), axis=1)

# Berechnung des Rekonstruktionsfehlers auf dem Testdatensatz
X_test_pred = autoencoder.predict(X_test)
test_loss = np.mean(np.abs(X_test - X_test_pred), axis=1)

#Schwellenwert als mittlerer Validierungsverlust
threshold = np.mean(validation_split_loss)

# Markieren von Anomalien in den Testdaten basierend auf dem Schwellenwert
anomalies_indices = np.where(test_loss > threshold)[0]
print(f"Anzahl der Anomalien: {len(anomalies_indices)}")

# Anzeigen der Indizes der Anomalien
print("Indizes der Anomalien:", anomalies_indices)

# Extrahieren der als Anomalien erkannten Zeilen aus dem ursprünglichen DataFrame
anomalies = df_test.iloc[anomalies_indices]

# Anzeigen der als Anomalien erkannten Werte
print("Als Anomalien erkannte Werte:")
print(anomalies)

# Klassifizieren der Vorhersagen auf dem Testdatensatz
classified_predictions = np.where(test_loss > threshold, 1, 0)
df_test = pd.read_csv('testSet.csv')
true_labels = df_test['label']

# Berechnung von True Positives (TP), False Positives (FP), False Negatives (FN)
TP = np.sum((true_labels == 1) & (classified_predictions == 1))
FP = np.sum((true_labels == 0) & (classified_predictions == 1))
FN = np.sum((true_labels == 1) & (classified_predictions == 0))

#Berechnung der korrekten Klassifizierungen und der Anzahl aller Samples
correct_predictions = np.sum(true_labels == classified_predictions)
total_samples = len(true_labels)

# Berechnung von Precision, Recall, F1-Score und Accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = correct_predictions / total_samples

# Ausgabe der Metriken
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1_score}')
print(f'Accuracy: {accuracy}')

# Anzeigen der Konfusionsmatrix
confusion_matrix = confusion_matrix(true_labels, classified_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negativ', 'positiv'], yticklabels=['negativ', 'positiv'])
plt.xlabel('Klassifiziert')
plt.ylabel('Tatsächlich')
plt.title('Konfusionsmatrix')
plt.show()
