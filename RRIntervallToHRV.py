import numpy as np
from numpy import trapz
import csv
import os
from scipy.signal import welch

# Funktion zum Lesen von RR-Intervallen aus einer Datei
def read_rr_intervals_from_file(file_path):
    with open(file_path, 'r') as file:
        # Lesen der Daten, Umwandlung in Float und Entfernen von leeren Zeilen und Nicht-Zahlen
        rr_intervals = [float(line.strip()) for line in file.readlines() if line.strip() and line.strip().replace('.', '', 1).isdigit()]
    return rr_intervals

# Methode zur Berechnung der Zeitdomäne-Metriken SDNN und RMSSD
def calculate_time_domain_metrics(rr_intervals):
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return {
        'SDNN': sdnn,
        'RMSSD': rmssd
    }

# Methode zur Berechnung der Frequenzdomäne-Metriken LF, HF und LF/HF Ratio
def calculate_frequency_domain_metrics(rri, fs=4):
    # Berechnung der Leistungsdichtespektrumschätzung mittels Welch-Verfahren
    fxx, pxx = welch(x=rri, fs=fs)
    # Definition von Frequenzbereichen für LF und HF
    condition_lf = (fxx >= 0.04) & (fxx < 0.15)
    condition_hf = (fxx >= 0.15) & (fxx < 0.4)
    lf = trapz(pxx[condition_lf], fxx[condition_lf]) 
    hf = trapz(pxx[condition_hf], fxx[condition_hf])
    results = {
        'LF': lf,
        'HF': hf,
        'LF/HF Ratio': (lf/hf)  # Berechnung der LF/HF-Ratio durch Division
    }
    return results, fxx, pxx

# Methode zum Erstellen der Datenzeilen der HRV-Metriken in eine CSV-Datei
def export_to_csv_with_hrv_metrics(filename, hrv_metrics_list):
    fieldnames = ['SDNN', 'RMSSD', 'LF', 'HF', 'LF/HF Ratio']  # Spaltennamen in der CSV-Datei
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for hrv_metrics in hrv_metrics_list:
            writer.writerow(hrv_metrics)

# Verzeichnispfade, in denen die Dateien liegen
directory_paths = ['Daten\PhysioNet', 'Daten\MMASH Data Umgewandelt']
#Fuer die Verarbeitung der Daten der bipolaren und schizophrenen Personen, muss hier folgende Zeile stattdessen genutzt werden
#directory_paths = ['Daten\HRV_BS']

# Liste für alle HRV-Metriken
all_hrv_metrics = []

# Berechnung der Metriken für jede Datei in jedem Verzeichnis
for directory_path in directory_paths:
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.txt')]
    
    for file_path in file_paths:
        rr_intervals = read_rr_intervals_from_file(file_path)
        
        # Aufteilung in 5-Minuten-Abschnitte
        segment_size = 5 * 60 
        num_segments = len(rr_intervals) // segment_size
        
        for i in range(num_segments):
            segment_rr_intervals = rr_intervals[i * segment_size : (i + 1) * segment_size]  # RR-Intervalle des aktuellen Abschnitts
            hrv_metrics = calculate_time_domain_metrics(segment_rr_intervals)
            fd_results, _, _ = calculate_frequency_domain_metrics(segment_rr_intervals)
            hrv_metrics.update(fd_results)
            all_hrv_metrics.append(hrv_metrics)

# CSV-Dateiname für den Export der Metriken
csv_filename = 'healthy_hrv_metrics.csv'
#Fuer die Erstellung der Datei der HRV-Metriken der bipolaren und schizophrenen Personen, muss hier folgende Zeile stattdessen genutzt werden
#csv_filename = 'BS_hrv_metrics.csv'

# Exportieren der HRV-Metriken in die CSV-Datei
export_to_csv_with_hrv_metrics(csv_filename, all_hrv_metrics)
