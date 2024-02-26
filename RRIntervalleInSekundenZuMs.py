import pandas as pd
import os

# Umwandlung von Sekunden in Millisekunden
def convert_seconds_to_milliseconds(seconds):
    return seconds * 1000

def process_csv_files(input_dir, output_dir):

    # Iterieration über alle CSV-Dateien
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(input_file_path)

            # Extraktion der relevanten Spalten (ibi_s)
            ibi_s_data = df['ibi_s']

            ibi_ms_data = ibi_s_data.apply(convert_seconds_to_milliseconds)

            # Erstellung der neuen Textdatei
            output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_ibis.txt")
            with open(output_file_path, 'w') as file:
                for ibi_ms in ibi_ms_data:
                    file.write(f"{ibi_ms}\n")

def process_bs_csv_files(input_dir, output_dir):

    # Iteration über alle CSV-Dateien
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(input_file_path)

            # Extraktion der relevanten Spalten
            bs_ms_data = df['Phone timestamp;RR-interval'].str.split(';').str[1]

            output_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_BS.txt")
            with open(output_file_path, 'w') as file:
                for bs_ms in bs_ms_data:
                    file.write(f"{bs_ms}\n")

if __name__ == "__main__":
    # Pfade zu den Eingabe- und Ausgabe-Verzeichnissen    
    input_directory = 'Daten\MMASH'
    output_directory = 'Daten\MMASH Data Umgewandelt'

    input_directory_bs = 'Daten\HRV_bipolarund_schizophren'
    output_directory_bs = 'Daten\HRV_BS'

    # Verarbeitung der CSV-Dateien und Erstellung der Textdateien
    process_csv_files(input_directory, output_directory)
    process_bs_csv_files(input_directory_bs, output_directory_bs)
