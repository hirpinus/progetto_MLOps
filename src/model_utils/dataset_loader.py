import kagglehub
import pandas as pd
from datasets import load_dataset
import os
import subprocess
import json
from pathlib import Path

def download_kaggle_dataset(dataset_name, output_path, output_filename=None):
    """
    Scarica un dataset da Kaggle
    
    :param dataset_name: Nome del dataset su Kaggle (es. "kazanova/sentiment140")
    :param output_path: Percorso dove salvare il dataset
    :param output_filename: Nome del file di output (opzionale)
    :return: Path al file scaricato o None in caso di errore
    """
    try:
        # Crea la directory se non esiste
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Download dataset {dataset_name}...")
        
        # Scarica in una directory temporanea
        temp_dir = Path(output_path) / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Usa kaggle API per scaricare il dataset
        command = f"kaggle datasets download -d {dataset_name} -p {temp_dir} --unzip"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Trova il file scaricato
            downloaded_files = list(temp_dir.glob('*'))
            if not downloaded_files:
                print("Nessun file trovato dopo il download")
                return False
                
            # Se specificato output_filename, rinomina il file
            if output_filename:
                source_file = downloaded_files[0]  # prendi il primo file
                dest_file = Path(output_path) / output_filename
                source_file.rename(dest_file)
                print(f"Dataset rinominato e salvato come: {dest_file}")
            else:
                # Sposta semplicemente il file nella directory di output
                for file in downloaded_files:
                    dest_file = Path(output_path) / file.name
                    file.rename(dest_file)
                    print(f"Dataset salvato come: {dest_file}")
            
            # Pulisci la directory temporanea
            temp_dir.rmdir()
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione del comando kaggle: {str(e)}")
        print("Assicurati di aver configurato correttamente le API Kaggle")
        return False
    except Exception as e:
        print(f"Errore generico: {str(e)}")
        return False

def dataset_preprocess(path_read,filename_r, path_write, filename_w,sample_size=None):

    path_input_file = os.path.join(path_read, filename_r)
    path_output_file = os.path.join(path_write, filename_w)

    #Leggo il CSV originale
    df = pd.read_csv(path_input_file, encoding='latin-1', header=None)
    #Creo nuove colonne solo con testo e label mappata
    df_clean = pd.DataFrame()
    df_clean["text"] = df[5]
    df_clean["label"] = df[0].map({0: 0, 2: 1, 4: 2})
    df_clean = df_clean[df_clean["label"].notnull()]   #Rimuovi eventuali righe senza label

    # Se specificato sample_size, seleziona solo una porzione casuale del dataset
    if sample_size is not None:
        df_clean = df_clean.sample(n=min(sample_size, len(df_clean)), random_state=42)

    #rimuovo vecchio csv se presente
    if os.path.exists(path_output_file):
        try:
            os.remove(path_output_file)
            print(f"File di output precedente rimosso: {path_output_file}")
        except OSError as e:
            print(f"ERRORE: Impossibile rimuovere il file di output {path_output_file}. Processo interrotto. Dettagli: {e}")
            return False
        
    #Salvo
    df_clean.to_csv(path_output_file, index=False, encoding='utf-8')

def load_and_split_dataset(file_path, filename, test_size=0.2, seed=42, label_column='label',sample_size=None):
    """
    Carica un dataset da un file CSV, lo shuffla e lo divide in set di training e test.

    :param file_path: Il percorso (stringa) al file CSV da caricare.
    :param test_size: La frazione (float) da usare per il set di test (es. 0.2 per 20%).
    :param seed: Il seed (int) per la casualità, per garantire la riproducibilità.
    :param label_column: Il nome della colonna (stringa) da usare per la stratificazione.
    :return: Un oggetto DatasetDict contenente 'train' e 'test'.
    """
    path_file=f"{file_path}/{filename}"

    # 1. Caricamento del dataset dal file CSV
    try:
        dataset = load_dataset('csv', data_files=path_file)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato al percorso: {file_path}")
        return None
    
    #Mi assicuro che il dataset abbbia la split 'train' (tipico per i file singoli)
    if 'train' not in dataset:
        print("ERRORE: La split 'train' non è stata trovata nel dataset caricato.")
        return None
    
     #Se specificato sample_size, seleziona solo una porzione del dataset
    if sample_size is not None:
        dataset['train'] = dataset['train'].shuffle(seed=seed).select(range(sample_size))

    # Verifica che la colonna label esista
    if label_column not in dataset['train'].column_names:
        print(f"ERRORE: La colonna '{label_column}' non è presente nel dataset.")
        return None

    # Converti la colonna label in ClassLabel in modo che sia possibile usare stratify_by_column
    try:
        dataset = dataset.class_encode_column(label_column)
    except Exception as e:
        print(f"ERRORE nella codifica della colonna '{label_column}' per stratificazione: {e}")
        return None

    # 2. Divisione (Split) in Training e Test
    # train_test_split shuffla automaticamente prima della divisione.
    try:
        splitted = dataset['train'].train_test_split(
            test_size=test_size, 
            seed=seed, 
            stratify_by_column=label_column
        )
        
        return splitted # Restituisce l'oggetto DatasetDict
        
    except ValueError as e:
        # Cattura l'errore se la colonna di stratificazione non esiste o altri problemi
        print(f"ERRORE nella divisione (split): {e}")
        print(f"Assicurati che la colonna '{label_column}' esista nel dataset e contenga classi valide per la stratificazione.")
        return None

#dataset_preprocess("/home/codespace/.cache/kagglehub/datasets/kazanova/sentiment140/versions/2")