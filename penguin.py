import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
import csv
from geopy.distance import geodesic
import math
import pyproj
from mpl_toolkits.mplot3d import Axes3D
# -*- coding: utf-8 -*-

print("librairies importées")
print("Répertoire de travail actuel :", os.getcwd())

fichier = [[r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\chick-rearing', "A"],
           [r'D:\PenguinsData\Data_4\AdeliePenguin\Year2\chick-rearing', "A"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year1\Guard', "L"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year1\PostGuard', "L"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year1\Incubation', "L"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year2\Guard', "L"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year2\PostGuard', "L"],
           [r'D:\PenguinsData\Data_4\LittlePenguin\year2\Incubation', "L"]]



def g(lat) :
    #g : m/s²
    #lat : rad
    #g = 9.780327* (1 + 5.3024e-3 * np.sin(lat)**2 - 5.8e-6 * np.sin(2*lat)**2 - 3.086e-7 * 6384415)
    g = 9.7803267715 * (1 + 0.0052790414 * np.sin(lat)**2 + 0.0000232718* np.sin(lat)**4)

    return g

def ρ (T, S) :
    #T : K
    #S : ‰
    ρ = 1000 - 0.12 * T + 0.35 * S
    return ρ

def pressure_to_altitude(P, P0, lat, T, S):
    P, P0 = 100*P, 100*P0
    z = (P0 - P) / (ρ(T,S) * g(np.radians(lat)))
    return round(z,1)

def read_csv_with_different_encodings(file_path, encodings=['utf-8', 'latin1', 'iso-8859-1']):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"None of the encodings worked for file {file_path}")
    
def detect_separator(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ',' in first_line:
            return ','
        elif '	' in first_line :
            return '	'
        elif r'\t' in first_line :
            return r'\t'
        elif ' ' in first_line:
            return ' '
        
        else:
            raise ValueError(f"Can't recognize the separator in : {file_path}")

def display_column_names(file_path):
    try:
        sep = detect_separator(file_path)
        print("Separator: [", sep, "]")
        with open(file_path, 'r') as file:
            # Utiliser la bibliothèque `csv` pour lire uniquement les en-têtes et éviter les erreurs de tokenisation
            reader = csv.reader(file, delimiter=sep)
            headers = next(reader)
            print(f"Column names in {file_path}: {headers}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        
        
def get_column_types(csv_file):
    column_types = {}
    
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Initialize column types with None
        for col_name in reader.fieldnames:
            column_types[col_name] = None
        
        # Iterate over each row to determine column types
        for row in reader:
            for col_name, col_value in row.items():
                if col_value.strip() != '':
                    # Try to convert the value to int, float or str
                    try:
                        int(col_value)
                        column_types[col_name] = 'int'
                    except ValueError:
                        try:
                            float(col_value)
                            column_types[col_name] = 'float'
                        except ValueError:
                            column_types[col_name] = 'str'
    
    return column_types

def read_csv_lines(csv_file, start_line, end_line):
    line_data = []
    
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip lines until start_line
        for _ in range(start_line - 1):
            next(reader)
        
        # Read lines between start_line and end_line
        for i, row in enumerate(reader, start=start_line):
            if i > end_line:
                break
            
            line_data.append(row)
            print(line_data[-1])
    



def altitude_graph(filename):
    
    print(f"Used file name is : {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Extract the different values
    pressure_values = np.array(df['Pressure'].tolist()) * 100 #Pa
    timestamp_values = np.array(df['Timestamp'].tolist()) #str
    lon_values = np.array(df['location-lon'].tolist()) #
    lat_values = np.array(df['location-lat'].tolist()) #
    temperature_values = np.array(df['Temp. (?C)'].tolist()) + 273.15 #K
    
    
    #define the mean value of g
    lat_values_filtred = [lat for lat in lat_values if not np.isnan(lat)]
    lat_mean = np.mean(lat_values_filtred)
    g_mean = g(np.radians(lat_mean))
    print("Mean latitude value : ", lat_mean)
    print("Mean g value : ", g_mean)

    #define the value of ρ
    ρ_values =  np.array([ρ(temp, 34) for temp in temperature_values])

    

    # Convert time string format into float variables
    try:
        timestamp_values = [datetime.strptime(ts, '%d/%m/%Y %H:%M:%S.%f') for ts in timestamp_values]
    except ValueError:
        print("Error: Incorrect datetime format in the CSV file.")
        return

    # set the first timestamp at 0
    start_time = timestamp_values[0]
    time_in_hours = [(ts - start_time).total_seconds() / 3600 for ts in timestamp_values]

    pressure_values = np.array(pressure_values[1:])
    print("pressure value 0 : ", pressure_values[0])
    time_in_hours = time_in_hours[1:]
    P0 = min(pressure_values)

    z_values = []
    # Calcul de z_values
    for i in range (len(pressure_values)) :
        z_values.append(- (pressure_values[i] - P0) / (ρ_values[i] * g_mean) )

    # Tracer les valeurs
    plt.plot(time_in_hours, pressure_values * 1e-5, label='Experimental Pressure')
    plt.xlabel('t (h)')
    plt.ylabel('P (Bar)')
    plt.title('Experimental Bio-logging Pressure as function of time')
    plt.legend()
    plt.show()
    
    
    plt.plot(time_in_hours, z_values, label='Experimental Altitude')
    plt.xlabel('t (h)')
    plt.ylabel('z (m)')
    plt.title('Experimental Bio-logging Pressure as function of time')
    plt.legend()
    plt.show()
    
def altitude_graph_part(filename, a, b):
    
    print(f"Used file name is : {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Extract the different values
    pressure_values = np.array(df['Pressure'].tolist()) * 100 #Pa
    timestamp_values = np.array(df['Timestamp'].tolist()) #str
    lon_values = np.array(df['location-lon'].tolist()) #
    lat_values = np.array(df['location-lat'].tolist()) #
    temperature_values = np.array(df['Temp. (?C)'].tolist()) + 273.15 #K
    
    
    #define the mean value of g
    lat_values_filtred = [lat for lat in lat_values if not np.isnan(lat)]
    lat_mean = np.mean(lat_values_filtred)
    g_mean = g(np.radians(lat_mean))
    print("Mean latitude value : ", lat_mean)
    print("Mean g value : ", g_mean)

    #define the value of ρ
    ρ_values =  np.array([ρ(temp, 34) for temp in temperature_values])

    

    # Convert time string format into datetime objects
    try:
        timestamp_values = [datetime.strptime(ts, '%d/%m/%Y %H:%M:%S.%f') for ts in timestamp_values]
    except ValueError:
        print("Error: Incorrect datetime format in the CSV file.")
        return

    # Set the first timestamp at 0
    start_time = timestamp_values[a]
    time_in_minutes = [(ts - start_time).total_seconds() / 60 for ts in timestamp_values]

    pressure_values = np.array(pressure_values[1:])
    print("pressure value 0 : ", pressure_values[0])
    time_in_minutes = time_in_minutes[1:]
    P0 = min(pressure_values)

    z_values = []
    # Calcul de z_values
    for i in range (len(pressure_values)) :
        z_values.append(- (pressure_values[i] - P0) / (ρ_values[i] * g_mean) )

    # Tracer les valeurs
    plt.plot(time_in_minutes[a:b], pressure_values[a:b] * 1e-5, label='Experimental Pressure')
    plt.xlabel('t (h)')
    plt.ylabel('P (Bar)')
    plt.title('Experimental Bio-logging Pressure as function of time')
    plt.legend()
    plt.show()
    
    
    plt.plot(time_in_minutes[a:b], z_values[a:b], label='Experimental Altitude')
    plt.xlabel('t (min)')
    plt.ylabel('z (m)')
    plt.title('Experimental Bio-logging Pressure as function of time')
    plt.legend()
    plt.show()    

def filter_and_convert_csv(input_file, output_file):
    sep = detect_separator(input_file)
    print("Separator: [", sep, "]")
    df = pd.read_csv(input_file, sep=sep, on_bad_lines='skip')
    
    required_columns = ["Pressure"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    filtered_df = df.dropna(subset=required_columns)
    
    filtered_df.to_csv(output_file, index=False, sep=',')
    
def process_csv_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ignore hidden files or system files
        if filename.startswith('.'):
            continue
        
        if filename.endswith(".csv"):
            try:
                df = read_csv_with_different_encodings(file_path)
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"Error reading {file_path}: {e}")
                continue
            except PermissionError as e:
                print(f"Permission error for file {file_path}: {e}")
                continue
            
            df = df.iloc[:, :10]

            try:
                df.to_csv(file_path, index=False)
                print(f"Processed file: {file_path}")
            except PermissionError as e:
                print(f"Permission error while saving file {file_path}: {e}")

def create_altitude(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Ignore hidden or system files
        if filename.startswith('.'):
            continue
        
        if filename.endswith(".csv"):
            try:
                df = read_csv_with_different_encodings(file_path)
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                print(f"Error reading {file_path}: {e}")
                continue
            except PermissionError as e:
                print(f"Permission error for file {file_path}: {e}")
                continue

            # Extract the different values
            if 'location-lat' in df.columns and 'Temp. (?C)' in df.columns:
                lat_values = np.array(df['location-lat'].tolist())

                # Define the mean value of g
                lat_values_filtered = [lat for lat in lat_values if not np.isnan(lat)]
                lat_mean = np.mean(lat_values_filtered)

                # Check if 'Pressure' and 'Temp. (?C)' columns are present
                if 'Pressure' in df.columns and 'Temp. (?C)' in df.columns:
                    P0 = df['Pressure'].iloc[0]  # Take the first value of the 'Pressure' column

                    # Apply the transformation function with additional parameters
                    df['Altitude'] = df.apply(
                        lambda row: pressure_to_altitude(row['Pressure'], P0, lat_mean, row['Temp. (?C)'], 34), axis=1
                    )

                    # Drop the old 'Pressure' column and rename the new column
                    df.drop(columns=['Pressure'], inplace=True)

                    # Transform 'Timestamp' to day and seconds
                    if 'Timestamp' in df.columns:
                        try:
                            timestamps = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S.%f', dayfirst=True)
                            df['Day'] = timestamps.dt.date
                            df['Time_seconds'] = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
                        except ValueError as e:
                            print(f"Error parsing Timestamp in file {file_path}: {e}")
                            continue

                    # Drop the old 'Timestamp' column
                    df.drop(columns=['Timestamp'], inplace=True)

                    # Drop the 'Activity' column
                    if 'Activity' in df.columns:
                        df.drop(columns=['Activity'], inplace=True)

                    try:
                        df.to_csv(file_path, index=False)
                        print(f"Processed file: {file_path}")
                    except PermissionError as e:
                        print(f"Permission error while saving file {file_path}: {e}")
                else:
                    print(f"Missing 'Pressure' or 'Temp. (?C)' columns in file: {file_path}")
            else:
                print(f"Missing 'location-lat' or 'Temp. (?C)' columns in file: {file_path}")

def plot_altitude(filename):
    print(f"Used file name is : {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['Altitude', 'Day', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    z_values = np.array(df['Altitude'].tolist())  # Altitude in meters
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing Day column in file {filename}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_hours, z_values, label='Experimental Altitude')
    plt.xlabel('Time (hours)')
    plt.ylabel('Altitude (meters)')
    plt.title(f'Experimental Bio-logging Pressure as a Function of Time - {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.show()
                
def plot_altitude_part(filename, a, b):
    print(f"Used file name is: {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['Altitude', 'Day', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    z_values = np.array(df['Altitude'].tolist())  # Altitude in meters
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'])
    except ValueError as e:
        print(f"Error parsing Day column in file {filename}: {e}")
        return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())

    # Filter the values between a and b seconds
    mask = (total_elapsed_seconds >= a) & (total_elapsed_seconds <= b)
    filtered_z_values = z_values[mask]
    filtered_total_elapsed_seconds = total_elapsed_seconds[mask]
    time_values_minutes = filtered_total_elapsed_seconds / 60  # Convert total elapsed time to minutes

    # Extract the base name of the file (without path and extension)
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_minutes, filtered_z_values, label='Experimental Altitude')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (meters)')
    plt.title(f'Experimental Bio-logging Pressure as a Function of Time - {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_altitude_temperature(filename):
    print(f"Used file name is : {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['Altitude', 'Temperature', 'Day', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    z_values = np.array(df['Altitude'].tolist())  # Altitude in meters
    temp_values = np.array(df['Temperature'].tolist())
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing Day column in file {filename}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Altitude (meters)', color=color)
    ax1.plot(time_values_hours, z_values, color=color, label='Experimental Altitude')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Temperature (°C)', color=color)
    ax2.plot(time_values_hours, temp_values, color=color, label='Experimental Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Experimental Bio-logging Pressure and Altitude as a Function of Time - {base_filename}')
    fig.tight_layout()  # Adjust layout to fit titles, labels, etc. without overlap
    fig.legend()
    plt.grid(True)
    plt.show()
    
    
def process_csv_files_in_folder(folder_path):
    # List of encodings to try
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    
    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = None
            
            # Try reading the CSV file with different encodings
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break  # Exit the loop if reading is successful
                except Exception as e:
                    print(f"Error reading file {filename} with encoding {encoding}: {e}")
            
            if df is not None:
                try:
                    # Check if 'Altitude' column exists
                    if 'Altitude' in df.columns:
                        # Replace positive values with 0
                        df['Altitude'] = df['Altitude'].apply(lambda x: 0 if x > 0 else x)
                        
                        # Save the modified DataFrame, overwriting the original file
                        df.to_csv(file_path, index=False)
                        print(f"Processed and saved: {file_path}")
                    else:
                        print(f"'Altitude' column not found in file: {filename}")
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
            else:
                print(f"Failed to read file {filename} with any encoding.")

def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        print("Fichiers dans le répertoire :", files)
    except PermissionError:
        print(f"Permission denied: '{directory}'. Please check your directory permissions.")
    except FileNotFoundError:
        print(f"Directory not found: '{directory}'. Please check the path.")

def plot_position(csv_file):
    # Lire le fichier CSV avec un encodage spécifique
    if os.path.basename(csv_file).startswith('._'):
        print(f"Fichier ignoré: '{csv_file}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
    except PermissionError:
        print(f"Permission denied: '{csv_file}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{csv_file}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{csv_file}'. Please check the path and file name.")
        return
    
    # Afficher les colonnes pour débogage
    #print("Colonnes disponibles : ", df.columns)
    
    # Utiliser les noms de colonnes exacts obtenus de la vérification précédente
    required_columns = ['location-lat', 'location-lon', 'Altitude', 'Time_seconds']  # Mettre à jour selon les noms exacts
    
    # Vérifier si toutes les colonnes nécessaires sont présentes
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"La colonne requise '{col}' est manquante dans le fichier CSV.")
    
    # Filtrer les lignes avec des valeurs non nulles dans les colonnes requises
    df_filtered = df.dropna(subset=required_columns)
    
    # Récupérer les colonnes nécessaires
    latitude = df_filtered['location-lat']
    longitude = df_filtered['location-lon']
    altitude = df_filtered['Altitude']
    time_seconds = df_filtered['Time_seconds']
    
    # Convertir le temps en heures
    time_hours = time_seconds / 3600
    
    # Normaliser le temps pour la colormap
    norm = plt.Normalize(time_hours.min(), time_hours.max())
    cmap = plt.get_cmap('viridis')
    
    # Obtenir le nom du fichier pour le titre
    file_name = os.path.basename(csv_file)
    
    # Créer le plot avec l'évolution des couleurs en fonction du temps
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(longitude, latitude, c=time_hours, cmap=cmap, norm=norm, alpha=0.7)
    plt.colorbar(sc, label='Time (hours)')
    plt.title(f"Penguin's Movement Over Time - {file_name}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show() 
    
def plot_position_coordinates(csv_file):
    # Lire le fichier CSV avec un encodage spécifique
    print("on est là")
    if os.path.basename(csv_file).startswith('._'):
        print(f"Fichier ignoré: '{csv_file}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
    except PermissionError:
        print(f"Permission denied: '{csv_file}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{csv_file}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{csv_file}'. Please check the path and file name.")
        return
    
    # Afficher les colonnes pour débogage
    #print("Colonnes disponibles : ", df.columns)
    
    # Utiliser les noms de colonnes exacts obtenus de la vérification précédente
    required_columns = ['x', 'y', 'Altitude', 'Time_seconds']  # Mettre à jour selon les noms exacts
    
    # Vérifier si toutes les colonnes nécessaires sont présentes
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"La colonne requise '{col}' est manquante dans le fichier CSV.")
    
    # Filtrer les lignes avec des valeurs non nulles dans les colonnes requises
    df_filtered = df.dropna(subset=required_columns)
    
    # Récupérer les colonnes nécessaires
    y = np.array(df_filtered['y'])/1000
    x = np.array(df_filtered['x'])/1000
    time_seconds = df_filtered['Time_seconds']
    
    # Convertir le temps en heures
    time_hours = time_seconds / 3600
    
    # Normaliser le temps pour la colormap
    norm = plt.Normalize(time_hours.min(), time_hours.max())
    cmap = plt.get_cmap('viridis')
    
    # Obtenir le nom du fichier pour le titre
    file_name = os.path.basename(csv_file)
    
    # Créer le plot avec l'évolution des couleurs en fonction du temps
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x, y, c=time_hours, cmap=cmap, norm=norm, alpha=0.7)
    plt.colorbar(sc, label='Time (hours)')
    plt.title(f"Penguin's Movement Over Time - {file_name}")
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.grid(True)
    plt.show()

def check_columns(csv_file):
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
        print("Colonnes disponibles : ", df.columns)
    except PermissionError:
        print(f"Permission denied: '{csv_file}'. Please check your file permissions.")
    except UnicodeDecodeError:
        print(f"Error decoding file: '{csv_file}'. Please check the encoding.")
    except FileNotFoundError:
        print(f"File not found: '{csv_file}'. Please check the path and file name.")

def interpolate_missing_coordinates(csv_file):
    # Vérifier si le fichier commence par '._'
    if os.path.basename(csv_file).startswith('._'):
        print(f"Fichier ignoré: '{csv_file}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
    except PermissionError:
        print(f"Permission denied: '{csv_file}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{csv_file}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{csv_file}'. Please check the path and file name.")
        return
    
    # Afficher les colonnes pour débogage
    print("Colonnes disponibles : ", df.columns)
    
    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['location-lat', 'location-lon', 'Time_seconds']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"La colonne requise '{col}' est manquante dans le fichier CSV.")
    
    # Appliquer l'interpolation linéaire aux colonnes 'location-lat' et 'location-lon'
    df['location-lat'] = df['location-lat'].interpolate(method='linear')
    df['location-lon'] = df['location-lon'].interpolate(method='linear')
    
    # Remplir les éventuelles valeurs restantes en utilisant la méthode de remplissage avant et arrière
    df['location-lat'] = df['location-lat'].fillna(method='bfill').fillna(method='ffill')
    df['location-lon'] = df['location-lon'].fillna(method='bfill').fillna(method='ffill')
    
    # Vérifier s'il reste des valeurs manquantes
    if df[['location-lat', 'location-lon']].isnull().sum().sum() > 0:
        print("Il reste des valeurs manquantes après l'interpolation.")
    else:
        print("Toutes les valeurs manquantes ont été interpolées.")
    
    # Écrire les données interpolées dans le même fichier CSV
    df.to_csv(csv_file, index=False, encoding='latin1')
    
    return df

def clean_index_temperature(file_path):
    
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    # Lire le fichier CSV avec l'encodage 'latin1'
    df = pd.read_csv(file_path, encoding='latin1')
    
    # Remplacer la colonne 'TagID' par 'Index' et renommer 'Temp. (?C)' en 'Temperature'
    df.rename(columns={'TagID': 'Index', 'Temp. (?C)': 'Temperature'}, inplace=True)
    
    # Remplacer les valeurs de la colonne 'Index' par des entiers allant de 0 au nombre de lignes dans le fichier
    df['Index'] = range(len(df))
    
    # Sauvegarder le fichier CSV modifié (écrase le fichier original)
    df.to_csv(file_path, index=False, encoding='utf-8')


def validate_crs(crs_code):
    try:
        crs = pyproj.CRS(crs_code)
        print(f"CRS {crs_code} est valide: {crs}")
        return crs
    except Exception as e:
        print(f"Erreur lors de la validation du CRS {crs_code}: {e}")
        return None

# Valider les CRS
crs_wgs84 = validate_crs('EPSG:4326')
crs_terre_adelie = validate_crs('EPSG:32754')  # Utilisation d'un CRS général pour la Terre Adelie
crs_tasmanie = validate_crs('EPSG:7845')

def test_transformer(crs_source, crs_target):
    if crs_source is None or crs_target is None:
        print("CRS source ou cible non défini")
        return None
    try:
        transformer = pyproj.Transformer.from_crs(crs_source, crs_target, always_xy=True)
        print(f"Transformateur créé avec succès pour {crs_source} vers {crs_target}")
        return transformer
    except Exception as e:
        print(f"Erreur lors de la création du transformateur de {crs_source} vers {crs_target}: {e}")
        return None

# Test pour Terre Adelie avec le CRS général
transformer_terre_adelie = test_transformer(crs_wgs84, crs_terre_adelie)
# Test pour Tasmanie
transformer_tasmanie = test_transformer(crs_wgs84, crs_tasmanie)

def convert_to_cartesian(lon, lat, penguin):
    try:
        if penguin == "A":
            transformer = transformer_terre_adelie
        else:
            transformer = transformer_tasmanie

        if transformer is None:
            raise ValueError("Transformateur non défini correctement")

        # Convertir les coordonnées géographiques en coordonnées cartésiennes
        x, y = transformer.transform(lon, lat)
        return x, y
    except Exception as e:
        print(f"Erreur lors de la conversion des coordonnées lon={lon}, lat={lat}: {e}")
        return None, None


def convert_csv_to_cartesian(input_file, penguin):
    try:
        # Lit le fichier CSV
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        print("Fichier CSV lu avec succès")
        
        # Vérifie si les colonnes 'location-lon' et 'location-lat' existent
        if 'location-lon' not in df.columns or 'location-lat' not in df.columns:
            print("Les colonnes 'location-lon' ou 'location-lat' sont manquantes dans le fichier CSV")
            return
        
        # Vérifie le contenu des colonnes 'location-lon' et 'location-lat'
        print("Contenu des colonnes avant conversion :")
        print(df[['location-lon', 'location-lat']].head())
        
        # Convertit les coordonnées géographiques en coordonnées cartésiennes
        df['x'], df['y'] = zip(*df.apply(lambda row: convert_to_cartesian(row['location-lon'], row['location-lat'], penguin), axis=1))
        
        print("Conversion effectuée avec succès")
        
        # Ajoute des impressions pour déboguer
        print(df[['location-lon', 'location-lat', 'x', 'y']].head())  # Affiche les premières lignes pour vérifier
        
        # Écrit les nouvelles données dans le fichier CSV en écrasant l'ancien fichier
        df.to_csv(input_file, index=False)
        print("Fichier CSV mis à jour avec succès")
        
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
                
def rename_columns(file_path):
    # Lire le fichier CSV
    
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except PermissionError:
        print(f"Permission denied: '{file_path}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{file_path}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{file_path}'. Please check the path and file name.")
        return
    
    
    colonnes_a_supprimer = ['location-lon', 'location-lat']
    colonnes_existant = [col for col in colonnes_a_supprimer if col in df.columns]
    
    # Supprimer les colonnes
    df = df.drop(columns=colonnes_existant)
    
    
    # Dictionnaire de renommage des colonnes
    rename_dict = {
        'X': 'aX',
        'Y': 'aY',
        'Z': 'aZ',
    }
    
    # Renommer les colonnes
    df.rename(columns=rename_dict, inplace=True)
    
    # Enregistrer le fichier modifié
    df.to_csv(file_path, index=False)
    print(f"Colonnes renommées et fichier sauvegardé: {file_path}")


def centring(file_path, penguin) :
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except PermissionError:
        print(f"Permission denied: '{file_path}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{file_path}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{file_path}'. Please check the path and file name.")
        return
    
    if penguin == "A" :
        x0, y0 = 456055.352582322,2605835.728770192
    else :
        x0, y0 = 979248.3055429246, -4373574.081174698
    
    df['x'] = df['x'] - x0
    df['y'] = df['y'] - y0
    
    # Enregistrer le fichier modifié
    df.to_csv(file_path, index=False)
    print(f"Colonnes renommées et fichier sauvegardé: {file_path}")
    


def plot_3d_from_csv(file_path):
    """
    Cette fonction lit un fichier CSV et réalise un plot en 3 dimensions
    à partir des colonnes 'x', 'y', et 'Altitude', et ajoute un titre avec le nom du fichier.

    Parameters:
    file_path (str): Le chemin vers le fichier CSV à lire.
    """
    # Charger le fichier CSV
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except PermissionError:
        print(f"Permission denied: '{file_path}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{file_path}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{file_path}'. Please check the path and file name.")
        return

    # Lire les colonnes 'x', 'y', et 'Altitude'
    x = df['x']
    y = df['y']
    z = df['Altitude']

    # Extraire le nom du fichier sans l'extension
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Créer un plot en 3 dimensions
    fig = plt.figure(figsize=(10, 8), dpi=300)  # Augmenter la taille et la résolution de la figure
    ax = fig.add_subplot(111, projection='3d')

    # Ajouter les données au plot
    ax.scatter(x, y, z, c='r', marker='o', s=10, edgecolors='k', linewidths=0.5)  # Réduire la taille et ajouter un contour fin

    # Ajouter des labels
    ax.set_xlabel('X Label', fontsize=10)
    ax.set_ylabel('Y Label', fontsize=10)
    ax.set_zlabel('Altitude', fontsize=10)

    # Ajouter un titre
    title = f"Penguin's 3D Movement Representation - {file_name_without_extension}"
    ax.set_title(title, fontsize=12)

    # Améliorer la qualité visuelle
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.5)
    
    # Afficher le plot
    plt.show()
    
    
def plot_temperature(filename):
    print(f"Used file name is : {filename}")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['Temperature', 'Day', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    temp_values = np.array(df['Temperature'].tolist())  # Altitude in meters
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing Day column in file {filename}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_hours, temp_values, label='Experimental Temperature')
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Experimental Bio-logging Temperature as a Function of Time - {base_filename}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
#%%
"""
////////////////////////////////////////////////////////////////////////////////////
///                               Fonctions de débugging                         ///
////////////////////////////////////////////////////////////////////////////////////
"""

#%%

#Cherche les noms des colonnes du fichiers, ainsi que la valeur du séparateur


csv_file = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test\pa18_356y_F3_61_filtered.csv'



display_column_names(csv_file)


#%% 

#renvoie la liste des fichiers csv contenus dans un dossier

folder_path = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test'

list_files_in_directory(folder_path)

#%% 

#renvoie un dictionnaire avec les colonnes comme clés, et les types comme valeurs


csv_file = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test\pa18_356y_F3_61_filtered.csv'

get_column_types(csv_file)

#%% 

#Affiche le contenu des lignes, entre une ligne a et une ligne b

csv_file = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test\pa18_356y_F3_61_filtered.csv'

read_csv_lines(csv_file, 115, 120)

#%%
"""
////////////////////////////////////////////////////////////////////////////////////
///                       Fonctions de traitement de données                     ///
////////////////////////////////////////////////////////////////////////////////////
"""



#%% 

#Clean all the rows which not contain a 'pressure' value


input_folder = r'D:\PenguinsData\Filtered_Data\LittlePenguin\year2\PostGuard'



n = 1
for csv_file in glob.glob(os.path.join(input_folder, '*.csv')):
    try:
        base_name = os.path.basename(csv_file)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(input_folder, f"{name}_filtered{ext}")
        

        filter_and_convert_csv(csv_file, output_file)
        
        print(f"File {n}: {csv_file} filtered and saved as {output_file}")
        n += 1
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        
        

#%% 

#Clean all the columns which are not TagID	Timestamp	X	Y	Z	Activity	Pressure	Temp. (?C)	location-lat	location-lon

directory_path = r'D:\PenguinsData\Filtered_Data\AdeliePenguin\Year2\chick-rearing'


process_csv_files(directory_path)

#%% 

directory_path = r'D:\PenguinsData\Filtered_Data\LittlePenguin\year2\PostGuard'

# Définissez ici la fonction de transformation de la pression en altitude



create_altitude(directory_path)



        
#%% 

#Cleaning positive value
        
folder_path = r'D:\PenguinsData\Filtered_Data\AdeliePenguin\Year2\chick-rearing'


# Example of usage:
process_csv_files_in_folder(folder_path)



#%%

#Interpolating non existing coordinates points

filename = r'D:\PenguinsData\Data_finale\AdeliePenguin\Year2\chick-rearing_test\C4-AS04_S1_filtered.csv'

interpolate_missing_coordinates(filename)


#%%

#Interpolate a whole file


folder_path = r'D:\PenguinsData\Data_finale\LittlePenguin\year2\Incubation'



for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        #check_columns(file_path)
        interpolate_missing_coordinates(file_path)
        

#%%

#remplace la colonne TagID par Index, qui nomme les lignes de 0 à n
#renomme Temp (C ?) en Temperature



folder_path = r'D:\PenguinsData\Data_3\LittlePenguin\year2\PostGuard'

for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        #check_columns(file_path)
        clean_index_temperature(file_path)
        
#%%

#Transforme les coordonnées lon / lat en coordonnées x / y

        



for folder_path in fichier :
    for filename in os.listdir(folder_path[0]):
        print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path[0], filename)
            print(file_path)
            convert_csv_to_cartesian(file_path, folder_path[1])



#%%

# Renomme les colonnes accélérations et long / lat en x / y
for folder_path in fichier :
    for filename in os.listdir(folder_path[0]):
        print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path[0], filename)
            print(file_path)
            rename_columns(file_path)
        


#%%

# Renomme les colonnes accélérations et long / lat en x / y
for folder_path in fichier :
    for filename in os.listdir(folder_path[0]):
        print(filename)
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path[0], filename)
            print(file_path)
            centring(file_path, folder_path[1])       


#%%
"""
////////////////////////////////////////////////////////////////////////////////////
///                               Fonctions d'affichage                          ///
////////////////////////////////////////////////////////////////////////////////////
"""


#%% 

"""Data_0"""
"""Données brutes, suppression des lignes ne contenant pas de pression"""

#%% 

#Visualiser z(P) en fonction du temps (h), sur l'ensemble du temps

filename = r'D:\PenguinsData\Data_2\AdeliePenguin\Year2\chick-rearing\C40-AS83_S2_filtered.csv'

altitude_graph(filename)

#%% 

#Visualiser z(P) en fonction du temps (min), en donannt l'intervalle de temps 

filename = r'D:\PenguinsData\Data_0\AdeliePenguin\Year1\chick-rearing\pa18_360x_F12_04_filtered.csv'


altitude_graph_part(filename, 75000, 80000)



#%% 

"""Data_1"""
"""Altitudes calculées en fonction de la pression et du g / ρ"""
#%%

#Visualiser z(t) en fonction du temps (h), sur l'ensemble du temps


filename = r'D:\PenguinsData\Data_2\AdeliePenguin\Year2\chick-rearing\C40-AS83_S2_filtered.csv'


plot_altitude(filename)

#%%

#Visualiser T(t) et z(t) en fonction du temps (h), sur l'ensemble du temps
# deux plot séparés


filename = r'D:\PenguinsData\Data_4\AdeliePenguin\Year2\chick-rearing\C40-AS83_S2_filtered.csv'

plot_altitude(filename)
plot_temperature(filename)

#%%

#Visualiser T(t) et z(t) en fonction du temps (h), sur l'ensemble du temps
# un unique plot


filename = r'D:\PenguinsData\Data_4\AdeliePenguin\Year2\chick-rearing\C31-CR AS51_S1_filtered.csv'

plot_altitude_temperature(filename)

#%% 

#Visualiser z(t) en fonction du temps (h), en donannt l'intervalle de temps 

filename = r'D:\PenguinsData\Data_1\LittlePenguin\year2\Guard\20G3047FP16_S1_filtered.csv'


plot_altitude_part(filename, 71500, 72000)

#%% 

#Visualiser z(t) en fonction du temps (h), sur l'ensemble du temps, pour tous les manchots d'un dossier


folder_path = r'D:\PenguinsData\Data_1\LittlePenguin\year2\Guard'


for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        plot_altitude(file_path)


#%% 

#visualiser le déplacement avec lat = f(lon) pour un manchot donné

filename = r'D:\PenguinsData\Data_1\AdeliePenguin\Year1\chick-rearing\pa18_363z_F24_53_filtered.csv'


# Exemple d'utilisation de la fonction
plot_position(filename)


#%%

#visualiser le déplacement avec lat = f(lon) pour tous les manchots d'un dossier 


folder_path = r'D:\PenguinsData\Data_1\AdeliePenguin\Year1\chick-rearing'


    
for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        #check_columns(file_path)
        plot_position(file_path)

        
#%% 

"""Data_2"""
"""interpolation linéaire des coordonnées manquantes en lat / lon"""


#%% 

#visualiser le déplacement avec lat = f(lon) pour un manchot donné

filename = r'D:\PenguinsData\Data_2\AdeliePenguin\Year1\chick-rearing\pa18_363z_F24_53_filtered.csv'


# Exemple d'utilisation de la fonction
plot_position(filename)


#%%

#visualiser le déplacement avec lat = f(lon) pour tous les manchots d'un dossier 


folder_path = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test'


    
for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        #check_columns(file_path)
        plot_position(file_path)
#%% 

"""Data_3"""
"""Ajout d'un Index, et changement de nom pour la Temperature"""
   
#%% 

"""Data_4"""
"""Conversion du système de coordonnée lat / lon en y / x"""
#%% 
#visualiser le déplacement avec x = f(y) pour tous les manchots d'un dossier 

filename = r'D:\PenguinsData\Data_1\AdeliePenguin\Year1\chick-rearing\pa18_363z_F24_53_filtered.csv'


# Exemple d'utilisation de la fonction
plot_position_coordinates(filename)

#%%

#visualiser le déplacement 3d x/y/z pour tous les manchots d'un dossier 


folder_path = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\chick-rearing'


    
for filename in os.listdir(folder_path):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        #check_columns(file_path)
        plot_3d_from_csv(file_path)
        
        
#%%
      
file_path = r'D:\PenguinsData\Data_4\LittlePenguin\year2\Incubation\20I23090FP25_S1_filtered.csv'
    
plot_position_coordinates(file_path)
plot_altitude(file_path)     
plot_3d_from_csv(file_path)        
