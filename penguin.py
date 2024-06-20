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
from scipy.interpolate import interp1d
# -*- coding: utf-8 -*-

print("librairies importées")
print("Répertoire de travail actuel :", os.getcwd())



fichier = [[r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year1\chick-rearing', "A"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year2\chick-rearing', "A"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year1\Guard', "L"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year1\PostGuard', "L"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year1\Incubation', "L"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year2\Guard', "L"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year2\PostGuard', "L"],
           [r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\LittlePenguin\year2\Incubation', "L"]]

fichierB = [[r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\AdeliePenguin\Year1\chick-rearing', "A"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\AdeliePenguin\Year2\chick-rearing', "A"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year1\Guard', "L"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year1\PostGuard', "L"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year1\Incubation', "L"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year2\Guard', "L"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year2\PostGuard', "L"],
           [r'F:\PenguinsData\Data_interpolation_lineaire\Data_Simulation\LittlePenguin\year2\Incubation', "L"]]

"""

fichier = [[r'F:\PenguinsData\Data_correction\AdeliePenguin\Year1\chick-rearing', "A"],
           [r'F:\PenguinsData\Data_correction\AdeliePenguin\Year2\chick-rearing', "A"],]

"""
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
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        
        
def get_column_types(file_path):
    column_types = {}
    
    with open(file_path, 'r', newline='') as csvfile:
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

def read_csv_lines(file_path, start_line, end_line):
    line_data = []
    
    with open(file_path, 'r', newline='') as csvfile:
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
    

def get_files_in_directory(directory):
    """Return a set of file names in the given directory."""
    return set(os.listdir(directory))

def delete_files_not_in_directory_A(A, B):
    """Delete files in directory A that are not present in directory B."""
    files_in_A = get_files_in_directory(A)
    files_in_B = get_files_in_directory(B)

    files_to_delete = files_in_A - files_in_B

    for file_name in files_to_delete:
        file_path = os.path.join(A, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted {file_path}")
            
            

def altitude_graph(file_path):
    
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
    
def altitude_graph_part(file_path, a, b):
    
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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

def Suppression_Pression(input_file):
    if os.path.basename(input_file).startswith('._'):
        print(f"Fichier ignoré: '{input_file}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(input_file, encoding='latin1', on_bad_lines='skip')
    except PermissionError:
        print(f"Permission denied: '{input_file}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{input_file}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{input_file}'. Please check the path and file name.")
        return
    
    

    
    required_columns = ["Pressure"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    filtered_df = df.dropna(subset=required_columns)
    
    filtered_df.to_csv(input_file, index=False, sep=',')
    
    
    
def Suppression_veDBA(input_file):
    if os.path.basename(input_file).startswith('._'):
        print(f"Fichier ignoré: '{input_file}'")
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(input_file, encoding='latin1', on_bad_lines='skip')
    except PermissionError:
        print(f"Permission denied: '{input_file}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{input_file}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{input_file}'. Please check the path and file name.")
        return
    
    # Vérifier si la colonne 'veDBA' existe et la supprimer
    if 'veDBA' in df.columns:
        df = df.drop(columns=['veDBA'])
    else:
        print(f"Column 'veDBA' not found in the file: '{input_file}'")
        return
    
    # Sauvegarder le DataFrame modifié dans le fichier CSV
    df.to_csv(input_file, index=False, sep=',')
    
    
    
    
def process_file_paths(directory):
    for file_path in os.listdir(directory):
        file_path = os.path.join(directory, file_path)

        # Ignore hidden files or system files
        if file_path.startswith('.'):
            continue
        
        if file_path.endswith(".csv"):
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
    for file_path in os.listdir(directory):
        file_path = os.path.join(directory, file_path)

        # Ignore hidden or system files
        if file_path.startswith('.'):
            continue
        
        if file_path.endswith(".csv"):
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

def plot_altitude(file_path):
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
            print(f"Error parsing Day column in file {file_path}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_hours, z_values, label='Experimental Altitude')
    plt.xlabel('Time (hours)')
    plt.ylabel('Altitude (meters)')
    plt.title(f'Experimental Bio-logging Pressure as a Function of Time - {base_file_path}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_veDBA(file_path):
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['veDBA', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    veDBA = np.array(df['veDBA'].tolist())  # Altitude in meters
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing Day column in file {file_path}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_hours, veDBA, label='Experimental veDBA')
    plt.xlabel('Time (h)')
    plt.ylabel('veDBA (m/s²)')
    plt.title(f'Experimental Bio-logging veDBA as a Function of Time - {base_file_path}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_pitch_roll(file_path):
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['veDBA', 'Time_seconds']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    pitch = np.array(df['pitch'].tolist())  # Altitude in meters
    roll = np.array(df['roll'].tolist())  # Altitude in meters
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'], format='%d/%m/%Y')
    except ValueError:
        try:
            df['Day'] = pd.to_datetime(df['Day'], format='%Y-%m-%d')
        except ValueError as e:
            print(f"Error parsing Day column in file {file_path}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(time_values_hours, pitch, label='Experimental pitch', linewidth=1.5)
    plt.plot(time_values_hours, roll, label='Experimental roll', linewidth=1.5)
    
    plt.xlabel('Time (h)', fontsize=14)
    plt.ylabel('Pitch / Roll (°)', fontsize=14)
    plt.title(f'Experimental Pitch and Roll as a Function of Time - {base_file_path}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
                
def plot_pitch_roll_part(file_path, a, b):
    print(f"Used file name is: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Validate if required columns exist
    required_columns = ['pitch', 'roll', 'Day', 'Time_seconds', 'Altitude']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing column: {column}")
            return
    
    # Extract the different values
    pitch = np.array(df['pitch'].tolist())  # Altitude in meters
    roll = np.array(df['roll'].tolist())
    altitude = np.array(df['Altitude'].tolist())
    time_values_seconds = np.array(df['Time_seconds'].tolist())  # Time in seconds

    # Convert 'Day' column to datetime
    try:
        df['Day'] = pd.to_datetime(df['Day'])
    except ValueError as e:
        print(f"Error parsing Day column in file {file_path}: {e}")
        return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())

    # Filter the values between a and b seconds
    mask = (total_elapsed_seconds >= a) & (total_elapsed_seconds <= b)
    filtered_pitch = pitch[mask]
    filtered_roll = roll[mask]
    filtered_altitude = altitude[mask]
    filtered_total_elapsed_seconds = total_elapsed_seconds[mask]
    time_values_minutes = filtered_total_elapsed_seconds / 60  # Convert total elapsed time to minutes

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    ax1.plot(time_values_minutes, filtered_pitch, label='Experimental pitch')
    #ax1.plot(time_values_minutes, filtered_roll, label='Experimental roll')
    ax1.set_ylabel('Pitch/Roll (degrees)')
    ax1.set_title(f'Experimental pitch and roll as a Function of Time - {base_file_path}')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_values_minutes, filtered_altitude, label='Altitude', color='green')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Altitude (meters)')
    ax2.set_title('Altitude as a Function of Time')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_altitude_part(file_path, a, b):
    print(f"Used file name is: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
        print(f"Error parsing Day column in file {file_path}: {e}")
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
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_minutes, filtered_z_values, label='Experimental Altitude')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Altitude (meters)')
    plt.title(f'Experimental Bio-logging Pressure as a Function of Time - {base_file_path}')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_altitude_temperature(file_path):
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
            print(f"Error parsing Day column in file {file_path}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Altitude (meters)', color=color)
    ax1.plot(time_values_hours, z_values, color=color, label='Experimental Altitude')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color)
    ax2.plot(time_values_hours, temp_values, color=color, label='Experimental Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Experimental Bio-logging Temperature and Altitude as a Function of Time - {base_file_path}')
    fig.tight_layout()  # Adjust layout to fit titles, labels, etc. without overlap
    fig.legend()
    plt.grid(True)
    plt.show()
    
    
def altitude_positive(folder_path):
    # List of encodings to try
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    
    # Iterate over all files in the given folder
    for file_path in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path, file_path)
            df = None
            
            # Try reading the CSV file with different encodings
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break  # Exit the loop if reading is successful
                except Exception as e:
                    print(f"Error reading file {file_path} with encoding {encoding}: {e}")
            
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
                        print(f"'Altitude' column not found in file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            else:
                print(f"Failed to read file {file_path} with any encoding.")

def list_files_in_directory(directory):
    try:
        files = os.listdir(directory)
        print("Fichiers dans le répertoire :", files)
    except PermissionError:
        print(f"Permission denied: '{directory}'. Please check your directory permissions.")
    except FileNotFoundError:
        print(f"Directory not found: '{directory}'. Please check the path.")

def plot_position(file_path):
    # Lire le fichier CSV avec un encodage spécifique
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
    
    # Afficher les colonnes pour débogage
    #print("Colonnes disponibles : ", df.columns)
    
    # Utiliser les noms de colonnes exacts obtenus de la vérification précédente
    required_columns = ['location-lat', 'location-lon', 'Time_seconds']  # Mettre à jour selon les noms exacts
    
    # Vérifier si toutes les colonnes nécessaires sont présentes
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"La colonne requise '{col}' est manquante dans le fichier CSV.")
    
    # Filtrer les lignes avec des valeurs non nulles dans les colonnes requises
    df_filtered = df.dropna(subset=required_columns)
    
    # Récupérer les colonnes nécessaires
    latitude = df_filtered['location-lat']
    longitude = df_filtered['location-lon']
    time_seconds = df_filtered['Time_seconds']
    
    # Convertir le temps en heures
    time_hours = time_seconds / 3600
    
    # Normaliser le temps pour la colormap
    norm = plt.Normalize(time_hours.min(), time_hours.max())
    cmap = plt.get_cmap('viridis')
    
    # Obtenir le nom du fichier pour le titre
    file_name = os.path.basename(file_path)
    
    # Créer le plot avec l'évolution des couleurs en fonction du temps
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(longitude, latitude, c=time_hours, cmap=cmap, norm=norm, alpha=0.7)
    plt.colorbar(sc, label='Time (hours)')
    plt.title(f"Penguin's Movement Over Time - {file_name}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show() 
    
def plot_position_coordinates(file_path):
    # Lire le fichier CSV avec un encodage spécifique
    print("on est là")
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
    file_name = os.path.basename(file_path)
    
    # Créer le plot avec l'évolution des couleurs en fonction du temps
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x, y, c=time_hours, cmap=cmap, norm=norm, alpha=0.7)
    plt.colorbar(sc, label='Time (hours)')
    plt.title(f"Penguin's Movement Over Time - {file_name}")
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    plt.grid(True)
    plt.show()

def check_columns(file_path):
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        print("Colonnes disponibles : ", df.columns)
    except PermissionError:
        print(f"Permission denied: '{file_path}'. Please check your file permissions.")
    except UnicodeDecodeError:
        print(f"Error decoding file: '{file_path}'. Please check the encoding.")
    except FileNotFoundError:
        print(f"File not found: '{file_path}'. Please check the path and file name.")

def interpolate_missing_coordinates(file_path):
    # Vérifier si le fichier commence par '._'
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
    
    # Afficher les colonnes pour débogage
    print("Colonnes disponibles : ", df.columns)
    
    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = ['y', 'x', 'Time_seconds']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"La colonne requise '{col}' est manquante dans le fichier CSV.")
    
    # Appliquer l'interpolation linéaire aux colonnes 'location-lat' et 'location-lon'
    df['y'] = df['y'].interpolate(method='linear')
    df['x'] = df['x'].interpolate(method='linear')
    
    # Remplir les éventuelles valeurs restantes en utilisant la méthode de remplissage avant et arrière
    df['y'] = df['y'].fillna(method='bfill').fillna(method='ffill')
    df['x'] = df['x'].fillna(method='bfill').fillna(method='ffill')
    
    # Vérifier s'il reste des valeurs manquantes
    if df[['y', 'x']].isnull().sum().sum() > 0:
        print("Il reste des valeurs manquantes après l'interpolation.")
    else:
        print("Toutes les valeurs manquantes ont été interpolées.")
    
    # Écrire les données interpolées dans le même fichier CSV
    df.to_csv(file_path, index=False, encoding='latin1')
    
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
        if type(df[['location-lon']] ) != float('nan') :
            
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
    
    
    colonnes_a_supprimer = ['location-lon', 'location-lat', 'Temperature']
    colonnes_existant = [col for col in colonnes_a_supprimer if col in df.columns]
    
    # Supprimer les colonnes
    df = df.drop(columns=colonnes_existant)
    
    
    
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
    
    
def plot_temperature(file_path):
    print(f"Used file name is : {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
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
            print(f"Error parsing Day column in file {file_path}: {e}")
            return

    # Calculate total elapsed time in seconds
    start_day = df['Day'].min()
    df['Elapsed_seconds'] = (df['Day'] - start_day).dt.total_seconds() + time_values_seconds
    total_elapsed_seconds = np.array(df['Elapsed_seconds'].tolist())
    time_values_hours = total_elapsed_seconds / 3600  # Convert total elapsed time to hours

    # Extract the base name of the file (without path and extension)
    base_file_path = os.path.splitext(os.path.basename(file_path))[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_values_hours, temp_values, label='Experimental Temperature')
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Experimental Bio-logging Temperature as a Function of Time - {base_file_path}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
def smooth_data(file_path):
    # Ignorer les fichiers commençant par '._'
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    # Détecter le séparateur du fichier CSV
    try:
        separator = detect_separator(file_path)
    except ValueError as e:
        print(e)
        return
    
    # Lire le fichier CSV avec un encodage spécifique
    try:
        df = pd.read_csv(file_path, encoding='latin1', sep=separator)
    except PermissionError:
        print(f"Permission denied: '{file_path}'. Please check your file permissions.")
        return
    except UnicodeDecodeError:
        print(f"Error decoding file: '{file_path}'. Please check the encoding.")
        return
    except FileNotFoundError:
        print(f"File not found: '{file_path}'. Please check the path and file name.")
        return
    
    # Vérifier si les colonnes 'X', 'Y', 'Z' existent
    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        raise ValueError("Le fichier CSV doit contenir les colonnes 'X', 'Y' et 'Z'")
    
    # Fonction pour lisser une colonne
    def smooth_column(column):
        smoothed = []
        length = len(column)
        for i in range(0, length, 24):
            chunk = column[i:i+25]
            mean_value = round(chunk.mean(), 3)
            smoothed.extend([mean_value] * len(chunk))
        # Tronquer les données lissées pour qu'elles aient la même longueur que les données d'origine
        smoothed = smoothed[:length]
        return smoothed
    
    # Appliquer le lissage aux colonnes 'X', 'Y' et 'Z'
    df['X'] = smooth_column(df['X'])
    df['Y'] = smooth_column(df['Y'])
    df['Z'] = smooth_column(df['Z'])

    # Sauvegarder le dataframe lissé dans le même fichier CSV avec des virgules comme séparateur
    df.to_csv(file_path, index=False, sep=',')

    print(f"Le fichier '{file_path}' a été lissé et enregistré avec des virgules comme séparateur.")

    
    
def rename_csv(repertoire):
    """
    Renomme tous les fichiers CSV dans le répertoire spécifié en supprimant '_filtered' de leur nom.
    Écrase les anciens fichiers s'ils existent.

    Paramètres:
    repertoire (str): Chemin du répertoire contenant les fichiers CSV à renommer.
    """
    # Assurez-vous que le chemin du répertoire est valide
    if not os.path.isdir(repertoire):
        print(f"Le répertoire spécifié n'existe pas : {repertoire}")
        return

    # Créer le chemin de recherche pour les fichiers CSV
    chemin_de_recherche = os.path.join(repertoire, '*_filtered.csv')

    # Trouver tous les fichiers CSV qui contiennent '_filtered' dans leur nom
    fichiers_a_renommer = glob.glob(chemin_de_recherche)

    # Pour chaque fichier trouvé, renommer en enlevant '_filtered'
    for fichier in fichiers_a_renommer:
        # Générer le nouveau nom de fichier en enlevant '_filtered'
        nouveau_nom = fichier.replace('_filtered', '')

        # Vérifier si un fichier avec le même nom sans '_filtered' existe déjà
        if os.path.exists(nouveau_nom):
            # Si le fichier existe, supprimer l'ancien fichier pour permettre l'écrasement
            os.remove(nouveau_nom)
            print(f"Ancien fichier supprimé : {nouveau_nom}")

        # Renommer le fichier actuel en supprimant '_filtered'
        os.rename(fichier, nouveau_nom)
        print(f"Renommé : {fichier} -> {nouveau_nom}")

    print("Tous les fichiers ont été renommés et les anciens fichiers ont été écrasés si nécessaire.")


def pitch_roll(file_path):
    """
    Fonction pour lire un fichier CSV, modifier certaines colonnes,
    ajouter de nouvelles colonnes 'pitch' et 'roll', et sauvegarder le fichier modifié.
    
    Paramètre:
    file_path (str): Chemin du fichier CSV à traiter.
    """
    # Vérifier si le fichier est ignoré
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
    
    # Vérifier si les colonnes 'X', 'Y', 'Z' existent
    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        raise ValueError("Le fichier CSV doit contenir les colonnes 'X', 'Y' et 'Z'")

    # Multiplier les colonnes 'X', 'Y' et 'Z' par 9.81 pour convertir en m/s^2
    df['X'] = df['X'] * 9.81 
    df['Y'] = df['Y'] * 9.81 
    df['Z'] = df['Z'] * 9.81 

    # Afficher les valeurs avant le calcul pour déboguer
    print("Valeurs de X, Y, Z après multiplication par 9.81 :")
    print(df[['X', 'Y', 'Z']])

    # Créer les nouvelles colonnes 'pitch' et 'roll' avec les formules correctes
    df['pitch'] = np.arctan2(df['Y'], np.sqrt(df['X']**2 + df['Z']**2)) * (180 / np.pi)
    df['roll'] = np.arctan2(df['X'], np.sqrt(df['Y']**2 + df['Z']**2)) * (180 / np.pi)

    # Arrondir les résultats à trois décimales
    df['pitch'] = df['pitch'].round(3)
    df['roll'] = df['roll'].round(3)

    # Afficher les valeurs calculées pour déboguer
    print("Valeurs calculées de pitch et roll :")
    print(df[['pitch', 'roll']])

    # Sauvegarder le fichier modifié en écrasant l'ancien fichier
    df.to_csv(file_path, index=False)

    print(f"Le fichier {file_path} a été modifié avec succès.")

def veDBA(file_path):
    """
    Fonction pour lire un fichier CSV, modifier certaines colonnes,
    ajouter de nouvelles colonnes 'pitch' et 'roll', et sauvegarder le fichier modifié.
    
    Paramètre:
    file_path (str): Chemin du fichier CSV à traiter.
    """
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
    
    # Vérifier si les colonnes 'aX', 'aY', 'aZ' existent
    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        raise ValueError("Le fichier CSV doit contenir les colonnes 'X', 'Y' et 'Z'")

    # Multiplier les colonnes 'aX', 'aY' et 'aZ' par 9.81
    df['X'] = round(df['X'] * 9.81 , 2)
    df['Y'] = round(df['Y'] * 9.81 , 2 )
    df['Z'] = round(df['Z'] * 9.81 , 2 )

    # Créer les nouvelles colonnes 'pitch' et 'roll' avec les formules données
    df['veDBA'] = round( np.sqrt(df['Y']**2 + df['Z']**2 + df['X']**2 ), 2)


    # Sauvegarder le fichier modifié en écrasant l'ancien fichier
    df.to_csv(file_path, index=False)

    print(f"Le fichier {file_path} a été modifié avec succès.")

def remove_temperature_column(file_path):
    """
    Cette fonction lit un fichier CSV, supprime la colonne 'Temperature',
    puis enregistre le résultat en remplaçant l'ancien fichier.

    :param file_path: Chemin du fichier CSV à modifier
    """
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    try:
        # Lire le fichier CSV
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
    
    # Vérifier si la colonne 'Temperature' existe
    if 'Temperature' not in df.columns:
        print("La colonne 'Temperature' n'existe pas dans le fichier CSV.")
        return
    
    # Supprimer la colonne 'Temperature'
    df = df.drop(columns=['Temperature'])

    # Sauvegarder le dataframe mis à jour dans le même fichier CSV
    df.to_csv(file_path, index=False)
    print(f"Le fichier mis à jour a été enregistré sous le nom '{file_path}'.")
    
    
    
def remove_acceleration_column(file_path):
    """
    Cette fonction lit un fichier CSV, supprime la colonne 'X', 'Y', 'Z',
    puis enregistre le résultat en remplaçant l'ancien fichier.

    :param file_path: Chemin du fichier CSV à modifier
    """
    if os.path.basename(file_path).startswith('._'):
        print(f"Fichier ignoré: '{file_path}'")
        return
    
    try:
        # Lire le fichier CSV
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
    
    # Vérifier si la colonne 'Temperature' existe
    if 'X' not in df.columns:
        print("La colonne 'X' n'existe pas dans le fichier CSV.")
        return
    
    # Supprimer la colonne 'Temperature'
    df = df.drop(columns=['X'])
    df = df.drop(columns=['Y'])
    df = df.drop(columns=['Z'])
    # Sauvegarder le dataframe mis à jour dans le même fichier CSV
    df.to_csv(file_path, index=False)
    print(f"Le fichier mis à jour a été enregistré sous le nom '{file_path}'.")



def interpolation_based_acceleration(file_path, sampling_rate=5):
    # Chargement des données depuis le fichier CSV
    def load_data(file_path):
        # Vérification si le fichier est caché
        if os.path.basename(file_path).startswith('._'):
            print(f"Ignoring hidden file: {file_path}")
            return None
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    # Intégration des accélérations pour estimer la vitesse et la position
    def integrate_acceleration(data, dt):
        velocities = np.zeros((len(data), 2))
        positions = np.zeros((len(data), 2))

        for i in range(1, len(data)):
            velocities[i] = velocities[i-1] + dt * np.array([data.loc[i, 'X'], data.loc[i, 'Y']])
            positions[i] = positions[i-1] + dt * velocities[i]

        return positions

    # Interpolation des positions manquantes en utilisant les données d'accélérations
    def interpolate_positions_using_acceleration(data, sampling_rate):
        # Calcul de l'intervalle de temps (dt) en secondes
        dt = sampling_rate
        
        # Intégration des accélérations pour obtenir les positions
        positions = integrate_acceleration(data, dt)

        # Mettre à jour les colonnes 'x' et 'y' avec les nouvelles positions
        data['x'] = data['x'].combine_first(pd.Series(positions[:, 0], index=data.index))
        data['y'] = data['y'].combine_first(pd.Series(positions[:, 1], index=data.index))

        return data

    # Fonction principale
    def main(file_path):
        data = load_data(file_path)
        
        if data is None:
            print(f"Skipping file due to loading issue: {file_path}")
            return

        # Rééchantillonner les données selon le sampling rate
        data = data.iloc[::sampling_rate, :].reset_index(drop=True)
        
        # Interpoler les positions en utilisant les données d'accélérations
        data = interpolate_positions_using_acceleration(data, sampling_rate)
        
        # Enregistrer les données traitées dans le fichier
        data.to_csv(file_path, index=False)
        print(f"Processed file saved: {file_path}")

    main(file_path)


def convertir_separateur_csv(nom_fichier):
    # Lis le fichier d'origine avec le séparateur de tabulation
    with open(nom_fichier, 'r', newline='', encoding='utf-8') as fichier_in:
        lecteur = csv.reader(fichier_in, delimiter= detect_separator(nom_fichier))
        lignes = list(lecteur)

    # Écris le fichier avec le nouveau séparateur de virgule
    with open(nom_fichier, 'w', newline='', encoding='utf-8') as fichier_out:
        ecrivain = csv.writer(fichier_out, delimiter=',')
        ecrivain.writerows(lignes)

    print(f"Le fichier {nom_fichier} a été converti avec des virgules comme séparateur.")

def generated_test(file_path):
    # Vérification si le fichier est caché
    if os.path.basename(file_path).startswith('._'):
        print(f"Ignoring hidden file: {file_path}")
        return

    # Chargement des données depuis le fichier CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    # Assurez-vous que la colonne 'location-lon' existe
    if 'location-lon' not in data.columns:
        print(f"'location-lon' column not found in {file_path}")
        return

    # Créer la colonne 'Generated' initialement remplie de 0
    data['Generated'] = 0

    # Trouver les index des valeurs non nulles de la colonne 'location-lon'
    non_null_indices = data.index[data['location-lon'].notnull()].tolist()

    # Marquer les positions selon les critères spécifiés
    for i in non_null_indices:
        data.loc[max(i-1, 0):min(i+1, len(data)-1), 'Generated'] = 1

    # Enregistrer le fichier CSV avec les nouvelles valeurs
    data.to_csv(file_path, index=False)
    print(f"Processed file saved: {file_path}")
    
    
    
def compare_depth_and_altitude(depth_filename, altitude_filename):
    depth_values = []
    altitude_values = []
    
    # Lire les valeurs de profondeur depuis le fichier texte
    with open(depth_filename, 'r') as file:
        header = file.readline().strip().split()
        
        if 'depth' not in header:
            raise ValueError("La colonne 'depth' est absente du fichier")
        
        depth_index = header.index('depth')
        
        for line in file:
            values = line.strip().split()
            depth_values.append(float(values[depth_index]))
    
    # Lire les valeurs d'altitude depuis le fichier CSV
    with open(altitude_filename, 'r') as file:
        reader = csv.DictReader(file)
        
        if 'Altitude' not in reader.fieldnames:
            raise ValueError("La colonne 'Altitude' est absente du fichier")
        
        for row in reader:
            altitude_values.append(float(row['Altitude']))
    
    depth_values = np.array(depth_values)
    altitude_values = np.array(altitude_values)
    
    # Vérifier que les deux ensembles de données ont la même longueur
    if len(depth_values) != len(altitude_values):
        raise ValueError("Les fichiers depth et altitude n'ont pas le même nombre de lignes")
    
    # Créer une figure avec deux sous-graphiques empilés verticalement
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=300)
    
    # Tracer les valeurs de profondeur
    axs[0].plot(-depth_values, color='b', label='Depth')
    axs[0].set_title('Depth Values')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Depth')
    axs[0].legend()
    axs[0].grid(True)
    
    # Tracer les valeurs d'altitude
    axs[1].plot(altitude_values, color='r', label='Altitude')
    axs[1].set_title('Altitude Values')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Altitude')
    axs[1].legend()
    axs[1].grid(True)
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
    
    # Calculer la différence entre altitude et profondeur
    difference = altitude_values + depth_values
    
    # Tracer la différence sur un autre graphique
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(difference, color='g', label='Altitude + Depth')
    plt.title('Difference between Altitude and Depth')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def compare_pitch(pitch_file1, pitch_file2):
    pitch_values1 = []
    pitch_values2 = []
    
    # Lire les valeurs de pitch depuis le premier fichier texte (séparé par des virgules)
    with open(pitch_file1, 'r') as file:
        header = file.readline().strip().split(',')
        
        if 'pitch' not in header:
            raise ValueError("La colonne 'pitch' est absente du fichier 1")
        
        pitch_index = header.index('pitch')
        
        for line in file:
            values = line.strip().split(',')
            pitch_values1.append(float(values[pitch_index]))
    
    # Lire les valeurs de pitch depuis le deuxième fichier texte
    with open(pitch_file2, 'r') as file:
        header = file.readline().strip().split(',')
        
        if 'pitch' not in header:
            raise ValueError("La colonne 'pitch' est absente du fichier 2")
        
        pitch_index = header.index('pitch')
        
        for line in file:
            values = line.strip().split(',')
            pitch_values2.append(float(values[pitch_index]))
    
    pitch_values1 = np.array(pitch_values1)
    pitch_values2 = np.array(pitch_values2)
    
    # Trancher les valeurs de pitch du premier fichier pour que leur taille soit un multiple de 100
    trunc_size1 = (len(pitch_values1) // 100) * 100
    pitch_values1 = pitch_values1[:trunc_size1]
    
    # Moyenner les valeurs de pitch du premier fichier par paquets de 100
    averaged_pitch_values1 = np.mean(pitch_values1.reshape(-1, 100), axis=1)
    
    # Trancher les valeurs de pitch du deuxième fichier pour qu'elles aient la même longueur que les valeurs moyennées du premier fichier
    trunc_size2 = len(averaged_pitch_values1)
    pitch_values2 = pitch_values2[:trunc_size2]
    
    # Vérifier que les deux ensembles de données ont la même longueur
    if len(averaged_pitch_values1) != len(pitch_values2):
        raise ValueError("Les fichiers pitch n'ont pas le même nombre de lignes après tronquage et moyenne")
    
    # Créer une figure avec deux sous-graphiques empilés verticalement
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=300)
    
    # Tracer les valeurs de pitch du premier fichier (après moyenne)
    axs[0].plot(averaged_pitch_values1, color='b', label='Averaged Pitch File 1')
    axs[0].set_title('Averaged Pitch Values from File 1')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Pitch')
    axs[0].legend()
    axs[0].grid(True)
    
    # Tracer les valeurs de pitch du deuxième fichier
    axs[1].plot(pitch_values2, color='r', label='Pitch File 2')
    axs[1].set_title('Pitch Values from File 2')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Pitch')
    axs[1].legend()
    axs[1].grid(True)
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
    
    # Calculer la différence entre les valeurs moyennes de pitch du premier fichier et les valeurs de pitch du deuxième fichier
    difference = averaged_pitch_values1 - pitch_values2
    
    # Tracer la différence sur un autre graphique
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(difference, color='g', label='Averaged Pitch File 1 - Pitch File 2')
    plt.title('Difference between Averaged Pitch Values from File 1 and Pitch Values from File 2')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    plt.show()


    
def compare_pitch_part(pitch_file1, pitch_file2, a, b):
    pitch_values1 = []
    pitch_values2 = []
    
    # Lire les valeurs de pitch depuis le premier fichier texte (séparé par des virgules)
    with open(pitch_file1, 'r') as file:
        header = file.readline().strip().split(',')
        
        if 'pitch' not in header:
            raise ValueError("La colonne 'pitch' est absente du fichier 1")
        
        pitch_index = header.index('pitch')
        
        for line in file:
            values = line.strip().split(',')
            pitch_values1.append(float(values[pitch_index]))
    
    # Lire les valeurs de pitch depuis le deuxième fichier texte
    with open(pitch_file2, 'r') as file:
        header = file.readline().strip().split(',')
        
        if 'pitch' not in header:
            raise ValueError("La colonne 'pitch' est absente du fichier 2")
        
        pitch_index = header.index('pitch')
        
        for line in file:
            values = line.strip().split(',')
            pitch_values2.append(float(values[pitch_index]))
    
    pitch_values1 = np.array(pitch_values1)
    pitch_values2 = np.array(pitch_values2)
    
    # Filtrer les valeurs entre les indices a et b pour les deux fichiers
    pitch_values1 = pitch_values1[a:b]
    pitch_values2 = pitch_values2[a:b]
    
    # Trancher les valeurs de pitch du premier fichier pour que leur taille soit un multiple de 100
    trunc_size1 = (len(pitch_values1) // 100) * 100
    pitch_values1 = pitch_values1[:trunc_size1]
    
    # Moyenner les valeurs de pitch du premier fichier par paquets de 100
    averaged_pitch_values1 = np.mean(pitch_values1.reshape(-1, 100), axis=1)
    
    # Trancher les valeurs de pitch du deuxième fichier pour qu'elles aient la même longueur que les valeurs moyennées du premier fichier
    trunc_size2 = len(averaged_pitch_values1)
    pitch_values2 = pitch_values2[:trunc_size2]
    
    # Vérifier que les deux ensembles de données ont la même longueur
    if len(averaged_pitch_values1) != len(pitch_values2):
        raise ValueError("Les fichiers pitch n'ont pas le même nombre de lignes après tronquage et moyenne")
    
    # Créer une figure avec deux sous-graphiques empilés verticalement
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), dpi=300)
    
    # Tracer les valeurs de pitch du premier fichier (après moyenne)
    axs[0].plot(averaged_pitch_values1, color='b', label='Averaged Pitch File 1')
    axs[0].set_title('Averaged Pitch Values from File 1')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Pitch (degrees)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Tracer les valeurs de pitch du deuxième fichier
    axs[1].plot(pitch_values2, color='r', label='Pitch File 2')
    axs[1].set_title('Pitch Values from File 2')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Pitch (degrees)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()
    
    # Calculer la différence entre les valeurs moyennes de pitch du premier fichier et les valeurs de pitch du deuxième fichier
    difference = averaged_pitch_values1 - pitch_values2
    
    # Tracer la différence sur un autre graphique
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(difference, color='g', label='Averaged Pitch File 1 - Pitch File 2')
    plt.title('Difference between Averaged Pitch Values from File 1 and Pitch Values from File 2')
    plt.xlabel('Index')
    plt.ylabel('Difference (degrees)')
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


file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year2\chick-rearing\C1-AL35_S1.csv'



display_column_names(file_path)


#%% 

#renvoie la liste des fichiers csv contenus dans un dossier

folder_path = r'D:\PenguinsData\Data_interpolation_acceleration\Data_1\AdeliePenguin\Year1\chick-rearing'

list_files_in_directory(folder_path)

#%% 

#renvoie un dictionnaire avec les colonnes comme clés, et les types comme valeurs


file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year2\chick-rearing\C1-AL35_S1.csv'

get_column_types(file_path)

#%% 

#Affiche le contenu des lignes, entre une ligne a et une ligne b

file_path = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test\pa18_356y_F3_61_filtered.csv'

read_csv_lines(file_path, 115, 120)






#%%
"""
////////////////////////////////////////////////////////////////////////////////////
///                       Fonctions de traitement de données                     ///
////////////////////////////////////////////////////////////////////////////////////
"""


#%% 

#Supprime les fichiers n'apparaissant pas dans le deuxième dossier indiquié

for i in range (len(fichier)) :
    A = fichier[i][0]
    B = fichierB[i][0]

    
    delete_files_not_in_directory_A(A,B) 

#%% 
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year2\chick-rearing\C1-AL35_S1.csv'

#convertit le séparateur en ','


convertir_separateur_csv(file_path)



#%%
# Moyenne les accélérations par paquet de 25
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)
            smooth_data(file_path)
            
            
              
#%% 

#Supprime toutes les lignes qui ne contiennent pas une valeur valide de 'Pressure'



for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            Suppression_Pression(file_path)

        
        

#%% 

#Supprime toutes les colonne qui ne sont pas :
#TagID	Timestamp	X	Y	Z	Activity	Pressure	Temp. (?C)	location-lat	location-lon


for folder_path in fichier :
    process_file_paths(folder_path[0]) 


#%% 


# Créer une colonne 'Altitude' calculée avec les lois de l'hydrostatiques
#Nécessaire d'accéder à la latitude, la pression et la température


for folder_path in fichier :
    create_altitude(folder_path[0]) 




        
#%% 

#supprime les points positifs en altitude, en les remettant à 0

for folder_path in fichier :
    print(folder_path)
    for file_path in os.listdir(folder_path[0]):
        altitude_positive(folder_path[0]) 
        

#%% 

#vérifie si la ligne est interpolée ou pas, en plaçant des 1 dans les lignes où la valeur est de base (ainsi que sur les lignes adjacentes)

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            generated_test(file_path)


#%%

#remplace la colonne TagID par Index, qui nomme les lignes de 0 à n
#renomme Temp (C ?) en Temperature

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)
            clean_index_temperature(file_path)
            
            
            
#%%

#Transforme les coordonnées lon / lat en coordonnées x / y

        

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)
            convert_csv_to_cartesian(file_path, folder_path[1])



#%%

#Interpolation linéaire

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            interpolate_missing_coordinates(file_path)



#%%

# Supprime les colonnes location-lat et location-lon

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)
            rename_columns(file_path)



#%%

# Recentre l'entièreté des coordonnées en fonctions d'un point de référence
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)
            centring(file_path, folder_path[1])       


            

    
#%%
    
#génère les valeurs de pitch et de roll en fonction de l'accélération
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)  
            pitch_roll(file_path)
            

            
            
""""#%%"""

#Enlève les colonnes X Y Z
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)  
            remove_acceleration_column(file_path)           
            
            


#Enlève la colonne Temperature
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)  
            remove_temperature_column(file_path)   
        
            

#%%
    
#génère les valeurs de veDBA en fonction de l'accélération

for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)  
            veDBA(file_path)            
            
        
        
        
#%%

#Enlève la colonne veDBA
for folder_path in fichier :
    for file_path in os.listdir(folder_path[0]):
        print(file_path)
        if file_path.endswith('.csv'):
            file_path = os.path.join(folder_path[0], file_path)
            print(file_path)  
            Suppression_veDBA(file_path)   
         
             
                

        
    
#%%           


#Enlève les _filtered à la fin des noms de fichiers

for folder_path in fichier :
    rename_csv(folder_path[0])
    
#%%
"""
////////////////////////////////////////////////////////////////////////////////////
///                               Fonctions d'affichage                          ///
////////////////////////////////////////////////////////////////////////////////////
"""



#Visualiser z(P) en fonction du temps (h), sur l'ensemble du temps

file_path = r'D:\PenguinsData\Data_2\AdeliePenguin\Year2\chick-rearing\C40-AS83_S2_filtered.csv'

altitude_graph(file_path)

#%% 

#Visualiser z(P) en fonction du temps (min), en donannt l'intervalle de temps 

file_path = r'D:\PenguinsData\Data_0\AdeliePenguin\Year1\chick-rearing\pa18_360x_F12_04_filtered.csv'


altitude_graph_part(file_path, 75000, 80000)



#%%

#Visualiser z(t) en fonction du temps (h), sur l'ensemble du temps


file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'


plot_altitude(file_path)



#%% 

#Visualiser z(t) en fonction du temps (h), en donannt l'intervalle de temps 

file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_3\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'


plot_altitude_part(file_path, 167200, 168200)

#%% 

#Visualiser z(t) en fonction du temps (h), sur l'ensemble du temps, pour tous les manchots d'un dossier




for file_path in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_path.endswith('.csv'):
        file_path = os.path.join(folder_path, file_path)
        plot_altitude(file_path)


#%% 

#visualiser le déplacement avec lat = f(lon) pour un manchot donné

file_path = r'D:\PenguinsData\Data_1\AdeliePenguin\Year1\chick-rearing\pa18_363z_F24_53_filtered.csv'


# Exemple d'utilisation de la fonction
plot_position(file_path)


#%%

#visualiser le déplacement avec lat = f(lon) pour tous les manchots d'un dossier 


folder_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year1\chick-rearing'


    
for file_path in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_path.endswith('.csv'):
        file_path = os.path.join(folder_path, file_path)
        print(file_path)
        #check_columns(file_path)
        plot_position(file_path)

        
#%% 

"""Data_2"""
"""interpolation linéaire des coordonnées manquantes en lat / lon"""


#%% 

#visualiser le déplacement avec lat = f(lon) pour un manchot donné

file_path = r'D:\PenguinsData\Data_2\AdeliePenguin\Year1\chick-rearing\pa18_363z_F24_53_filtered.csv'


# Exemple d'utilisation de la fonction
plot_position(file_path)


#%%

#visualiser le déplacement avec lat = f(lon) pour tous les manchots d'un dossier 


folder_path = r'D:\PenguinsData\Data_4\AdeliePenguin\Year1\test'


    
for file_path in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_path.endswith('.csv'):
        file_path = os.path.join(folder_path, file_path)
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
#visualiser le déplacement avec x = f(y) pour un manchot 

file_path = r'D:\PenguinsData\Data_interpolation_acceleration\Data_1\AdeliePenguin\Year1\pa18_363z_F24_53.csv'


# Exemple d'utilisation de la fonction
plot_position_coordinates(file_path)



#%%
#visualiser le déplacement avec x = f(y) pour tous un manchot d'un dossier

folder_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_0\AdeliePenguin\Year1\chick-rearing'

for file_path in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_path.endswith('.csv'):
        file_path = os.path.join(folder_path, file_path)
        print(file_path)
        #check_columns(file_path)
        plot_position_coordinates(file_path)
        
        
        
#%%

#visualiser le déplacement 3d x/y/z pour tous les manchots d'un dossier 


folder_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu\AdeliePenguin\Year1\chick-rearing'


    
for file_path in os.listdir(folder_path):
    # Check if the file is a CSV
    if file_path.endswith('.csv'):
        file_path = os.path.join(folder_path, file_path)
        print(file_path)
        #check_columns(file_path)
        plot_3d_from_csv(file_path)
        
#%%

#Visualiser T(t) et z(t) en fonction du temps (h), sur l'ensemble du temps
# deux plot séparés


file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu\AdeliePenguin\Year1\chick-rearing\pa18_365z_F30_05.csv'

plot_altitude(file_path)
plot_temperature(file_path)


        
#%%
      
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu\AdeliePenguin\Year1\chick-rearing\pa18_365z_F30_05.csv'
    
plot_position_coordinates(file_path)
plot_altitude(file_path)     
plot_3d_from_csv(file_path)        


#%%

#Visualiser T(t) et z(t) en fonction du temps (h), sur l'ensemble du temps
# un unique plot


file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu\AdeliePenguin\Year1\chick-rearing\pa18_365z_F30_05.csv'

plot_altitude_temperature(file_path)

#%%
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_finale\AdeliePenguin\Year1\chick-rearing\pa18_364z_F26_52.csv'

plot_veDBA(file_path)

#%%
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_3\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'

plot_pitch_roll(file_path)

#%%

#plot le pitch (et le roll si on veut), avec l'altitude en bas

file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_3\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'


plot_pitch_roll_part(file_path, 177600, 179400)

#%% 

#compare l'altitude entre les valeurs de Kato et de Blanchard, sur le premier manchot adélie pa18_355z_F1_08
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_4\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'

compare_depth_and_altitude(r"F:\PenguinsData\depth.txt", file_path)   


#%% 

#compare l'altitude entre les valeurs de Kato et de Blanchard, sur le premier manchot adélie pa18_355z_F1_08
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_4\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'


compare_pitch(r"F:\PenguinsData\pitch.txt", file_path)



#%% 

#compare l'altitude entre les valeurs de Kato et de Blanchard, sur le premier manchot adélie pa18_355z_F1_08
file_path = r'F:\PenguinsData\Data_interpolation_acceleration\Data_simu_4\AdeliePenguin\Year1\chick-rearing\pa18_355z_F1_08.csv'


compare_pitch_part(r"F:\PenguinsData\pitch.txt", file_path, 100000, 200000)