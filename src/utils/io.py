import json
import numpy as np
import io

def save_data(data, filename):
    """
    Guarda un diccionario de arrays (incluidos los ndarrays de NumPy) en un archivo JSON.

    :param data: Diccionario de arrays a guardar.
    :param filename: Nombre del archivo donde se guardará el JSON.
    """
    try:
        # Convertir todos los ndarrays a listas
        data_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        
        # Guardar el diccionario serializable en un archivo JSON
        with open(filename, 'w') as file:
            json.dump(data_serializable, file, indent=4)
        
        print(f"Datos guardados exitosamente en {filename}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")