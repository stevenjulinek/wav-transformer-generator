import shutil
import torch
import os


def import_wav_files(directory):
    # List to hold .wav file paths
    wav_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .wav file
        if filename.endswith(".wav"):
            # Get the full file path
            file_path = os.path.join(directory, filename)
            wav_files.append(file_path)

    return wav_files


def clean_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def count_wavs_in_folder(path):
    number_of_wavs = 0
    for filename in os.listdir(path):
        # Check if the file is a .wav file
        if filename.endswith(".wav"):
            number_of_wavs += 1
    return number_of_wavs


import os
import pathlib

def save_model_with_version(model, base_filename, directory):
    """
    Save the model with a version number.

    Parameters:
    model: The model to be saved.
    base_filename: The base name of the file (without version number).
    directory: The directory where to save the file.
    """
    # Convert the directory to a Path object to handle long paths
    directory = pathlib.Path(directory)

    # Ensure the directory exists
    directory.mkdir(parents=True, exist_ok=True)

    # Find the highest existing version number
    highest_version = 0
    for filename in directory.iterdir():
        if filename.name.startswith(base_filename):
            version = filename.name.rsplit('_v', 1)[-1]
            if version.isdigit():
                highest_version = max(highest_version, int(version))

    # Compute the next version number
    next_version = highest_version + 1

    # Save the model with the next version number
    torch.save(model.state_dict(), directory / f"{base_filename}_v{next_version}.pth")

def get_latest_model(base_filename, directory):
    # Convert the directory to a Path object to handle long paths
    directory = pathlib.Path(directory)

    # Find the highest existing version number
    highest_version = 0
    for filename in directory.iterdir():
        if filename.name.startswith(base_filename):
            version = filename.name.rsplit('_v', 1)[-1]
            if version.isdigit():
                highest_version = max(highest_version, int(version))

    return tf.keras.models.load_model(directory / f"{base_filename}_v{highest_version}")

def find_highest_version(base_filename, directory):
    # Convert the directory to a Path object to handle long paths
    directory = pathlib.Path(directory)

    # Find the highest existing version number
    highest_version = 0
    for filename in directory.iterdir():
        if filename.name.startswith(base_filename):
            version = filename.name.rsplit('_v', 1)[-1]
            if version.isdigit():
                highest_version = max(highest_version, int(version))

    return highest_version

def data_chunk_generator(data_generator, chunk_size):
    data_chunk = []
    for data in data_generator:
        data_chunk.append(data)
        if len(data_chunk) == chunk_size:
            yield data_chunk
            data_chunk = []
    if data_chunk:
        yield data_chunk