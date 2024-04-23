import shutil
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