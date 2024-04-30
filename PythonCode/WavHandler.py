import os

import librosa
import numpy as np
import soundfile as sf
import FolderHandlers

class AudioSlicer:
    def __init__(self, file_path, output_folder_path, clip_length_second, overlay_second=0, sample_rate = 8000):
        self.file_path = file_path
        self.output_folder_path = output_folder_path
        self.clip_length_second = clip_length_second
        self.overlay_second = overlay_second
        self.sample_rate = sample_rate

    def slice_audio(self):
        # Load wav file
        audio, sr = librosa.load(self.file_path, sr=self.sample_rate)

        # Parameters
        clip_length_samples = int(sr * self.clip_length_second)  # Convert to samples
        overlay_samples = int(sr * self.overlay_second)  # Convert to samples
        start_sample = 0  # Start sample for slicing

        # List to hold output file paths
        output_files = []

        while start_sample + clip_length_samples < len(audio):
            # Slice audio
            clip = audio[start_sample:start_sample + clip_length_samples]

            # Generate output file path
            file_name = self.file_path.split('\\')[-1]
            output_file = f"{self.output_folder_path}\\{file_name}_clip_{start_sample // sr}.wav"
            output_files.append(output_file)

            # Export as wav
            sf.write(output_file, clip, sr)

            # Move start sample with overlay
            start_sample += (clip_length_samples - overlay_samples)

        return output_files

def prepare_slices(clip_length_sec, overlay_sec, sample_rate):
    # Usage
    directory = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Wav"
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    FolderHandlers.clean_folder(output_folder)
    wav_files = FolderHandlers.import_wav_files(directory)

    for file in wav_files:
        audslice = AudioSlicer(file, output_folder, clip_length_sec, overlay_sec, sample_rate)
        audslice.slice_audio()

def length_of_a_clip(path):
    for filename in os.listdir(path):
        # Check if the file is a .wav file
        if filename.endswith(".wav"):
            audio, sr = librosa.load(f"{path}\\{filename}", sr=None)
            return len(audio)

def return_slices(percentage):
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    number_of_wavs = FolderHandlers.count_wavs_in_folder(output_folder)
    wavs = []
    for file in os.listdir(output_folder):
        if(len(wavs)<int(percentage/100*number_of_wavs)):
            wavs.append(librosa.load(f"{output_folder}\\{file}", sr=None)[0])
        else:
            break

    return wavs

def create_dequantised_output(quantised_sequence, directory, file_name, num_bins=1024, sample_rate = 24000):
    """Dequantise a sequence."""
    # Convert the quantised values back to the range [-1.0, 1.0]
    dequantised_sequence = (quantised_sequence / (num_bins - 1)) * 2.0 - 1.0

    # Save the dequantised sequence as a .wav file
    sf.write(f'{directory}\\{file_name}.wav', dequantised_sequence.flatten(), samplerate=sample_rate)
