import os
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf
import FolderHandlers


class AudioSlicer:
    def __init__(self, file_path, output_folder_path, clip_length_second, overlay_second=0, sample_rate=8000):
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
            output_file = f"{self.output_folder_path}\\{file_name}_clip_{start_sample // clip_length_samples}.wav"
            output_files.append(output_file)

            # Export as wav
            sf.write(output_file, clip, sr)

            # Move start sample with overlay
            start_sample += (clip_length_samples - overlay_samples)

        return output_files


def prepare_slices(clip_length_sec, overlay_sec, sample_rate):
    print("Starting slicing of input samples. This may take a while.")
    # Usage
    directory = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Wav"
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    FolderHandlers.clean_folder(output_folder)
    wav_files = FolderHandlers.import_wav_files(directory)

    for file in tqdm(wav_files, bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):
        audslice = AudioSlicer(file, output_folder, clip_length_sec, overlay_sec, sample_rate)
        audslice.slice_audio()

    print("Finished slicing samples.")


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
        if (len(wavs) < int(percentage / 100 * number_of_wavs)):
            wavs.append(librosa.load(f"{output_folder}\\{file}", sr=None)[0])
        else:
            break

    return wavs


def create_dequantised_output(quantised_sequence, directory, file_name, num_bins=1024, sample_rate=24000):
    """Dequantise a sequence."""
    # Convert the quantised values back to the range [-1.0, 1.0]
    dequantised_sequence = (quantised_sequence / (num_bins - 1)) * 2.0 - 1.0

    # Save the dequantised sequence as a .wav file
    sf.write(f'{directory}\\{file_name}.wav', dequantised_sequence.flatten(), samplerate=sample_rate)


def save_generated_output(sequence, directory, file_name, sample_rate=24000):
    sf.write(f'{directory}\\{file_name}.wav', sequence.detach().cpu().numpy().flatten(), samplerate=sample_rate)

def load_samples(num_samples, directory):
    files = os.listdir(directory)
    samples = []
    for i in range(num_samples):
        audio = librosa.load(f"{directory}\\{files[i]}", sr=24000)[0]
        samples.append(audio)
    return samples

def load_quantised_samples_generator(percentage):
    print("Loading training data.")
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    number_of_wavs = FolderHandlers.count_wavs_in_folder(output_folder)
    num_samples = int(percentage / 100 * number_of_wavs)

    files = os.listdir(output_folder)
    for i in tqdm(range(num_samples - 1),
                  bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):  # subtract 1 to avoid index out of range for next file
        current_file = files[i]
        next_file = files[i + 1]

        current_audio = librosa.load(f"{output_folder}\\{current_file}", sr=24000)[0]
        next_audio = librosa.load(f"{output_folder}\\{next_file}", sr=24000)[0]

        # Quantise the waveform values into 256 discrete values
        quantised_current_audio = np.digitize(current_audio, np.linspace(-1.0, 1.0, 65535)) - 1
        quantised_next_audio = np.digitize(next_audio, np.linspace(-1.0, 1.0, 65535)) - 1

        yield quantised_current_audio, quantised_next_audio


def load_samples_generator(percentage):
    print("Loading music clips with yield")
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    number_of_wavs = FolderHandlers.count_wavs_in_folder(output_folder)
    num_samples = int(percentage / 100 * number_of_wavs)

    files = os.listdir(output_folder)
    for i in tqdm(range(num_samples - 1),
                  bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):  # subtract 1 to avoid index out of range for next file
        current_file = files[i]
        next_file = files[i + 1]

        current_audio = librosa.load(f"{output_folder}\\{current_file}", sr=24000)[0]
        next_audio = librosa.load(f"{output_folder}\\{next_file}", sr=24000)[0]

        # Normalize the values to the [0, 1] range
        if np.nanmax(current_audio) == np.nanmin(current_audio):
            current_audio = current_audio.clip(0, 1)
        else:
            current_audio = (current_audio - np.nanmin(current_audio)) / (
                        np.nanmax(current_audio) - np.nanmin(current_audio))
        if np.nanmax(next_audio) == np.nanmin(next_audio):
            next_audio = next_audio.clip(0, 1)
        else:
            next_audio = (next_audio - np.nanmin(next_audio)) / (np.nanmax(next_audio) - np.nanmin(next_audio))

        yield current_audio, next_audio

def load_samples_generator_batches(percentage, batch_size):
    print("Loading music clips in batches with yield")
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    all_files = os.listdir(output_folder)
    num_files = len(all_files)
    num_train = int(num_files * percentage / 100)

    # Yield batches of files
    for i in tqdm(range(0, num_train, batch_size), bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):
        batch_files = all_files[i:i + batch_size]
        batch_data = []
        for file in batch_files:
            audio = librosa.load(f"{output_folder}\\{file}", sr=24000)[0]
            # Normalize the values to the [0, 1] range
            if np.nanmax(audio) == np.nanmin(audio):
                audio = audio.clip(0, 1)
            else:
                audio = (audio - np.nanmin(audio)) / (np.nanmax(audio) - np.nanmin(audio))
            batch_data.append(audio)
        yield batch_data
def load_quantised_samples(percentage):
    print("Loading data.")
    output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
    number_of_wavs = FolderHandlers.count_wavs_in_folder(output_folder)
    num_samples = int(percentage / 100 * number_of_wavs)

    files = os.listdir(output_folder)
    quantised_samples = []
    for i in tqdm(range(num_samples - 1),
                  bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):  # subtract 1 to avoid index out of range for next file
        current_file = files[i]

        current_audio = librosa.load(f"{output_folder}\\{current_file}", sr=24000)[0]

        # Quantise the waveform values into 256 discrete values
        quantised_current_audio = np.digitize(current_audio, np.linspace(-1.0, 1.0, 65535)) - 1

        quantised_samples.append(quantised_current_audio)

    return quantised_samples

def pad_sequences(sequences):
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = [np.concatenate((seq, np.zeros(max_length - len(seq)))) for seq in sequences]
    return padded_sequences

