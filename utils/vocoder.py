import numpy as np
import librosa
import scipy.signal as signal
import soundfile as sf

def vocoder(modulator_path, output_path, sample_rate=44100, num_bands=20):
    # Step 1: Load the voice file (modulator)
    modulator, sr = librosa.load(modulator_path, sr=sample_rate)

    # Step 2: Generate a sawtooth wave as the carrier
    duration = len(modulator) / sample_rate
    t = np.linspace(0, duration, len(modulator), endpoint=False)
    #carrier = signal.sawtooth(2 * np.pi * 150 * t)  # Sawtooth wave at 150 Hz

    carrier = (
            signal.sawtooth(2 * np.pi * 400 * t) +  # Fundamental frequency
            0.5 * signal.sawtooth(2 * np.pi * 200 * t) +  # First harmonic
            0.25 * signal.sawtooth(2 * np.pi * 300 * t) +  # Second harmonic
            0.125 * signal.sawtooth(2 * np.pi * 100 * t)  # Third harmonic
    )

    # Adjust the band edges to avoid zero frequency
    # Compute band edges safely, avoiding 0 or Nyquist frequency issues
    band_edges = np.linspace(20, sample_rate // 2 - 50, num_bands + 1)  # Start at 20 Hz, end before Nyquist

    # Step 3: Create filter banks and analyze the modulator
    #band_edges = np.linspace(0, sample_rate // 2, num_bands + 1)
    envelopes = []

    for i in range(num_bands):
        # Band-pass filter for each band
        band = signal.butter(4, [band_edges[i] / (sample_rate / 2), band_edges[i + 1] / (sample_rate / 2)], btype='band')
        filtered = signal.filtfilt(*band, modulator)

        # Extract the envelope
        envelope = np.abs(signal.hilbert(filtered))
        envelopes.append(envelope)

    # Step 4: Apply envelopes to the carrier
    vocoded = np.zeros_like(modulator)
    for i, envelope in enumerate(envelopes):
        # Modulate the carrier for each band and combine
        band = signal.butter(4, [band_edges[i] / (sample_rate / 2), band_edges[i + 1] / (sample_rate / 2)], btype='band')
        filtered_carrier = signal.filtfilt(*band, carrier)
        vocoded += envelope * filtered_carrier

    # Normalize the vocoded output
    vocoded = vocoded / np.max(np.abs(vocoded))

    # Step 5: Save the output as a new audio file
    sf.write(output_path, vocoded, sample_rate)
    print(f"Vocoded file saved to {output_path}")

# Example usage
vocoder('voice2.wav', 'vocoded_output6.wav')
