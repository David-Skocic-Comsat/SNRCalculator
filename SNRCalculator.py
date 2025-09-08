import numpy as np
from scipy import signal
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def safe_xml_text(element, default=None):
    return element.text if element is not None else default


def load_iq_data(xml_file_path):
    """Load IQ data from the file specified in the XML metadata"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        data_filename = safe_xml_text(root.find('DataFilename'))
        if data_filename is None:
            raise ValueError("DataFilename not found in XML")

        iq_file_path = xml_file_path.parent / data_filename

        sample_rate_text = safe_xml_text(root.find('Clock'))
        if sample_rate_text is None:
            raise ValueError("Clock (sample rate) not found in XML")
        sample_rate = float(sample_rate_text)

        iq_data = np.fromfile(iq_file_path, dtype=np.float32)
        iq_data = iq_data[::2] + 1j * iq_data[1::2]  # Convert to I + jQ

        return iq_data, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading {xml_file_path.name}: {str(e)}")


def plot_frequency_response(frequencies, power_spectrum_linear, signal_mask, noise_floor_dB, threshold_dB, title="Power Spectral Density"):
    power_spectrum_dB = 10 * np.log10(power_spectrum_linear)
    signal_peak_dB = np.max(power_spectrum_dB)
    min_power_dB = min(power_spectrum_dB)

    if signal_mask.any():
        bw_low = frequencies[signal_mask][0]
        bw_high = frequencies[signal_mask][-1]
        bandwidth = bw_high - bw_low
    else:
        bw_low = bw_high = bandwidth = 0

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, power_spectrum_dB, label='PSD')
    plt.axhline(noise_floor_dB, color='green',
                linestyle='--', label='Noise Floor')
    plt.axhline(signal_peak_dB, color='red',
                linestyle='--', label='Peak Power')
    plt.axhline(threshold_dB, color='yellow',
                linestyle='--', label='Threshold')

    if bandwidth > 0:
        plt.axvline(bw_low, color='purple', linestyle='--',
                    label=f'Start: {bw_low:.0f} Hz')
        plt.axvline(bw_high, color='purple', linestyle='--',
                    label=f'End: {bw_high:.0f} Hz')
        plt.fill_between(frequencies, power_spectrum_dB, y2=min_power_dB, where=(frequencies >= bw_low) & (frequencies <= bw_high),
                         color='orange', alpha=0.3, label=f'Bandwidth ≈ {bandwidth:.0f} Hz')

    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dBm/Hz]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_time_domain(iq_data, sample_rate, title="Time-Domain Signal"):
    duration = len(iq_data) / sample_rate
    time = np.linspace(0, duration, len(iq_data))

    plt.figure(figsize=(12, 4))
    plt.plot(time, np.real(iq_data), label='In-phase (I)')
    plt.plot(time, np.imag(iq_data), label='Quadrature (Q)', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_noise_floor_and_threshold_dB(power_spectrum_linear, threshold_multiplier):
    power_spectrum_dB = 10 * np.log10(power_spectrum_linear)

    noise_floor = np.percentile(power_spectrum_linear, 10)
    noise_floor_dB = 10 * np.log10(noise_floor)
    signal_peak_dB = np.max(power_spectrum_dB)
    threshold_dB = noise_floor_dB + \
        (signal_peak_dB - noise_floor_dB) * threshold_multiplier

    return noise_floor_dB, threshold_dB


def compute_channel_difference(power_spectrum_dB, signal_mask):
    power_spectrum_dB_channel_1 = power_spectrum_dB[signal_mask]
    power_spectrum_dB_channel_2 = power_spectrum_dB[~signal_mask]

    power_dB_channel_1 = np.mean(power_spectrum_dB_channel_1)
    power_dB_channel_2 = np.mean(power_spectrum_dB_channel_2)

    try:
        power_dB_signal = 10 * \
            np.log10(10 ** (power_dB_channel_1 / 10) -
                     10 ** (power_dB_channel_2 / 10))
    except:
        # In case subtraction is zero or negative
        power_dB_signal = float('-inf')

    snr = power_dB_signal - power_dB_channel_2

    print(f"Channel 1 (Signal) Power: {power_dB_channel_1:.2f} dB")
    print(f"Channel 2 (Noise) Power: {power_dB_channel_2:.2f} dB")
    print(f"Signal Power (dB): {power_dB_signal:.2f}")
    print(f'SNR = {snr:.2f}')


def get_power_spectrum(iq_data, sample_rate):
    frequencies, power_spectrum_linear = signal.welch(
        iq_data, fs=sample_rate, nperseg=1024)
    sorted_indices = np.argsort(frequencies)
    frequencies = frequencies[sorted_indices]
    power_spectrum_linear = power_spectrum_linear[sorted_indices]
    power_spectrum_dB = 10 * np.log10(power_spectrum_linear)

    return frequencies, power_spectrum_linear, power_spectrum_dB


def compute_signal_mask(power_spectrum_dB, threshold_dB):
    signal_peak_index = np.argmax(power_spectrum_dB)

    power_before_peak_dB = power_spectrum_dB[:signal_peak_index]
    power_after_peak_dB = power_spectrum_dB[signal_peak_index:]

    start_of_signal = np.where(
        power_before_peak_dB[::-1] <= threshold_dB)[0][0]
    end_of_signal = np.where(power_after_peak_dB <= threshold_dB)[0][0]

    signal_mask = (np.arange(len(power_spectrum_dB)) >= signal_peak_index-start_of_signal) & (
        np.arange(len(power_spectrum_dB)) < signal_peak_index+end_of_signal)

    return signal_mask


def main():
    # xml_path = Path("500MHz_SNR.iq.xml")
    xml_path = Path("adsb_20250821-155416-446.iq.xml")

    if not xml_path.exists():
        print("XML file not found.")
        return

    iq_data, sample_rate = load_iq_data(xml_path)
    frequencies, power_spectrum_linear, power_spectrum_dB = get_power_spectrum(
        iq_data, sample_rate)

    start_idx = np.searchsorted(frequencies, -150000, side='left')
    end_idx = np.searchsorted(frequencies, 150000, side='right')
    selected_values = np.arange(start_idx, end_idx)

    frequencies = frequencies[selected_values]
    power_spectrum_linear = power_spectrum_linear[selected_values]
    power_spectrum_dB = power_spectrum_dB[selected_values]

    noise_floor_dB, threshold_dB = compute_noise_floor_and_threshold_dB(
        power_spectrum_linear, threshold_multiplier=.15)
    signal_mask = compute_signal_mask(power_spectrum_dB, threshold_dB)

    plot_frequency_response(frequencies, power_spectrum_linear, signal_mask,
                            noise_floor_dB, threshold_dB, title=f"PSD of {xml_path.name}")
    # plot_time_domain(iq_data, sample_rate, title=f"Time-Domain View of {xml_path.name}")
    compute_channel_difference(power_spectrum_dB, signal_mask=signal_mask)
    print(threshold_dB)


if __name__ == "__main__":
    main()
