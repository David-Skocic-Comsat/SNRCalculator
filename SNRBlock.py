import numpy as np
from gnuradio import gr
from scipy import signal


class blk(gr.basic_block):
    def __init__(self, sample_rate=1000000, window_size=1024):
        gr.sync_block.__init__(
            self,
            name='SNR Calculator',
            in_sig=[np.complex64],  # Take in complex iq data
            out_sig=[np.float32]    # output snr as a float
        )

        self.sample_rate = sample_rate
        self.window_size = window_size

        self.iq_buffer = np.array([], dtype=np.complex64)

    def get_power_spectrum(self, iq_data):
        frequencies, power_spectrum_linear = signal.welch(
            iq_data, fs=self.sample_rate, nperseg=256, return_onesided=False)
        sorted_indices = np.argsort(frequencies)
        self.frequencies = frequencies[sorted_indices]
        self.power_spectrum_linear = power_spectrum_linear[sorted_indices]
        self.power_spectrum_dB = 10 * np.log10(self.power_spectrum_linear)

    def compute_noise_floor_and_threshold_dB(self, threshold_multiplier=0.15):
        self.noise_floor = np.percentile(self.power_spectrum_linear, 10)
        self.noise_floor_dB = 10 * np.log10(self.noise_floor)
        self.signal_peak_dB = np.max(self.power_spectrum_dB)
        self.threshold_dB = self.noise_floor_dB + \
            (self.signal_peak_dB - self.noise_floor_dB) * threshold_multiplier

    def compute_signal_mask(self):
        signal_peak_index = np.argmax(self.power_spectrum_dB)

        power_before_peak_dB = self.power_spectrum_dB[:signal_peak_index]
        power_after_peak_dB = self.power_spectrum_dB[signal_peak_index:]

        start_of_signal = np.where(
            power_before_peak_dB[::-1] <= self.threshold_dB)[0][0]
        end_of_signal = np.where(
            power_after_peak_dB <= self.threshold_dB)[0][0]

        self.signal_mask = (np.arange(len(self.power_spectrum_dB)) >= signal_peak_index -
                            start_of_signal) & (np.arange(len(self.power_spectrum_dB)) < signal_peak_index+end_of_signal)

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        self.iq_buffer = np.concatenate((self.iq_buffer, in0))

        out_idx = 0

        while len(self.iq_buffer) >= self.window_size:
            window = self.iq_buffer[:self.window_size]

            self.get_power_spectrum(window)
            self.compute_noise_floor_and_threshold_dB()
            self.compute_signal_mask()

            power_spectrum_dB_channel_1 = self.power_spectrum_dB[self.signal_mask]
            power_spectrum_dB_channel_2 = self.power_spectrum_dB[~self.signal_mask]

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

            out[out_idx] = snr
            out_idx += 1

            # Slide window forward
            self.printed = True
            # 50% overlap
            self.iq_buffer = self.iq_buffer[self.window_size//2:]
        self.consume(0, len(in0))
        return out_idx
