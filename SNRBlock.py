
import numpy as np
from gnuradio import gr
from scipy import signal

class blk(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    def __init__(self, sample_start_frequency=None, sample_end_frequency=None, sample_rate=1000000, window_size=1024):  # only default arguments here
        gr.sync_block.__init__(
            self,
            name='SNR Calculator (manual)',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.float32]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.sample_start = sample_start_frequency
        self.sample_end = sample_end_frequency

        self.buffer = np.array([], dtype=np.complex64)

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]

        self.buffer = np.concatenate((self.buffer, in0))

        out_idx = 0
        while len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size]

            f, power_spectrum_linear = signal.welch(window, fs=self.sample_rate, nperseg=256)

            power_spectrum_dB = 10 * np.log10(power_spectrum_linear)

            noise_floor = np.percentile(power_spectrum_linear, 10)
            noise_floor_dB = 10 * np.log10(noise_floor)
            signal_peak_dB = np.max(power_spectrum_dB)
            threshold = noise_floor_dB + (signal_peak_dB - noise_floor_dB) * .15

            signal_mask = power_spectrum_dB > threshold

            power_spectrum_dB_channel_1 = power_spectrum_dB[signal_mask]
            power_spectrum_dB_channel_2 = power_spectrum_dB[~signal_mask]

            power_dB_channel_1 = np.mean(power_spectrum_dB_channel_1)
            power_dB_channel_2 = np.mean(power_spectrum_dB_channel_2)

            try:
                power_dB_signal = 10 * np.log10(10 ** (power_dB_channel_1 / 10) - 10 ** (power_dB_channel_2 / 10))
            except:
                power_dB_signal = float('-inf')  # In case subtraction is zero or negative
            
            snr = power_dB_signal - power_dB_channel_2

            out[out_idx] = snr
            out_idx += 1

            # Slide window forward
            self.printed = True
            self.buffer = self.buffer[self.window_size//2:]  # 50% overlap
        self.consume(0, len(in0))
        return out_idx

