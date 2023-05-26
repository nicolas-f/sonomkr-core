import scipy.signal as signal
import json

class FilterDesign:
    """
     Create Filters parameters according to provided audio signal characteristics and wanted noise indicators.
    """
    def __init__(self):
        self.sample_rate = 48000
        self.G10 = 10.0 ** (3.0 / 10.0)
        self.G2 = 2.0
        self.down_sampling = self.G10
        self.first_band = -31
        self.last_band = 13
        self.band_division = 3
        self.filter_order = 6
        self.order_aliasing = 20

    def generate_configuration(self):
        """
        :return: Python dictionary to use in audio processing code
        """
        assert self.last_band > self.first_band
        assert self.band_division in [1, 3]
        assert self.filter_order > 0
        assert self.down_sampling in [self.G10, self.G2]
        assert self.sample_rate > 0
        frequencies = {}
        for x in range(self.first_band, self.last_band + 1):
            fmid = (self.down_sampling ** (x / self.band_division)) * 1000
            fmax = (self.down_sampling ** (1 / (2 * self.band_division))) * fmid
            fmin = (self.down_sampling ** (- 1 / (2 * self.band_division))) * fmid
            frequencies[x] = {
                "center_frequency": fmid,
                "max_frequency": fmax,
                "min_frequency": fmin
            }

        # Compute bandpass filters
        nyquist = self.sample_rate / 2.0
        for x in frequencies.keys():
            w = [frequencies[x]["min_frequency"] / nyquist, frequencies[x]["max_frequency"] / nyquist]
            w[0] = min(0.99999, max(0.00001, w[0]))
            w[1] = min(0.99999, max(0.00001, w[1]))
            sos = signal.butter(self.filter_order, w, 'bandpass', False, output='sos')
            frequencies[x]["filters"] = {"b0": sos[0], "b1": sos[1], "b2": sos[2], "a1": sos[4], "a2": sos[5]}

        # Compute antialiasing filter
        frequency_aliasing = self.sample_rate / 10 if self.down_sampling == self.G10 else self.sample_rate / 2
        aliasing_sos = signal.butter(self.order_aliasing, frequency_aliasing / self.sample_rate, 'low', False,
                                     output='sos')
        anti_aliasing = [{"b0": sos[0], "b1": sos[1], "b2": sos[2], "a1": sos[4], "a2": sos[5]} for sos in aliasing_sos]

        return {"bandpass": frequencies, "anti_aliasing": anti_aliasing}


f = FilterDesign()
print(json.dumps(f.generate_configuration(), sort_keys=True, indent=4))
