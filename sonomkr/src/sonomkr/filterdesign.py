import scipy.signal as signal
import json

"""
Create Filters parameters according to provided audio signal characteristics and wanted noise indicators.

BSD 3-Clause License

Copyright (c) 2023, University Gustave Eiffel
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__authors__ = ["Valentin Le Bescond, Université Gustave Eiffel",
               "Nicolas Fortin, Université Gustave Eiffel"]
__license__ = "BSD3"


class FilterDesign:
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
            frequency_mid = (self.down_sampling **
                             (x / self.band_division)) * 1000
            frequency_max = (self.down_sampling **
                             (1 / (2 * self.band_division))) * frequency_mid
            frequency_min = (self.down_sampling **
                             (- 1 / (2 * self.band_division))) * frequency_mid
            subsampling_depth = 0
            down_sampling_frequency_div = 10 if self.down_sampling == self.G10\
                else 2
            while self.sample_rate % \
                    down_sampling_frequency_div ** (subsampling_depth+1) == 0\
                    and self.sample_rate / down_sampling_frequency_div **\
                    (subsampling_depth+1) >= frequency_mid * 2:
                subsampling_depth += 1
            frequencies[str(x)] = {
                "center_frequency": frequency_mid,
                "max_frequency": frequency_max,
                "min_frequency": frequency_min,
                "subsampling_depth": subsampling_depth,
                "subsampling_filter_index":
                    str(x + subsampling_depth * (10 if self.down_sampling == self.G10 else 3))
            }

        # Compute bandpass filters
        nyquist = self.sample_rate / 2.0
        for x in frequencies.keys():
            w = [frequencies[x]["min_frequency"] / nyquist,
                 frequencies[x]["max_frequency"] / nyquist]
            w[0] = min(0.99999, max(0.00001, w[0]))
            w[1] = min(0.99999, max(0.00001, w[1]))
            sos_bank = signal.butter(self.filter_order, w, 'bandpass',
                                     False, output='sos')
            frequencies[str(x)]["filters"] = \
                        {"b0": [sos[0] for sos in sos_bank],
                         "b1": [sos[1] for sos in sos_bank],
                         "b2": [sos[2] for sos in sos_bank],
                         "a1": [sos[4] for sos in sos_bank],
                         "a2": [sos[5] for sos in sos_bank]}
        # Compute antialiasing filter
        frequency_aliasing = self.sample_rate / 10 \
            if self.down_sampling == self.G10 else self.sample_rate / 2
        aliasing_sos = signal.butter(self.order_aliasing,
                                     frequency_aliasing / self.sample_rate,
                                     'low', False, output='sos')
        anti_aliasing = {"b0": [sos[0] for sos in aliasing_sos],
                         "b1": [sos[1] for sos in aliasing_sos],
                         "b2": [sos[2] for sos in aliasing_sos],
                         "a1": [sos[4] for sos in aliasing_sos],
                         "a2": [sos[5] for sos in aliasing_sos]}
        anti_aliasing["sample_ratio"] = 10 if self.down_sampling == self.G10 \
            else 2
        return {"bandpass": frequencies, "anti_aliasing": anti_aliasing}

