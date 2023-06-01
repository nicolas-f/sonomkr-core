from .biquadfilter import BiquadFilter
import numpy
import math

""" SpectrumChannel execute the processing of audio samples according to
a configuration file using Cascaded filters

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


def compute_leq(samples):
    samples = samples.astype(float)
    return 20 * math.log10(math.sqrt((samples * samples).sum() / len(samples)))


class SpectrumChannel:
    def __init__(self, configuration, use_scipy=False, use_cascade=True):
        # init sub_samplers with anti aliasing filters parameters
        # Sub samplers goal is only to reduce the sample rate for band pass
        # filter that accept lower frequency rates in order to reduce the
        # computation time.
        self.configuration = configuration
        self.use_scipy = use_scipy
        bp = configuration["bandpass"]
        max_subsampling = max([v["subsampling_depth"] for v in
                               bp])
        if not use_cascade:
            max_subsampling = 0
        self.subsampling_ratio = configuration["anti_aliasing"]["sample_ratio"]
        self.minimum_samples_length = self.subsampling_ratio ** max_subsampling
        ref_filter_config = configuration["anti_aliasing"]
        self.sub_samplers = []
        for i in range(max_subsampling):
            if not self.use_scipy:
                iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                             numpy.array(ref_filter_config["b1"]),
                             numpy.array(ref_filter_config["b2"]),
                             numpy.array(ref_filter_config["a1"]),
                             numpy.array(ref_filter_config["a2"]))
            else:
                iir_filter = numpy.array(
                    [[ref_filter_config["b0"][filter_index],
                        ref_filter_config["b1"][filter_index],
                        ref_filter_config["b2"][filter_index],
                        1.0,
                        ref_filter_config["a1"][filter_index],
                        ref_filter_config["a2"][filter_index]]
                     for filter_index in range(len(ref_filter_config["b0"]))])
            self.sub_samplers.append(iir_filter)

        self.iir_filters = [list() for i in range(max_subsampling + 1)]
        for id_frequency, freq in enumerate(bp):
            if use_cascade:
                ref_filter_config = freq["subsampling_filter"]["sos"]
            else:
                ref_filter_config = freq["sos"]
            if not self.use_scipy:
                iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                                          numpy.array(ref_filter_config["b1"]),
                                          numpy.array(ref_filter_config["b2"]),
                                          numpy.array(ref_filter_config["a1"]),
                                          numpy.array(ref_filter_config["a2"]))
            else:
                iir_filter = numpy.array(
                    [[ref_filter_config["b0"][filter_index],
                        ref_filter_config["b1"][filter_index],
                        ref_filter_config["b2"][filter_index],
                        1.0,
                        ref_filter_config["a1"][filter_index],
                        ref_filter_config["a2"][filter_index]]
                     for filter_index in range(len(ref_filter_config["b0"]))])
            if use_cascade:
                self.iir_filters[freq["subsampling_depth"]]\
                    .append((id_frequency, iir_filter))
            else:
                self.iir_filters[0].append((id_frequency, iir_filter))

    def process_samples(self, samples):
        """
        Compute the leq for provided samples
        :param samples:
        :return:
        """
        if len(samples) % self.minimum_samples_length != 0:
            raise ValueError("Provided samples len should be a factor of "
                             "%d samples" % self.minimum_samples_length)

        last_filter_samples = samples
        leqs = [0 for i in range(len(self.configuration["bandpass"]))]
        for cascade_index, cascade_element in enumerate(self.iir_filters):
            # Use last filter samples into each IIRFilter
            for frequency_id, iir_filter in cascade_element:
                if not self.use_scipy:
                    leqs[frequency_id] = iir_filter.filter_then_leq(
                        last_filter_samples)
                else:
                    from scipy import signal
                    filtered_signal = signal.sosfilt(iir_filter,
                                                     last_filter_samples)
                    leqs[frequency_id] = compute_leq(filtered_signal)

            # sub sampling for next filters (lower frequency)
            if cascade_index < len(self.sub_samplers):
                if not len(last_filter_samples) % self.subsampling_ratio == 0:
                    raise ValueError(
                        "Sub-sampled length is not round %d / %d = %f" %
                        (len(last_filter_samples), self.subsampling_ratio,
                         len(last_filter_samples) / self.subsampling_ratio))
                if not self.use_scipy:
                    next_filter_samples = numpy.zeros(
                        int(len(last_filter_samples) / self.subsampling_ratio))
                    antialiasing = self.sub_samplers[cascade_index]
                    antialiasing.filter_slice(last_filter_samples,
                                              next_filter_samples,
                                              self.subsampling_ratio)
                else:
                    from scipy import signal
                    next_filter_samples = signal.sosfilt(
                        self.sub_samplers[cascade_index],
                        last_filter_samples)
                    next_filter_samples = \
                        next_filter_samples[::self.subsampling_ratio]

                last_filter_samples = next_filter_samples
        return leqs

