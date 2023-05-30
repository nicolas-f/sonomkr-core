from biquadfilter import BiquadFilter
import numpy
import math


def compute_leq(samples):
    samples = samples.astype(float)
    return 20 * math.log10(math.sqrt((samples * samples).sum() / len(samples)))

class SpectrumChannel:
    def __init__(self, configuration):
        # init sub_samplers with anti aliasing filters parameters
        # Sub samplers goal is only to reduce the sample rate for band pass
        # filter that accept lower frequency rates in order to reduce the
        # computation time.
        self.configuration = configuration
        bp = configuration["bandpass"]
        max_subsampling = max([v["subsampling_depth"] for k, v in
                               bp.items()])
        self.subsampling_ratio = configuration["anti_aliasing"]["sample_ratio"]
        self.minimum_samples_length = self.subsampling_ratio ** max_subsampling
        ref_filter_config = configuration["anti_aliasing"]
        self.sub_samplers = [
            BiquadFilter(numpy.array(ref_filter_config["b0"]),
                         numpy.array(ref_filter_config["b1"]),
                         numpy.array(ref_filter_config["b2"]),
                         numpy.array(ref_filter_config["a1"]),
                         numpy.array(ref_filter_config["a2"]))
            for i in range(max_subsampling)]

        self.iir_filters = [list() for i in range(max_subsampling + 1)]
        for idfreq, freq in bp.items():
            ref_filter_config = freq["subsampling_filter"]
            iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                                      numpy.array(ref_filter_config["b1"]),
                                      numpy.array(ref_filter_config["b2"]),
                                      numpy.array(ref_filter_config["a1"]),
                                      numpy.array(ref_filter_config["a2"]))
            self.iir_filters[freq["subsampling_depth"]]\
                .append((idfreq, iir_filter))

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
        leqs = {}
        for cascade_index, cascade_element in enumerate(self.iir_filters):
            # Use last filter samples into each IIRFilter
            for frequency_id, iir_filter in cascade_element:
                out_samples = last_filter_samples.copy()
                iir_filter.filter(out_samples)
                leqs[frequency_id] = compute_leq(out_samples)
            # sub sampling for next filters (lower frequency)
            if cascade_index < len(self.sub_samplers):
                next_filter_samples = numpy.zeros(
                    int(len(last_filter_samples) / self.subsampling_ratio))
                antialiasing = self.sub_samplers[cascade_index]
                antialiasing.filter_slice(last_filter_samples,
                                          next_filter_samples,
                                          self.subsampling_ratio)
                last_filter_samples = next_filter_samples
        return leqs

