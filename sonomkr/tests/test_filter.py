import unittest
import numpy
from sonomkr.filterdesign import FilterDesign
from sonomkr.spectrumchannel import SpectrumChannel, compute_leq
import math


def generate_signal(sample_rate, duration, signal_frequency):
    samples_time = numpy.arange(
        int(sample_rate * duration)) / sample_rate
    samples = numpy.sin(2.0 * numpy.pi * signal_frequency * samples_time)
    return samples

class TestBiQuadFilter(unittest.TestCase):
    def test_sinus(self):
        f = FilterDesign(sample_rate=32000, first_frequency_band=50,
                     last_frequency_band=16000)
        configuration = f.generate_configuration()
        # generate signal
        samples = generate_signal(f.sample_rate, duration=10,
                                  signal_frequency=1000)

        sc = SpectrumChannel(configuration, use_scipy=False)

        filtered_samples = sc.process_samples()

        self.assertAlmostEqual(first=compute_leq(samples),
                               second=compute_leq(filtered_samples),
                               delta=0.004)

        print("original %.3f filtered %.3f" % (compute_leq(samples),
                                               compute_leq(filtered_samples)))
        # pick 500Hz filter
        ref_filter_config = configuration["bandpass"]["-3"]["filters"]
        iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                                  numpy.array(ref_filter_config["b1"]),
                                  numpy.array(ref_filter_config["b2"]),
                                  numpy.array(ref_filter_config["a1"]),
                                  numpy.array(ref_filter_config["a2"]))

        filtered_samples = samples.copy()
        iir_filter.filter(filtered_samples)
        self.assertLess(compute_leq(filtered_samples), -58)


if __name__ == '__main__':
    unittest.main()
