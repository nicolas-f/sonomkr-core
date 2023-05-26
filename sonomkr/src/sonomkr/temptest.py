from filterdesign import FilterDesign
from biquadfilter import BiquadFilter
import unittest
import numpy
import math
import time
import numba
from multiprocessing import Pool

def generate_signal(sample_rate, duration, signal_frequency):
    samples_time = numpy.arange(
        int(sample_rate * duration)) / sample_rate
    samples = numpy.sin(2.0 * numpy.pi * signal_frequency * samples_time)
    return samples


def compute_leq(samples):
    return 20 * math.log10(math.sqrt((samples * samples).sum() / len(samples)))


def process(config):
    ref_filter_config, samples = config
    filtered_samples = samples.copy()
    iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                 numpy.array(ref_filter_config["b1"]),
                 numpy.array(ref_filter_config["b2"]),
                 numpy.array(ref_filter_config["a1"]),
                 numpy.array(ref_filter_config["a2"]))
    iir_filter.filter(filtered_samples)
    return compute_leq(filtered_samples)


def test_sinus():
    f = FilterDesign()
    f.sample_rate = 48000
    configuration = f.generate_configuration()
    #import json
    #print(json.dumps(configuration, sort_keys=True, indent=4))
    # generate signal
    samples = generate_signal(f.sample_rate, duration=1,
                              signal_frequency=1000)

    # pick 500Hz filter
    iir_filters = BiquadFilter([bandpass_config["filters"] for bandpass_config
                                in configuration["bandpass"].values()],
                               len(samples))

    deb = time.time()
    for i in range(500):
        filtered = iir_filters.filter(samples)
    print("Done in %.3f" % (time.time() - deb))

    #parallel_filter.parallel_diagnostics(level=4)


test_sinus()



