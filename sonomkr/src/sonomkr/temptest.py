from filterdesign import FilterDesign
from biquadfilter import BiquadFilter
from spectrumchannel import SpectrumChannel

import unittest
import numpy
import math
import time
import numba
from multiprocessing import Pool
from scipy import signal


def generate_signal(sample_rate, duration, signal_frequency):
    samples_time = numpy.arange(
        int(sample_rate * duration)) / sample_rate
    samples = numpy.sin(2.0 * numpy.pi * signal_frequency * samples_time)
    return samples


def compute_leq(samples):
    samples = samples.astype(float)
    return 20 * math.log10(math.sqrt((samples * samples).sum() / len(samples)))


def process(config):
    ref_filter_config, samples = config
    filtered_samples = samples.astype(numpy.single)
    iir_filter = BiquadFilter(numpy.array(ref_filter_config["b0"]),
                 numpy.array(ref_filter_config["b1"]),
                 numpy.array(ref_filter_config["b2"]),
                 numpy.array(ref_filter_config["a1"]),
                 numpy.array(ref_filter_config["a2"]))
    iir_filter.filter(filtered_samples)
    return compute_leq(filtered_samples)


def process_scipy(config):
    ref_filter_config, samples = config
    sos = numpy.array([[ref_filter_config["b0"][filter_index],
                        ref_filter_config["b1"][filter_index],
                        ref_filter_config["b2"][filter_index],
                        1.0,
                        ref_filter_config["a1"][filter_index],
                        ref_filter_config["a2"][filter_index]]
                       for filter_index in
                       range(len(ref_filter_config["b0"]))])
    filtered = signal.sosfilt(sos, samples)
    return compute_leq(filtered)


def test_sinus():
    f = FilterDesign()
    f.sample_rate = 48000
    f.first_band = -13
    f.last_band = 13
    f.down_sampling = f.G2
    configuration = f.generate_configuration()
    import json
    print(json.dumps(configuration, sort_keys=False, indent=4))
    # generate signal
    samples = generate_signal(f.sample_rate, duration=60,
                              signal_frequency=1000)

    sc = SpectrumChannel(configuration)

    deb = time.time()
    # find appropriate sampling
    stride = int(1/3 * f.sample_rate)
    stride = round(stride / sc.minimum_samples_length)\
             * sc.minimum_samples_length
    spectrums = []
    for sample_index in range(0, len(samples), stride):
        sub_samples = samples[sample_index:sample_index+stride]
        spectrum_dictionary = sc.process_samples(sub_samples)
        spectrum = [spectrum_dictionary[str(frequency_name)] for
                    frequency_name in
                    sorted(map(int, spectrum_dictionary.keys()))]
        spectrums.append(("%.3f s" % ((sample_index + stride) / f.sample_rate), spectrum))
    for t, spectrum in spectrums:
        print(t, spectrum)
    print("numba Done in %.3f" % (time.time() - deb))



test_sinus()



