from filterdesign import FilterDesign
from spectrumchannel import SpectrumChannel
import numpy
import time


def generate_signal(sample_rate, duration, signal_frequency):
    samples_time = numpy.arange(
        int(sample_rate * duration)) / sample_rate
    samples = numpy.sin(2.0 * numpy.pi * signal_frequency * samples_time)
    return samples


def test_sinus():
    f = FilterDesign(sample_rate=32000, first_frequency_band=50,
                     last_frequency_band=16000)
    f.down_sampling = f.G2
    configuration = f.generate_configuration()
    import json
    print(json.dumps(configuration, sort_keys=False, indent=4))

    # generate signal
    samples = generate_signal(f.sample_rate, duration=15,
                              signal_frequency=1000.0)

    sc = SpectrumChannel(configuration, use_scipy=False)

    # heat up process (numba compilation time)
    sc.process_samples(samples)

    deb = time.time()
    # find appropriate sampling
    stride = int(f.sample_rate)
    stride = round(stride / sc.minimum_samples_length)\
             * sc.minimum_samples_length
    spectrums = []
    for sample_index in range(0, len(samples), stride):
        sub_samples = samples[sample_index:sample_index+stride]
        spectrum_dictionary = sc.process_samples(sub_samples)
        spectrum = [spectrum_dictionary[str(frequency_name)] for
                    frequency_name in
                    sorted(map(int, spectrum_dictionary.keys()))]
        spectrums.append(("%.3f s" % ((sample_index + stride) / f.sample_rate)
                          , ", ".join(["%.2f" % spl for spl in spectrum])))
    for t, spectrum in spectrums:
        print(t, spectrum)
    print("Done in %.3f" % (time.time() - deb))


test_sinus()



