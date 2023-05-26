import numpy
from numba import float64    # import the types
from numba import njit, prange, jit

""" A digital biquad filter is a second order recursive linear filter,
 containing two poles and two zeros. "Biquad" is an abbreviation of "bi-quadratic",
  which refers to the fact that in the Z domain, its transfer function is the ratio of two quadratic functions
    
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


@njit(parallel=False)
def filter(delay_1: float64[:], delay_2: float64[:], b0: float64[:],
           b1: float64[:], b2: float64[:], a1: float64[:], a2: float64[:],
           samples: float64[:], samples_out: float64[:]):
    bquad_filter_length = len(b0)
    samples_len = len(samples_out[0])
    for h in prange(bquad_filter_length):
        filter_length = len(b0[h])
        for i in range(samples_len):
            input_acc = samples[i]
            for j in range(filter_length):
                input_acc -= delay_1[h][j] * a1[h][j]
                input_acc -= delay_2[h][j] * a2[h][j]
                output_acc = input_acc * b0[h][j]
                output_acc += delay_1[h][j] * b1[h][j]
                output_acc += delay_2[h][j] * b2[h][j]

                delay_2[h][j] = delay_1[h][j]
                delay_1[h][j] = input_acc

                input_acc = output_acc
            samples_out[h][i] = input_acc

class BiquadFilter:
    def __init__(self, filters: list, samples_len):
        self.delay_1 = numpy.zeros((len(filters), len(filters[0]["b0"])))
        self.delay_2 = numpy.zeros((len(filters), len(filters[0]["b0"])))
        self.b0 = numpy.array([numpy.array(bq_filter["b0"]) for bq_filter in
                               filters])
        self.b1 = numpy.array([numpy.array(bq_filter["b1"]) for bq_filter in
                               filters])
        self.b2 = numpy.array([numpy.array(bq_filter["b2"]) for bq_filter in
                               filters])
        self.a1 = numpy.array([numpy.array(bq_filter["a1"]) for bq_filter in
                               filters])
        self.a2 = numpy.array([numpy.array(bq_filter["a2"]) for bq_filter in
                               filters])
        self.samples_out = numpy.zeros(shape=(len(filters), samples_len))

    def filter(self, samples):
        assert len(samples) == len(self.samples_out[0])
        filter(self.delay_1, self.delay_2, self.b0, self.b1,
                               self.b2, self.a1, self.a2, samples,
                               self.samples_out)
        return self.samples_out


