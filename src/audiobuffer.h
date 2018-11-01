#ifndef AUDIOBUFFER_H
#define AUDIOBUFFER_H

// #define SINE_TEST
// #define SINE_FREQ 1000.0
// #define SINE_GAIN 1.0
// #define SINE_RATE 44100.0

#include <vector>
#include <math.h>

#include <zmqpp/zmqpp.hpp>

#include "ringbuffer.h"
#include "configuration.h"

class AudioBuffer {
private:
    Configuration* config_;
    zmqpp::context* zmq_context_;
    zmqpp::socket zmq_pub_socket_;
    
    unsigned long buffer_size_;
    int nb_channels_;
    std::vector<RingBuffer*> channel_buffers_;

#ifdef SINE_TEST
    double last_time_;
#endif

    float decodeAudio24bit(const char* input_buffer);
    float decodeAudio16bit(const char* input_buffer);

    void pubAudioBuffer(int channel, const float* buffer, const int &buffer_size);

public:
    AudioBuffer(Configuration* config, zmqpp::context* zmq);
    ~AudioBuffer();

    void resetBuffers();
    void writeAudioToBuffers(const char* input_buffer,
                             const int& size_to_write,
                             int& nb_channels,
                             int& format_bit);
    RingBuffer* getChannelBuffer(int channel);
};

#endif // AUDIOBUFFER_H
