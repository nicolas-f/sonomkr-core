#ifndef AUDIOCAPTURE_H
#define AUDIOCAPTURE_H

#include <thread>
#include <string>
#include <alsa/asoundlib.h>
#include "audiobuffer.h"

using namespace std;

class AudioCapture
{
private:
    AudioBuffer* _audioBuffer;
    thread _captureThread;
    snd_pcm_t* _captureHandle;
    char* _periodBuf;
    string _pcmName;
    int _samplesize, _rate, _framesize, _channels, _periodsize, _periods, _formatBit;
    bool _doCapture;

    snd_pcm_t* open_pcm();

public:
    AudioCapture(AudioBuffer* audioBuffer);
    ~AudioCapture();
    void run();
    void start();
    void stop();
};

#endif // AUDIOCAPTURE_H
