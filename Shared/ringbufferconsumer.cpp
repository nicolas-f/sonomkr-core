#include "ringbufferconsumer.h"
#include "ringbuffer.h"

template<class T>
RingBufferConsumer<T>::RingBufferConsumer(RingBuffer<T> *buffer, int sizeToRead):
    _buffer(buffer),
    _sizeToRead(sizeToRead)
{
    _readerIndex = _buffer->registerReader();
    _bufferPtr = _buffer->getBufferPtr();
    _bufferSize = _buffer->getBufferSize();
}

template<class T>
RingBufferConsumer<T>::~RingBufferConsumer<T>()
{
    _doRead = false;
    _readThread.join();
}

template<class T>
void RingBufferConsumer<T>::run()
{
    int sizeRed = 0;
    unsigned long readPosition = 0;

    while (_doRead)
    {
        if (_buffer->waitToBeginRead(_readerIndex, _sizeToRead, readPosition))
        {
            sizeRed = processData(readPosition);
            if (sizeRed < 0)
            {
                continue;
            }
            _buffer->endRead(_readerIndex, sizeRed);
        }
    }
}

template<class T>
void RingBufferConsumer<T>::start()
{
    _doRead = true;
    _readThread = thread(&RingBufferConsumer<T>::run, this);
}

template<class T>
void RingBufferConsumer<T>::pause()
{
    _doRead = false;
}
