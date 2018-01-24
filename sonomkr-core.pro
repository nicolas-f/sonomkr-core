TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

LIBS += -pthread -lasound -lconfig++ -lzmq -lzmqpp

QMAKE_CXXFLAGS += -O3

SOURCES += main.cpp \
    SignalProcessing/audiocapture.cpp \
    SignalProcessing/biquadfilter.cpp \
    SignalProcessing/iirfilter.cpp \
    Shared/ringbuffer.cpp \
    Shared/ringbufferconsumer.cpp \
    SignalProcessing/audiobuffer.cpp \
    SignalProcessing/sinegenerator.cpp \
    maincontroller.cpp \
    configuration.cpp \
    spectrumchannel.cpp \
    spectrum.cpp \
    SignalProcessing/leqfilter.cpp \
    SignalProcessing/antialiasingfilter.cpp

HEADERS += \
    SignalProcessing/audiocapture.h \
    SignalProcessing/biquadfilter.h \
    SignalProcessing/iirfilter.h \
    Shared/ringbuffer.h \
    Shared/ringbufferconsumer.h \
    SignalProcessing/audiobuffer.h \
    defines.h \
    SignalProcessing/sinegenerator.h \
    maincontroller.h \
    configuration.h \
    spectrumchannel.h \
    spectrum.h \
    SignalProcessing/leqfilter.h \
    SignalProcessing/antialiasingfilter.h \
    Shared/dbus_vtable.hpp

DISTFILES += \
    sonomkr.conf \
    filters.conf
