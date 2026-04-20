import numpy as np
import scipy.signal

from viewephys.viewer import pgtools


def test_spectrograms(qtbot):
    time = np.arange(2000) * 0.002
    waveform = scipy.signal.chirp(time, f0=4, f1=80, t1=3)
    window = pgtools.ImShowSpectrogram()
    window.set_data(waveform, 500)

    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)


def test_imshow(qtbot):
    time = np.arange(2000) * 0.002
    waveform = scipy.signal.chirp(time, f0=4, f1=80, t1=3)
    fscale, tscale, tf = scipy.signal.spectrogram(
        waveform, fs=500, nperseg=256, nfft=512, window="cosine", noverlap=250
    )

    imshowitem = pgtools.imshow(tf, tscale, fscale)
    qtbot.addWidget(imshowitem.plotwidget)
    qtbot.wait(50)

    waveform = scipy.signal.chirp(time, f0=4, f1=160, t1=3)
    fscale, tscale, tf = scipy.signal.spectrogram(
        waveform, fs=500, nperseg=256, nfft=512, window="cosine", noverlap=250
    )
    imshowitem.set_image(tf, tscale, fscale)
