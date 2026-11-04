"""Image quality metrics."""
import numpy as np

BORDER = 8  # ignore a frame of pixels where border clamping distorts results


def psnr(img, ref, border=BORDER):
    """Peak signal-to-noise ratio in dB, for images in [0, 1]. Higher is better;
    +6 dB means the RMS error halved."""
    sl = (slice(border, -border), slice(border, -border))
    mse = np.mean((img[sl] - ref[sl]) ** 2)
    return 10 * np.log10(1.0 / mse)
