import unittest

try:
    from .test_fft import FftRoundTripTest
except ImportError:
    from test_fft import FftRoundTripTest


if __name__ == "__main__":
    unittest.main()
