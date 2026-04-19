import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.fft_display import fft_display

data_path = os.path.join(project_root, "data")

black_image = fft_display(f'{data_path}/Black.png') #this is a pith back png image with no watermarks or hidden data
stripe_image = fft_display(f'{data_path}/bws.png') #this file is a screenshot of a black and white striped pattern, so the FFT should show some noise
