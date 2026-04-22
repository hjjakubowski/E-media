import os
import sys
from src.chunks import display_IHDR_chunks_info, load_all_chunks_and_anonimize
from src.fft_display import fft_display
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), "data")

    display_IHDR_chunks_info(f'{data_path}/PWr.png')
    fft_display(f'{data_path}/PWr_gray.png')
    load_all_chunks_and_anonimize(f'{data_path}/PWr.png', f'{data_path}/PWr_anon.png')


