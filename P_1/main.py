import os
import sys

project_dir = os.path.abspath(os.path.dirname(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.chunks import display_IHDR_chunks_info, load_all_chunks, anonymize_png_chunks, describe_chunk


if __name__ == '__main__':
    from src.fft_display import fft_display

    data_path = os.path.join(os.path.dirname(__file__), "data")
    source_image = f'{data_path}/NewTux.png'
    output_image = f'{data_path}/NewTux_anon.png'

    display_IHDR_chunks_info(source_image)
    fft_display(source_image)

    chunks = load_all_chunks(source_image)
    for chunk in chunks:
        print(describe_chunk(chunk))

    report = anonymize_png_chunks(chunks, output_image)
    print(f"Anonimizacja: kept={report['kept']}, removed={report['removed']}, removed_types={report['removed_types']}")
