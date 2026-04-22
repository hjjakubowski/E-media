import struct as st 
import zlib


def read_chunk(f):
    length_raw = f.read(4)
    if len(length_raw) != 4:
        raise EOFError("Brak danych na długość chunka")
    (chunk_length,) = st.unpack(">I", length_raw)

    chunk_type = f.read(4)
    if len(chunk_type) != 4:
        raise EOFError("Brak danych na typ chunka")

    chunk_data = f.read(chunk_length)
    if len(chunk_data) != chunk_length:
        raise EOFError("Brak danych chunka")

    crc_raw = f.read(4)
    if len(crc_raw) != 4:
        raise EOFError("Brak CRC")
    (chunk_crc,) = st.unpack(">I", crc_raw)

    calc_crc = zlib.crc32(chunk_type)
    calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF

    if chunk_crc != calc_crc:
        raise ValueError(f"Invalid CRC: read=0x{chunk_crc:08X}, calc=0x{calc_crc:08X}")

    return chunk_type, chunk_data

def display_IHDR_chunks_info(image):
    with open(image, "rb") as f:
        PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
        if f.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        chunk_type, chunk_data = read_chunk(f)

        if chunk_type != b"IHDR":
            raise ValueError(f"The first chunk is not IDHR - {chunk_type!r}")

        if len(chunk_data) != 13:
            raise ValueError(f"IHDR should have 13 byes, read in chunks has {len(chunk_data)} bytes")

        width, height, bit_depth, color_type, compression, filter_method, interlace_method = st.unpack(">IIBBBBB", chunk_data)



    print(f'Width: {width},\n Height: {height},\n Bit depth: {bit_depth},\n '
          f'Color: {color_type},\n Compression: {compression},\n Filter method: {filter_method},\n'
          f' interlace method: {interlace_method}')


def load_all_chunks_and_anonimize(image, output_image):
    with open(image, "rb") as f_in, open(output_image, "wb") as f_out:
        PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
        signature = f_in.read(len(PNG_SIGNATURE))
        if signature != PNG_SIGNATURE:
            raise ValueError("Invalid PNG signature")

        # Zapis początkowej sygnatury do nowego pliku
        f_out.write(signature)

        chunks = []
        while True:
            chunk_type, chunk_data = read_chunk(f_in)
            chunks.append((chunk_type, chunk_data))
            if chunk_type == b'IEND':
                break

        print(f"\n--- Przetwarzanie i anonimizacja pliku: {image} ---")

        for chunk_type, chunk_data in chunks:
            # Pierwsza litera typu określa czy chunk jest krytyczny.
            # W ASCII 5 bit (wartość 32) odróżnia małe i wielkie litery: 0 to wielka litera (critical).
            is_critical = not (chunk_type[0] & 32)

            if is_critical:
                print(f"Zachowuję segment KRYTYCZNY: {chunk_type.decode('ascii')} (Rozmiar: {len(chunk_data)} B)")

                # Przepisywanie chunka
                # 1. Długość danych
                f_out.write(st.pack(">I", len(chunk_data)))
                # 2. Typ chunka
                f_out.write(chunk_type)
                # 3. Dane
                f_out.write(chunk_data)

                calc_crc = zlib.crc32(chunk_type)
                calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF
                f_out.write(st.pack(">I", calc_crc))
            else:
                print(f"Usuwam segment POBOCZNY (ancillary): {chunk_type.decode('ascii')}")

        print("Anonimizacja zakończona sukcesem.\n")
