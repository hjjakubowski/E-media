# Dokumentacja projektu P_1

## Cel projektu 

Projekt realizuje temat z pliku `E_media_projekt_1-1.pdf`: analiza wybranego formatu pliku multimedialnego. W naszej implementacji wybranym formatem jest PNG.

Najwazniejsze wymagania z PDF, ktore sa bezposrednio powiazane z kodem:

- reczne odczytanie informacji z formatu PNG na podstawie kolejnych bajtow pliku;
- pokazanie atrybutow pliku, w szczegolnosci rozmiaru, glebi koloru i informacji z naglowka;
- prezentacja pliku jako obrazu;
- wyznaczenie i pokazanie widma Fouriera, z uwzglednieniem pelniejszej informacji: modul/log-widmo oraz faza;
- zaproponowanie i zaimplementowanie sposobu testowania poprawnosci FFT przez transformacje odwrotna IFFT;
- anonimizacja pliku, czyli usuwanie zbednych informacji bez zmiany obrazu;
- wczytanie i wyswietlenie zawartosci critical chunks oraz czyszczenie informacji dodatkowych.


## Struktura projektu

- `main.py` - prosty skrypt uruchomieniowy dla analizy przykladowego pliku.
- `app/__main__.py` - uruchomienie aplikacji GUI przez `python -m app`.
- `app/gui.py` - aplikacja PySide6: wybor plikow, analiza, prezentacja wynikow, anonimizacja.
- `src/chunks.py` - reczny parser PNG, opis chunkow, zapis chunkow i anonimizacja.
- `src/fft_models.py` - struktury danych z wynikami FFT i metrykami.
- `src/fft_contracts.py` - protokoly/kontrakty dla loadera, analizatora i prezentera.
- `src/fft_services.py` - wczytanie obrazu OpenCV, obliczenia FFT/IFFT, metryki bledow.
- `src/fft_presenters.py` - prezentacja wynikow FFT w Matplotlib.
- `src/fft_display.py` - prosty interfejs laczacy analize FFT i presenter Matplotlib.
- `test/test_fft.py` - testy jednostkowe FFT.
- `test/test_chunks.py` - testy parsera PNG i anonimizacji.
- `test/fft_test.py` - wrapper dla starszego sposobu uruchamiania testow.

## Podstawy formatu PNG uzyte w kodzie

PNG zaczyna sie 8-bajtowa sygnatura:

```text
89 50 4E 47 0D 0A 1A 0A
```

W kodzie jest ona zapisana jako:

```python
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
```

Po sygnaturze plik sklada sie z kolejnych chunkow. Kazdy chunk ma stala strukture:

```text
4 bajty: length      dlugosc pola data, big-endian unsigned int
4 bajty: chunk type  nazwa chunka, np. IHDR, PLTE, IDAT, IEND, tEXt
N bajtow: data       dane chunka, gdzie N = length
4 bajty: CRC         suma kontrolna liczona po chunk type + data
```

PNG rozroznia critical chunks i ancillary chunks. Kod robi to zgodnie z bitem wielkosci pierwszej litery nazwy chunka:

```python
is_critical = not (chunk_type[0] & 32)
```

Dlaczego to dziala:

- w ASCII wielka litera `A` ma wartosc 65, a mala `a` ma wartosc 97;
- roznica miedzy nimi to bit `32`;
- jesli pierwszy znak typu chunka jest wielki, chunk jest krytyczny;
- jesli pierwszy znak jest maly, chunk jest pomocniczy, czyli ancillary.

Przyklady:

- `IHDR`, `PLTE`, `IDAT`, `IEND` - critical chunks;
- `tEXt`, `tIME`, `pHYs`, `gAMA`, `sBIT` - ancillary chunks.

## Modul `src/chunks.py`

Ten modul realizuje najwazniejsza czesc wymagana przez PDF: reczne czytanie PNG z bajtow. Nie korzysta z gotowego parsera PNG do metadanych, tylko sam otwiera plik binarnie, czyta bajty, interpretuje pola i sprawdza CRC.

### Stale

#### `PNG_SIGNATURE`

8 bajtow identyfikujacych plik PNG. Funkcje `display_IHDR_chunks_info` i `load_all_chunks` porownuja poczatek pliku z ta wartoscia. Jezeli sygnatura sie nie zgadza, plik nie jest poprawnym PNG i kod rzuca `ValueError`.

#### `MAX_CHUNK_LENGTH`

Limit bezpieczenstwa ustawiony na `256 * 1024 * 1024` bajtow. PNG przechowuje dlugosc chunka w 4 bajtach, wiec teoretycznie chunk moze deklarowac bardzo duza dlugosc. Bez limitu zlosliwy plik moglby zmusic program do prob odczytu ogromnej ilosci danych.

#### `RENDERING_RELEVANT_ANCILLARY_CHUNKS`

Zbior ancillary chunks, ktore kod zachowuje podczas anonimizacji, bo moga wplywac na wyglad obrazu:

- `bKGD` - sugerowane tlo;
- `cHRM` - chromatycznosc;
- `gAMA` - gamma;
- `iCCP` - profil ICC;
- `sBIT` - istotne bity oryginalnych probek;
- `sRGB` - standardowa przestrzen barw;
- `tRNS` - przezroczystosc.

To jest kompromis miedzy wymaganiem czyszczenia dodatkowych informacji a wymaganiem anonimizacji bez ingerencji w obraz. Kod usuwa typowe metadane, np. `tEXt` i `tIME`, ale nie usuwa informacji, ktore moga zmienic renderowanie pikseli.

#### `COLOR_TYPES`

Mapa kodow typu koloru z `IHDR` na opis tekstowy:

- `0` - grayscale;
- `2` - truecolor;
- `3` - indexed-color;
- `4` - grayscale with alpha;
- `6` - truecolor with alpha.

Te wartosci sa standardowymi typami koloru w PNG i sa pokazywane uzytkownikowi przy opisie naglowka.

### Klasa `PngChunk`

`PngChunk` jest dataclass reprezentujaca jeden chunk PNG po recznym odczytaniu z pliku.

Pola:

- `chunk_type` - 4-bajtowa nazwa chunka, np. `b"IHDR"`;
- `data` - surowe dane chunka;
- `length` - dlugosc danych z pola length;
- `crc_read` - CRC odczytane z pliku;
- `crc_calc` - CRC policzone przez program;
- `is_critical` - `True`, jesli chunk jest krytyczny;
- `index` - numer chunka w kolejnosci odczytu;
- `offset` - pozycja w pliku, od ktorej zaczyna sie chunk, czyli offset pola length;
- `data_offset` - pozycja w pliku, od ktorej zaczynaja sie dane chunka;
- `trailing_data` - bajty znalezione po `IEND`; normalny PNG nie powinien ich potrzebowac do obrazu.

Pole `offset` jest wazne w kontekscie PDF, bo dokument wspomina o mozliwosci zapisywania informacji w sposobie konstrukcji pliku, np. przez offsety i dane za obrazem.

### `read_chunk(f)`

Funkcja czyta dokladnie jeden chunk z otwartego pliku binarnego.

Kolejne kroki:

1. Czyta 4 bajty `length_raw`.
2. Sprawdza, czy faktycznie dostala 4 bajty. Jesli nie, plik jest uciety.
3. Interpretuje dlugosc przez `struct.unpack(">I", length_raw)`.
4. `">I"` oznacza:
   - `>` - big-endian, czyli najstarszy bajt pierwszy;
   - `I` - unsigned 32-bit integer.
5. Sprawdza, czy dlugosc nie przekracza `MAX_CHUNK_LENGTH`.
6. Czyta 4 bajty typu chunka.
7. Sprawdza, czy typ ma tylko litery ASCII. PNG wymaga liter w nazwie chunka.
8. Czyta `chunk_length` bajtow danych.
9. Czyta 4 bajty CRC.
10. Oblicza CRC samodzielnie:

```python
calc_crc = zlib.crc32(chunk_type)
calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF
```

CRC jest liczone nie po polu length, tylko po `chunk_type + chunk_data`. Maska `& 0xFFFFFFFF` zapewnia wynik jako 32-bitowa wartosc bez znaku.

11. Porownuje `chunk_crc` z `calc_crc`.
12. Jesli CRC sie nie zgadza, rzuca `ValueError`.
13. Zwraca: typ, dane, dlugosc, CRC odczytane i CRC obliczone.

To jest centralna funkcja recznego dekodowania PNG. Kazdy kolejny element pliku jest analizowany na podstawie bajtow, zgodnie z wymaganiem PDF.

### `parse_ihdr(chunk_data)`

Funkcja interpretuje dane chunka `IHDR`.

`IHDR` musi miec dokladnie 13 bajtow:

```text
4 bajty width
4 bajty height
1 bajt bit_depth
1 bajt color_type
1 bajt compression
1 bajt filter_method
1 bajt interlace_method
```

Kod sprawdza dlugosc:

```python
if len(chunk_data) != 13:
    raise ValueError(...)
```

Nastepnie rozpakowuje dane:

```python
width, height, bit_depth, color_type, compression, filter_method, interlace_method = st.unpack(
    ">IIBBBBB", chunk_data
)
```

Znaczenie formatu `">IIBBBBB"`:

- `>` - big-endian;
- `I` - 4-bajtowa liczba calkowita bez znaku;
- `I` - druga 4-bajtowa liczba calkowita bez znaku;
- `B` - pojedynczy bajt bez znaku;
- kolejne `B` - nastepne pojedyncze pola.

Funkcja zwraca slownik z wartosciami i tekstowa nazwa typu koloru. To spelnia wymaganie pokazania rozmiaru i glebi koloru.

### `display_IHDR_chunks_info(image)`

Funkcja wypisuje podstawowe informacje z `IHDR`.

Kroki:

1. Otwiera plik w trybie binarnym `rb`.
2. Czyta pierwsze 8 bajtow i porownuje z `PNG_SIGNATURE`.
3. Czyta pierwszy chunk przez `read_chunk`.
4. Sprawdza, czy pierwszy chunk to `IHDR`. W poprawnym PNG `IHDR` musi byc pierwszy.
5. Parsuje dane przez `parse_ihdr`.
6. Wypisuje:
   - szerokosc;
   - wysokosc;
   - glebie bitowa;
   - typ koloru;
   - metode kompresji;
   - metode filtrowania;
   - metode interlace.

Ta funkcja jest wykorzystywana w GUI do sekcji "Naglowek IHDR".

### `load_all_chunks(image)`

Funkcja czyta wszystkie chunki PNG od poczatku do `IEND`.

Kroki:

1. Otwiera plik binarnie.
2. Sprawdza sygnature PNG.
3. Tworzy pusta liste `chunks`.
4. Ustawia `index = 0`.
5. W petli:
   - zapamietuje `offset = f.tell()`, czyli pozycje przed odczytem chunka;
   - wywoluje `read_chunk`;
   - oblicza `is_critical`;
   - tworzy obiekt `PngChunk`;
   - ustawia `data_offset = offset + 8`, bo dane zaczynaja sie po 4 bajtach length i 4 bajtach type;
   - dodaje chunk do listy;
   - jesli typ to `IEND`, czyta jeszcze wszystkie pozostale bajty do `trailing_data` i konczy petle;
   - w przeciwnym razie zwieksza indeks.

Dzieki temu program nie tylko widzi standardowe chunki, ale tez moze wykryc ukryte bajty po `IEND`. To jest wazne dla anonimizacji i rozmowy o mozliwosci ukrywania informacji w konstrukcji pliku.

### `decode_time(data)`

Funkcja dekoduje ancillary chunk `tIME`.

`tIME` ma dokladnie 7 bajtow:

```text
2 bajty year
1 bajt month
1 bajt day
1 bajt hour
1 bajt minute
1 bajt second
```

Kod:

```python
year, month, day, hour, minute, second = st.unpack(">HBBBBB", data)
```

`H` oznacza 2-bajtowa liczbe bez znaku. Funkcja zwraca tekst w formacie `YYYY-MM-DD HH:MM:SS`. Jesli dlugosc danych nie wynosi 7, zwraca `None`.

### `decode_text(data)`

Funkcja dekoduje ancillary chunk `tEXt`.

Struktura `tEXt`:

```text
keyword\0text
```

Kod dzieli dane po pierwszym bajcie zerowym:

```python
parts = data.split(b"\x00", 1)
```

Jesli nie ma separatora, funkcja zwraca `None`. Jesli separator istnieje:

- pierwsza czesc to klucz, np. `Author`;
- druga czesc to wartosc tekstowa.

PNG `tEXt` uzywa kodowania Latin-1, dlatego kod robi:

```python
decode("latin-1", errors="replace")
```

`errors="replace"` zapobiega wyjatkom przy nietypowych bajtach; niepoprawne znaki zostana zastapione.

### `decode_exif_summary(data)`

Funkcja robi podstawowe rozpoznanie ancillary chunka `eXIf`. Nie dekoduje wszystkich tagow EXIF, tylko pokazuje informacje podsumowujace.

Kroki:

1. Sprawdza, czy danych jest co najmniej 8 bajtow.
2. Czyta dwa pierwsze bajty endian:
   - `II` oznacza little-endian;
   - `MM` oznacza big-endian.
3. Czyta magic number z bajtow 2-3. Poprawny TIFF/EXIF ma wartosc `42`.
4. Czyta offset do pierwszego IFD z bajtow 4-7.
5. Sprawdza, czy offset miesci sie w danych.
6. Czyta liczbe tagow z pierwszych 2 bajtow IFD.
7. Zwraca slownik z endian i liczba tagow.

Jest to celowo podsumowanie, a nie pelny parser EXIF. Kod opisuje istniejaca funkcjonalnosc bez dodawania nowego zakresu.

### `decode_plte_summary(data)`

Funkcja dekoduje critical chunk `PLTE`, czyli palete kolorow.

`PLTE` sklada sie z kolejnych trojek bajtow:

```text
R G B | R G B | R G B | ...
```

Dlatego dlugosc danych musi byc dodatnia i podzielna przez 3. Jesli nie jest, funkcja zwraca `None`.

Petla:

```python
for i in range(0, len(data), 3):
    colors.append(tuple(data[i : i + 3]))
```

czyta kazde 3 bajty jako jedna krotke `(R, G, B)`. Funkcja zwraca:

- liczbe wpisow palety;
- liste wszystkich kolorow.

To jest wazne dla wymagania pokazania critical chunks: `PLTE` nie jest tylko pokazywany jako liczba bajtow, ale jako zawartosc palety.

### `bytes_to_hex(data, limit=16)`

Funkcja zamienia bajty na tekst szesnastkowy, np.:

```text
89 50 4E 47
```

Jesli `limit` jest liczba, pokazuje tylko pierwsze `limit` bajtow i dodaje `...`, gdy danych jest wiecej. Jesli `limit=None`, pokazuje wszystkie bajty. Jest to uzywane do prezentacji danych chunkow w konsoli GUI.

### `chunk_sha256(data)`

Funkcja liczy SHA-256 danych chunka.

W projekcie uzywa sie tego glownie dla `IDAT`, bo `IDAT` moze byc bardzo duzy. Pokazanie calego hexa jest mozliwe, ale hash jest wygodnym skrotem do sprawdzenia, czy dane obrazu zostaly zachowane.

### `should_keep_chunk_for_anonymization(chunk)`

Funkcja okresla, czy chunk powinien zostac przepisany do pliku anonimowego.

Zwraca `True`, gdy:

- chunk jest critical, czyli jest potrzebny do poprawnego PNG;
- albo chunk jest ancillary, ale znajduje sie w `RENDERING_RELEVANT_ANCILLARY_CHUNKS`.

Uzasadnienie projektowe:

- PDF wymaga usuwania zbednych informacji;
- PDF wymaga anonimizacji bez ingerencji w obraz;
- nie kazdy ancillary chunk jest zbedny dla wygladu;
- dlatego kod usuwa metadane, ale zachowuje pomocnicze chunki mogace wplywac na renderowanie.

### `describe_chunk(chunk)`

Funkcja tworzy czytelny opis jednego chunka.

Najpierw tworzy czesc wspolna:

```text
[index] NAME (critical/ancillary, LENGTH B, offset=..., data_offset=..., crc=...)
```

Nastepnie zalezne od typu chunka:

- `IHDR` - dekoduje wszystkie pola naglowka przez `parse_ihdr`;
- `PLTE` - pokazuje liczbe wpisow palety, wszystkie kolory RGB i hex danych;
- `IDAT` - pokazuje, ze sa to skompresowane dane obrazu, SHA-256 i hex danych;
- `IEND` - pokazuje, czy chunk jest pusty oraz ile bajtow jest po `IEND`;
- `tIME` - pokazuje czas z `decode_time`;
- `tEXt` - pokazuje klucz i wartosc tekstowa;
- `eXIf` - pokazuje endian i liczbe tagow;
- inne critical chunks - pokazuje pelny hex danych;
- inne ancillary chunks - pokazuje ograniczony podglad hexa.

W kontekscie PDF funkcja jest odpowiedzialna za "wyswietlenie zawartosci segmentow".

### `write_chunk(f_out, chunk_type, chunk_data)`

Funkcja zapisuje jeden chunk PNG do pliku wyjsciowego.

Kolejne kroki:

1. Zapisuje dlugosc danych jako 4 bajty big-endian:

```python
f_out.write(st.pack(">I", len(chunk_data)))
```

2. Zapisuje 4 bajty typu chunka.
3. Zapisuje dane chunka.
4. Liczy CRC po `chunk_type + chunk_data`.
5. Zapisuje CRC jako 4 bajty big-endian.

Ta funkcja jest odwrotnoscia `read_chunk` dla zapisu. Jest uzywana przez anonimizacje i testy.

### `anonymize_png_chunks(chunks, output_image)`

Funkcja tworzy nowy plik PNG zanonimizowany na podstawie listy chunkow.

Kroki:

1. Ustawia liczniki:
   - `kept` - ile chunkow zostalo zachowanych;
   - `removed` - ile usunieto;
   - `removed_types` - ile usunieto chunkow kazdego typu;
   - `kept_types` - ile zachowano chunkow kazdego typu;
   - `kept_rendering_ancillary_types` - jakie pomocnicze chunki zachowano ze wzgledu na wyglad;
   - `trailing_removed` - ile bajtow po `IEND` usunieto.
2. Otwiera plik wyjsciowy w trybie `wb`.
3. Zapisuje sygnature PNG.
4. Iteruje po chunkach:
   - jesli `should_keep_chunk_for_anonymization` zwraca `True`, zapisuje chunk przez `write_chunk`;
   - jesli chunk to `IDAT`, zapisuje jego SHA-256 do raportu;
   - jesli chunk to `IEND`, liczy bajty po `IEND` jako usuniete;
   - jesli chunk nie powinien zostac, nie zapisuje go i zwieksza liczniki usunietych typow.
5. Zwraca raport.

Co jest usuwane:

- typowe tekstowe metadane, np. `tEXt`;
- czas modyfikacji `tIME`;
- inne pomocnicze chunki nieuznane za istotne dla renderowania;
- bajty dopisane po `IEND`.

Co jest zachowywane:

- critical chunks, np. `IHDR`, `PLTE`, `IDAT`, `IEND`;
- pomocnicze chunki wplywajace na wyglad, np. `gAMA`, `sBIT`, `tRNS`.

Dlaczego obraz nie powinien sie zmienic:

- dane `IDAT`, czyli skompresowane dane pikseli, sa przepisywane bez zmian;
- `PLTE` jest zachowany, bo jest critical dla obrazow paletowych;
- pomocnicze informacje o renderowaniu nie sa usuwane;
- dane za `IEND` nie sa czescia obrazu.

### `load_all_chunks_and_anonimize(image, output_image)`

Funkcja pomocnicza laczaca dwa kroki:

1. `load_all_chunks(image)`;
2. `anonymize_png_chunks(chunks, output_image)`.

Nazwa zawiera literowke `anonimize`, ale funkcja dziala jako prosty wrapper.

## Modul `src/fft_models.py`

Ten modul zawiera struktury danych, ktore przenosza wyniki obliczen FFT/IFFT.

### `ChannelMetrics`

Metryki dla jednego kanalu obrazu.

Pola:

- `channel_index` - indeks kanalu, np. `0` dla grayscale albo czerwonego po konwersji RGB;
- `mse` - mean squared error;
- `rmse` - root mean squared error;
- `mae` - mean absolute error;
- `max_abs_error` - najwiekszy blad bezwzgledny;
- `psnr_db` - PSNR w decybelach.

### `FftMetricsSummary`

Podsumowanie metryk dla calego obrazu.

Pola:

- `mse_mean` - srednia MSE po kanalach;
- `rmse_mean` - srednia RMSE po kanalach;
- `mae_mean` - srednia MAE po kanalach;
- `max_abs_error` - maksimum z kanalowych bledow maksymalnych;
- `psnr_mean_db` - srednia PSNR z kanalow, z pominieciem wartosci nieskonczonych, jezeli istnieja skonczone;
- `passed` - czy blad maksymalny miesci sie w tolerancji;
- `tolerance` - tolerancja porownania.

### `FftAnalysisResult`

Pelny wynik analizy FFT.

Pola:

- `mode` - `gray`, `rgb` albo `rgba`;
- `original` - obraz po wczytaniu i ewentualnej konwersji BGR/BGRA do RGB/RGBA;
- `spectrum_log_display` - logarytmiczny modul widma;
- `phase_display` - faza widma w zakresie od `-pi` do `pi`;
- `reconstructed` - obraz po IFFT, przywrocony do oryginalnego typu danych;
- `error_map` - mapa bledow bezwzglednych;
- `channel_metrics` - metryki per kanal;
- `summary` - podsumowanie metryk.

#### `reconstructed_uint8`

Wlasciwosc kompatybilnosci. Jesli `reconstructed` juz jest `uint8`, zwraca go bez zmian. W przeciwnym razie obcina wartosci do zakresu `0..255` i rzutuje na `uint8`. Do obliczen poprawnosci nalezy uzywac `reconstructed`, bo zachowuje np. `uint16`.

## Modul `src/fft_services.py`

Ten modul odpowiada za obliczenia Fouriera i testowanie poprawnosci transformacji przez IFFT.

### `to_rgb_or_gray(image)`

OpenCV wczytuje obrazy kolorowe jako BGR albo BGRA, a typowy zapis do prezentacji to RGB/RGBA. Funkcja normalizuje kolejnosc kanalow:

- obraz 2D zostaje bez zmian jako grayscale;
- obraz 3D z 4 kanalami jest konwertowany `BGRA -> RGBA`;
- obraz 3D z 3 kanalami jest konwertowany `BGR -> RGB`;
- inne ksztalty powoduja `ValueError`.

### `image_mode(image)`

Funkcja zwraca tekstowy tryb obrazu na podstawie wymiarow:

- 2 wymiary - `gray`;
- 3 wymiary i 3 kanaly - `rgb`;
- 3 wymiary i 4 kanaly - `rgba`.

Tryb jest potem uzywany przez GUI i Matplotlib do doboru sposobu wyswietlania.

### `Cv2ImageLoader.load(image_path)`

Wczytuje obraz przez:

```python
cv.imread(image_path, cv.IMREAD_UNCHANGED)
```

`IMREAD_UNCHANGED` jest wazne, bo zachowuje:

- kanal alpha;
- oryginalna glebie bitowa, np. `uint16`;
- obraz grayscale bez wymuszania RGB.

Jesli OpenCV nie potrafi wczytac pliku, funkcja rzuca `ValueError`.

### `NumpyRoundTripFftAnalyzer.__init__(tolerance)`

Ustawia tolerancje dla testu FFT/IFFT. Domyslnie `1e-6`.

Tolerancja jest potrzebna, poniewaz FFT i IFFT sa obliczeniami zmiennoprzecinkowymi. Nawet gdy matematycznie obraz powinien wrocic identycznie, moga pojawic sie bardzo male bledy numeryczne rzedu `1e-12`.

### `_data_range(channel)`

Funkcja wyznacza zakres wartosci uzywany do PSNR.

Dla typow calkowitych:

```python
np.iinfo(channel.dtype).max
```

Przyklady:

- `uint8` - zakres maksymalny 255;
- `uint16` - zakres maksymalny 65535.

Dla typow zmiennoprzecinkowych funkcja bierze skonczone wartosci i liczy:

```python
max_value - min_value
```

Minimum zakresu to 1.0, zeby uniknac logarytmu z zera.

### `_reconstruct_to_source_dtype(reconstructed, dtype)`

IFFT zwraca wartosci zmiennoprzecinkowe. Funkcja przywraca typ danych obrazu.

Dla typow calkowitych:

1. Bierze zakres typu, np. `0..255` dla `uint8`.
2. Zaokragla wartosci przez `np.rint`.
3. Obcina do dozwolonego zakresu przez `np.clip`.
4. Rzutuje na oryginalny typ.

Dla typow zmiennoprzecinkowych tylko rzutuje na docelowy typ.

To naprawia problem utraty informacji dla obrazow 16-bitowych: wynik nie jest juz zawsze sprowadzany do `uint8`.

### `_normalize_frequency_stack(values)`

Normalizuje log-widmo dla obrazow wielokanalowych.

Kod:

```python
max_value = float(np.max(values))
if max_value > 0:
    return values / max_value
return values
```

Jezeli widmo ma dodatnie maksimum, dzieli wszystkie wartosci przez maksimum. Wtedy dane trafiajace do wyswietlenia mieszcza sie w zakresie `0..1`. Jezeli maksimum jest zerowe, funkcja zwraca wartosci bez zmian.

### `_channel_analysis(channel, channel_index)`

Najwazniejsza funkcja obliczeniowa FFT dla jednego kanalu.

Kroki:

1. Konwersja kanalu do `float64`:

```python
channel_org = channel.astype(np.float64)
```

FFT wymaga obliczen numerycznych; `float64` daje stabilnosc i precyzje.

2. Obliczenie 2D FFT:

```python
freq = np.fft.fft2(channel_org)
```

Dla obrazu 2D `f(x, y)` wynik `freq(u, v)` jest zespolony. Kazdy punkt widma ma:

- modul - sila danej czestotliwosci;
- faze - przesuniecie skladnika sinusoidalnego.

3. Przesuniecie zera czestotliwosci do srodka:

```python
freq_shift = np.fft.fftshift(freq)
```

Bez `fftshift` skladowa DC, czyli srednia jasnosc obrazu, znajduje sie w rogu tablicy. Po `fftshift` znajduje sie w centrum, co jest czytelniejsze na wykresie.

4. Obliczenie logarytmicznego modulu:

```python
spectrum_log = np.log10(np.abs(freq_shift) + 1.0)
```

Wyjasnienie:

- `np.abs(freq_shift)` liczy modul liczby zespolonej;
- `+ 1.0` zapobiega logarytmowi z zera;
- `log10` kompresuje duzy zakres wartosci widma, dzieki czemu slabsze czestotliwosci sa widoczne.

5. Obliczenie fazy:

```python
phase = np.angle(freq_shift)
```

Faza jest w radianach, w zakresie `[-pi, pi]`. To uzupelnia modul widma i daje pelniejsza informacje o wyniku FFT.

6. Obliczenie IFFT:

```python
channel_rec = np.fft.ifft2(freq).real
```

Uzywany jest nieprzesuniety `freq`, bo `ifft2` oczekuje standardowego ukladu wyniku FFT. Wynik IFFT teoretycznie powinien byc rzeczywisty. Numerycznie moze miec minimalna czesc urojona, dlatego brane jest `.real`.

7. Obliczenie bledu:

```python
err = channel_rec - channel_org
```

8. MSE:

```python
mse = mean(err ** 2)
```

MSE karze wieksze bledy mocniej, bo blad jest podnoszony do kwadratu.

9. RMSE:

```python
rmse = sqrt(mse)
```

RMSE ma te sama jednostke co wartosci pikseli.

10. MAE:

```python
mae = mean(abs(err))
```

MAE to sredni blad bezwzgledny.

11. Maksymalny blad:

```python
max_abs = max(abs(err))
```

To najwazniejsza metryka dla testu round-trip: jezeli jest mniejsza od tolerancji, transformacja i odwrotna transformacja odtworzyly kanal z dokladnoscia numeryczna.

12. PSNR:

```python
psnr = inf if mse == 0 else 20 * log10(data_range) - 10 * log10(mse)
```

PSNR jest wysoki, gdy blad jest niski. Jezeli `mse == 0`, wynik jest idealny i PSNR jest nieskonczony.

13. Funkcja zwraca:

- log-widmo;
- faze;
- rekonstrukcje;
- blad;
- metryki kanalu.

### `_build_summary(channel_metrics)`

Funkcja laczy metryki ze wszystkich kanalow.

Obliczenia:

- `mse_mean` - srednia MSE po kanalach;
- `rmse_mean` - srednia RMSE po kanalach;
- `mae_mean` - srednia MAE po kanalach;
- `max_abs_error` - maksimum z `max_abs_error`;
- `psnr_mean_db` - srednia z wartosci skonczonych PSNR; jesli wszystkie sa nieskonczone, wynik to `inf`;
- `passed` - `True`, jesli `max_abs_error <= tolerance`.

Dlaczego maksimum bledu jest uzyte do `passed`: jezeli chociaz jeden kanal ma za duzy blad, caly obraz nie przeszedl testu poprawnosci.

### `analyze(image)`

Pelna analiza FFT dla obrazu.

Kroki:

1. Konwertuje BGR/BGRA na RGB/RGBA przez `to_rgb_or_gray`.
2. Ustala tryb przez `image_mode`.
3. Jesli obraz jest grayscale:
   - analizuje jeden kanal;
   - rekonstruuje go do oryginalnego typu danych;
   - buduje `FftAnalysisResult`.
4. Jesli obraz ma wiele kanalow:
   - iteruje po wszystkich kanalach, rowniez alpha;
   - dla kazdego kanalu liczy FFT, faze, IFFT i metryki;
   - sklada wyniki przez `np.stack`;
   - normalizuje log-widmo do wyswietlania;
   - buduje `FftAnalysisResult`.

To zapewnia poprawna obsluge obrazow:

- grayscale;
- RGB;
- RGBA;
- `uint8`;
- `uint16`.

### `FftVerificationService`

Serwis laczacy loader obrazu i analizator FFT.

#### `__init__(loader, analyzer)`

Przyjmuje dwa komponenty:

- `loader` - obiekt z metoda `load`;
- `analyzer` - obiekt z metoda `analyze`.

Dzieki temu w testach albo przyszlych zmianach mozna wymienic sposob wczytywania lub analizy.

#### `analyze_image(image_path)`

1. Wczytuje obraz przez `loader.load(image_path)`.
2. Przekazuje tablice pikseli do `analyzer.analyze(image)`.
3. Zwraca `FftAnalysisResult`.

## Modul `src/fft_presenters.py`

### `MatplotlibFftPresenter.show(result)`

Funkcja rysuje wynik FFT w Matplotlib.

Tworzy figure `13 x 8` i uklada wykresy w siatce `2 x 3`:

1. Oryginalny obraz.
2. Logarytmiczny modul widma.
3. Faza FFT.
4. Rekonstrukcja po IFFT.
5. Mapa bledu bezwzglednego.

Dla fazy:

- jesli obraz ma wiele kanalow, faza jest usredniana po kanalach;
- kolorystyka `twilight` nadaje sie do wartosci cyklicznych od `-pi` do `pi`;
- `vmin=-np.pi`, `vmax=np.pi` stabilizuja skale kolorow.

Dla mapy bledu:

- jesli obraz ma wiele kanalow, blad jest usredniany po kanalach;
- `hot` pokazuje obszary z wiekszym bledem jako jasniejsze/cieplejsze.

Tytul calej figury zawiera najwazniejsze metryki: `passed`, tolerancje, MSE, RMSE, max error i PSNR.

## Modul `src/fft_display.py`

### `analyze_fft(image_path, tolerance=1e-6)`

Tworzy `FftVerificationService` z:

- `Cv2ImageLoader`;
- `NumpyRoundTripFftAnalyzer`.

Nastepnie analizuje wskazany plik i zwraca `FftAnalysisResult`.

### `fft_display(image_path, tolerance=1e-6)`

1. Wywoluje `analyze_fft`.
2. Przekazuje wynik do `MatplotlibFftPresenter().show(result)`.
3. Zwraca wynik analizy.

Ta funkcja jest prostym interfejsem dla skryptowego uzycia bez GUI.

## Modul `src/fft_contracts.py`

Ten modul definiuje protokoly, czyli oczekiwane interfejsy obiektow.

### `ImageLoader.load(image_path)`

Kontrakt: obiekt loadera przyjmuje sciezke do pliku i zwraca obraz jako `np.ndarray`.

### `FftAnalyzer.analyze(image)`

Kontrakt: analizator przyjmuje tablice obrazu i zwraca `FftAnalysisResult`.

### `FftPresenter.show(result)`

Kontrakt: presenter przyjmuje wynik analizy i prezentuje go uzytkownikowi.

## Modul `app/gui.py`

GUI laczy wymagania projektowe w jeden przeplyw:

1. Uzytkownik wybiera albo przeciaga pliki PNG.
2. Aplikacja analizuje kazdy plik w tle.
3. Pokazuje obraz, widmo, faze, rekonstrukcje i blad.
4. Pokazuje tekstowy raport z naglowka i chunkow.
5. Uzytkownik moze osobno uruchomic anonimizacje wybranego pliku.

### `PngAnalysis`

Dataclass przechowujaca wynik analizy jednego pliku:

- `image_path` - sciezka do oryginalnego pliku;
- `anonymized_path` - sciezka do pliku po anonimizacji albo `None`;
- `chunks` - tuple odczytanych `PngChunk`;
- `fft` - wynik FFT;
- `console_output` - tekst raportu pokazywany w GUI.

### `_unique_anonymized_path(image_path)`

Funkcja wybiera nazwe pliku wyjsciowego dla anonimizacji.

Kolejne proby:

- `nazwa_anon.png`;
- `nazwa_anon_1.png`;
- `nazwa_anon_2.png`;
- itd.

Funkcja nie nadpisuje istniejacych plikow.

### `analyze_png_file(image_path, tolerance=1e-6)`

Glowna funkcja analizy jednego PNG w GUI.

Kroki:

1. Normalizuje sciezke:

```python
path = Path(image_path).expanduser().resolve()
```

2. Sprawdza rozszerzenie `.png`.
3. Tworzy `StringIO`, zeby przechwycic tekst wypisywany przez funkcje parsera.
4. Wypisuje sciezke pliku.
5. Wypisuje `IHDR` przez `display_IHDR_chunks_info`.
6. Tworzy `FftVerificationService`.
7. Analizuje obraz przez FFT/IFFT.
8. Wypisuje podsumowanie metryk.
9. Wypisuje metryki kazdego kanalu.
10. Laduje wszystkie chunki przez `load_all_chunks`.
11. Dla kazdego chunka wypisuje `describe_chunk`.
12. Informuje, ze anonimizacja nie zostala wykonana automatycznie.
13. Zwraca `PngAnalysis`.

Wazne: analiza i anonimizacja sa rozdzielone. To odpowiada wymaganiu "dodac opcje kasowania", a nie wykonywac kasowanie przy samym otwarciu pliku.

### `anonymize_analysis_result(result)`

Funkcja wykonuje anonimizacje dla juz przeanalizowanego pliku.

Kroki:

1. Wybiera unikalna sciezke wyjsciowa przez `_unique_anonymized_path`.
2. Wywoluje `anonymize_png_chunks`.
3. Buduje komunikat z raportem:
   - liczba zachowanych chunkow;
   - liczba usunietych chunkow;
   - zachowane ancillary chunks istotne dla renderowania;
   - usuniete typy chunkow;
   - liczba usunietych bajtow po `IEND`;
   - informacja, czy `IDAT` zostal zachowany.
4. Tworzy kopie `PngAnalysis` z uzupelnionym `anonymized_path` i rozszerzonym raportem.

### `_normalize_to_uint8(array)`

Funkcja przygotowuje dowolna tablice numeryczna do wyswietlania jako obraz 8-bitowy.

Kroki:

1. Rzutuje dane na `float64`.
2. Sprawdza wartosci skonczone przez `np.isfinite`.
3. Jesli nie ma zadnych skonczonych wartosci, zwraca czarna tablice `uint8`.
4. Wyznacza minimum i maksimum.
5. Jesli maksimum jest mniejsze lub rowne minimum, zwraca czarny obraz.
6. Normalizuje:

```python
normalized = (clean - min_value) / (max_value - min_value)
```

7. Skaluje do `0..255`, zaokragla i rzutuje na `uint8`.

To jest uzywane tylko do wizualizacji, nie do oceny poprawnosci FFT.

### `_hot_colormap(array)`

Funkcja tworzy prosta mape kolorow typu "hot" dla mapy bledu.

1. Najpierw normalizuje dane do `uint8`.
2. Skaluje do zakresu `0..1`.
3. Liczy kanaly:

```python
red = clip(values * 3.0, 0.0, 1.0)
green = clip(values * 3.0 - 1.0, 0.0, 1.0)
blue = clip(values * 3.0 - 2.0, 0.0, 1.0)
```

Dla malych bledow obraz jest ciemny. Dla duzych bledow przechodzi przez czerwien/zolc do jasnych kolorow.

### `numpy_to_qimage(array, force_gray=False, hot=False)`

Konwertuje tablice NumPy na `QImage` do wyswietlenia w PySide6.

Tryby:

- `hot=True` - najpierw naklada `_hot_colormap`;
- `force_gray=True` - wymusza obraz grayscale;
- tablica 2D - grayscale;
- tablica 3D z 3 kanalami - RGB;
- tablica 3D z 4 kanalami - RGBA.

Dla danych nie-`uint8` funkcja normalizuje je do zakresu wyswietlania. Dla RGBA zachowuje 4 kanaly i uzywa `QImage.Format_RGBA8888`.

### `AnalysisSignals`

Klasa sygnalow Qt dla pracy w tle:

- `result_ready` - wynik analizy jednego pliku;
- `error` - blad analizy jednego pliku;
- `finished` - zakonczenie calej paczki.

### `AnalysisWorker`

Worker uruchamiany w `QThreadPool`, zeby GUI nie zamarzalo podczas obliczen.

#### `__init__(paths)`

Zapamietuje liste sciezek i tworzy obiekt sygnalow.

#### `run()`

Iteruje po plikach z indeksami:

- analizuje plik przez `analyze_png_file`;
- emituje `result_ready(index, result)`;
- w przypadku bledu emituje `error(index, path, message)`;
- po wszystkim emituje `finished`.

Indeks jest wazny, bo dwa razy wybrany ten sam plik nadal powinien miec dwa osobne wiersze w GUI.

### `DropArea`

Widget do przeciagania plikow i otwierania dialogu.

#### `__init__()`

Buduje UI drop-area:

- wlacza przyjmowanie drag-and-drop;
- dodaje tytul;
- dodaje podtytul;
- dodaje przycisk "Otworz pliki PNG".

#### `dragEnterEvent(event)`

Akceptuje przeciaganie tylko wtedy, gdy w zdarzeniu sa lokalne pliki `.png`.

#### `dropEvent(event)`

Po upuszczeniu plikow zbiera sciezki PNG i emituje `files_selected`.

#### `_event_png_paths(event)`

Wyciaga lokalne sciezki z `event.mimeData().urls()` i filtruje tylko `.png`.

#### `_open_dialog()`

Otwiera systemowy dialog wyboru plikow PNG. Domyslnie startuje z katalogu `data`.

### `UploadPage`

Pierwsza strona GUI.

#### `__init__()`

Tworzy naglowek, status oraz `DropArea`. Sygnal `files_selected` z `DropArea` przekazuje dalej.

#### `set_status(text)`

Ustawia komunikat statusu, np. gdy uzytkownik nie wybierze PNG.

### `PlotPanel`

Panel do wyswietlania pojedynczego obrazu/wykresu.

#### `__init__(title)`

Tworzy ramke z tytulem i `QLabel` na pixmap.

#### `set_image(image)`

Konwertuje `QImage` na `QPixmap`, zapamietuje oryginalna pixmap i odswieza wyswietlanie.

#### `resizeEvent(event)`

Przy zmianie rozmiaru panelu ponownie skaluje obraz.

#### `_refresh_pixmap()`

Skaluje pixmap do aktualnego rozmiaru labela z zachowaniem proporcji.

### `AnalysisPage`

Strona wynikow analizy.

#### `__init__()`

Tworzy:

- liste plikow;
- przycisk anonimizacji;
- przycisk wyboru kolejnych plikow;
- panele: oryginal, log-widmo, faza, rekonstrukcja, mapa bledu;
- pole konsoli z raportem tekstowym.

#### `reset(paths)`

Czyści poprzednie wyniki, ustawia liczbe oczekiwanych plikow i wypelnia liste nazwami plikow.

#### `add_result(row, result)`

Dodaje wynik analizy do konkretnego wiersza:

- zapisuje `PngAnalysis`;
- aktualizuje tekst i tooltip;
- ustawia pierwszy wynik jako aktualny, jesli nic nie wybrano;
- oznacza wiersz jako zakonczony;
- pokazuje wynik.

#### `add_error(row, path, message)`

Oznacza dany wiersz jako blad i dopisuje komunikat do konsoli.

#### `finish()`

Ustawia tytul "Analiza zakonczona" i pokazuje przycisk wyboru kolejnych plikow.

#### `show_result(row)`

Pokazuje wybrany wynik:

- oryginalny obraz;
- log-widmo;
- faze;
- rekonstrukcje;
- mape bledu;
- tekstowy raport chunkow.

Dla RGBA log-widmo i faza sa usredniane do grayscale, zeby uniknac niejednoznacznej prezentacji 4-kanalowej jako kolor.

#### `anonymize_selected()`

Uruchamia anonimizacje aktualnie wybranego wyniku.

Kroki:

1. Sprawdza, czy jest wybrany poprawny wiersz.
2. Pobiera `PngAnalysis`.
3. Wywoluje `anonymize_analysis_result`.
4. Aktualizuje wynik i opis na liscie.
5. Pokazuje zaktualizowany raport.

#### `_mark_finished(row)`

Dodaje numer wiersza do `finished_rows` i aktualizuje pasek postepu.

#### `_update_progress()`

Ustawia tekst postepu w formacie `Analiza w toku: X/Y`.

#### `_row_for_path(path)`

Pomocnicza funkcja wyszukujaca wiersz po sciezce. Po poprawkach glowny przeplyw uzywa indeksow, ale funkcja zostala jako pomocniczy mechanizm.

### `MainWindow`

Glowne okno aplikacji.

#### `__init__()`

Tworzy:

- tytul i rozmiar okna;
- globalny `QThreadPool`;
- stos stron `QStackedWidget`;
- `UploadPage`;
- `AnalysisPage`;
- polaczenia sygnalow.

#### `show_upload()`

Przelacza GUI z powrotem na strone wyboru plikow.

#### `start_analysis(paths)`

Uruchamia analize:

1. Filtruje tylko sciezki `.png`.
2. Jesli nie ma PNG, pokazuje status.
3. Resetuje strone wynikow.
4. Tworzy `AnalysisWorker`.
5. Podlacza sygnaly workera do metod GUI.
6. Dodaje workera do listy, zeby nie zostal usuniety przez garbage collector.
7. Startuje worker w `QThreadPool`.

### `apply_style(app)`

Ustawia style CSS/Qt dla calej aplikacji: kolory, fonty, przyciski, panele i konsola.

### `main()`

Tworzy `QApplication`, naklada styl, tworzy `MainWindow`, pokazuje okno i uruchamia petle zdarzen.

## Modul `main.py`

Skrypt demonstracyjny. Po uruchomieniu bezposrednim:

1. Ustawia katalog `data`.
2. Wybiera `NewTux.png` jako zrodlo.
3. Wypisuje informacje `IHDR`.
4. Pokazuje FFT przez Matplotlib.
5. Laduje i opisuje wszystkie chunki.
6. Tworzy plik anonimowy.

GUI jest pelniejszym sposobem korzystania z projektu, bo rozdziela analize i anonimizacje.

## Testowanie FFT

PDF wymaga zaproponowania sposobu testowania poprawnosci transformacji Fouriera. W projekcie metoda jest konkretna:

1. Dla obrazu liczona jest FFT.
2. Na wyniku FFT liczona jest IFFT.
3. Wynik IFFT powinien odtworzyc oryginalny obraz.
4. Program liczy blad:
   - MSE;
   - RMSE;
   - MAE;
   - maksymalny blad bezwzgledny;
   - PSNR.
5. Jezeli `max_abs_error <= tolerance`, test przechodzi.

### Testy jednostkowe w `test/test_fft.py`

#### `test_grayscale_fft_round_trip_reconstructs_original_pixels`

Tworzy maly obraz grayscale `uint8`. Sprawdza, czy:

- tryb to `gray`;
- rekonstrukcja jest identyczna z wejsciem;
- test `passed` jest prawdziwy;
- blad maksymalny miesci sie w tolerancji;
- istnieje jeden kanal metryk;
- log-widmo i faza maja taki sam ksztalt jak obraz.

#### `test_bgr_input_fft_round_trip_reconstructs_rgb_channels`

Tworzy sztuczny obraz BGR. Sprawdza, czy:

- OpenCV-owy BGR jest poprawnie konwertowany do RGB;
- rekonstrukcja RGB jest poprawna;
- metryki sa dla 3 kanalow;
- widmo, faza i mapa bledu maja ksztalt obrazu RGB.

#### `test_bgra_input_preserves_alpha_channel`

Tworzy obraz BGRA i sprawdza, czy:

- wynik ma tryb `rgba`;
- kanal alpha zostaje zachowany;
- analiza obejmuje 4 kanaly;
- rekonstrukcja jest identyczna.

#### `test_uint16_input_reconstructs_without_clipping_to_uint8`

Tworzy obraz `uint16` z wartosciami wiekszymi niz 255. Sprawdza, czy:

- rekonstrukcja ma typ `uint16`;
- wartosci nie zostaly obciete do `uint8`;
- test round-trip przechodzi.

#### `test_fft_metrics_report_round_trip_error_within_tolerance`

Sprawdza, czy dla prostego obrazu metryki bledu sa ponizej tolerancji.

#### `test_psnr_summary_uses_finite_values_when_only_some_channels_are_perfect`

Sprawdza przypadek, gdy jeden kanal ma idealne odtworzenie (`PSNR = inf`), a drugi ma skonczony PSNR. Srednia PSNR powinna byc liczona ze skonczonych wartosci, a nie automatycznie ustawiana na `inf`.

## Namacalne testy obrazami

Oprocz testow jednostkowych warto podczas prezentacji pokazac obrazy, ktorych widmo da sie intuicyjnie wyjasnic.

### Test 1: calkowicie czarny obraz PNG

Przykladowy plik w projekcie: `data/Black.png`.

Oczekiwane zachowanie:

- wszystkie piksele maja wartosc 0;
- obraz jest staly;
- FFT obrazu stalego ma tylko skladowa DC;
- dla obrazu calkowicie czarnego skladowa DC tez wynosi 0;
- `np.abs(freq_shift)` jest wszedzie 0;
- `log10(abs + 1)` daje wszedzie `log10(1) = 0`;
- log-widmo powinno byc czarne/jednolite;
- faza dla zerowego widma nie niesie praktycznej informacji, NumPy zwroci 0;
- IFFT powinna odtworzyc dokladnie czarny obraz;
- mapa bledu powinna byc calkowicie czarna;
- `MSE`, `RMSE`, `MAE`, `max_abs_error` powinny byc 0 albo numerycznie bardzo bliskie 0;
- `passed=True`.

Co ten test sprawdza:

- czy kod radzi sobie z obrazem bez kontrastu;
- czy normalizacja widma nie dzieli przez zero;
- czy IFFT nie wprowadza sztucznych wartosci;
- czy GUI pokazuje sensowny wynik dla stalego obrazu.

### Test 2: obraz w bialo-czarne pasy

Przykladowy plik w projekcie: `data/bws.png`.

Oczekiwane zachowanie:

- obraz zawiera regularny, okresowy wzor;
- FFT obrazu okresowego powinna pokazac wyrazne punkty/piki widma;
- skladowa DC w centrum odpowiada sredniej jasnosci obrazu;
- dodatkowe piki odpowiadaja czestotliwosci pasow;
- im gestsze pasy, tym dalej od centrum pojawiaja sie piki;
- dla pionowych pasow zmiana zachodzi w osi X, wiec piki sa rozlozone poziomo w widmie;
- dla poziomych pasow zmiana zachodzi w osi Y, wiec piki sa rozlozone pionowo;
- faza opisuje przesuniecie wzoru pasow;
- IFFT powinna odtworzyc oryginalne pasy;
- mapa bledu powinna byc czarna albo prawie czarna;
- `passed=True`.

Co ten test sprawdza:

- czy widmo faktycznie reaguje na strukture obrazu;
- czy `fftshift` przenosi centrum widma na srodek wykresu;
- czy logarytmiczna skala pokazuje piki czytelnie;
- czy rekonstrukcja zachowuje ostre przejscia czarny-bialy;
- czy parser PNG i FFT dzialaja dla realnego pliku z danymi `IDAT`.

### Jak wykonac testy namacalne

W GUI:

1. Uruchomic aplikacje.
2. Wybrac `Black.png`.
3. Obejrzec oryginal, log-widmo, faze, rekonstrukcje i mape bledu.
4. Powtorzyc dla `bws.png`.
5. Dla obu plikow sprawdzic w konsoli, czy `passed=True` i `max_abs` jest w tolerancji.

Skryptowo:

```bash
python -m unittest discover -s P_1/test
```

Dodatkowo mozna uruchomic GUI i recznie przejsc przez pliki z katalogu `P_1/data`.

## Testowanie parsera PNG i anonimizacji

### `test_load_all_chunks_reports_trailing_data_after_iend`

Test tworzy minimalny plik PNG-podobny i dopisuje bajty po `IEND`. Sprawdza, czy `load_all_chunks` zapisuje te bajty w `trailing_data`. To odpowiada wymaganiu analizowania konstrukcji pliku, a nie tylko standardowych metadanych.

### `test_anonymize_removes_ancillary_chunks_and_trailing_data`

Test tworzy plik z:

- `IHDR`;
- `tEXt`;
- `IDAT`;
- `IEND`;
- danymi po `IEND`.

Po anonimizacji oczekiwane sa tylko `IHDR`, `IDAT`, `IEND`. `tEXt` i dane po `IEND` musza zniknac.

### `test_anonymize_keeps_rendering_relevant_ancillary_chunks`

Test tworzy plik z `gAMA` i `tEXt`.

Oczekiwane:

- `gAMA` zostaje, bo moze wplywac na wyglad;
- `tEXt` znika, bo jest metadana tekstowa;
- raport pokazuje oba fakty.

### `test_describe_plte_contains_all_palette_entries`

Test tworzy maly `PLTE` z dwoma kolorami i sprawdza, czy opis chunka zawiera oba kolory oraz hex danych. To sprawdza, czy critical chunk `PLTE` jest prezentowany jako zawartosc, a nie tylko jako rozmiar.

## Anonimizacja a brak ingerencji w obraz

W PNG obraz pikselowy jest przechowywany glownie w `IDAT`, ale interpretacja moze zalezec tez od innych chunkow:

- `PLTE` dla obrazow paletowych;
- `tRNS` dla przezroczystosci;
- `gAMA`, `iCCP`, `sRGB`, `cHRM` dla kolorow;
- `sBIT` dla informacji o istotnych bitach;
- `bKGD` dla tla.

Dlatego kod nie usuwa mechanicznie wszystkich ancillary chunks. Usuwa informacje uznane za zbedne/metadane prywatne, ale zachowuje ancillary chunks, ktore moga zmieniac sposob wyswietlania.

Raport anonimizacji pokazuje:

- co zostalo zachowane;
- co zostalo usuniete;
- jakie rendering-relevant ancillary chunks zostaly zachowane;
- czy `IDAT` zostal przepisany bez zmian;
- ile danych po `IEND` usunieto.

To jest argument do uzasadnienia decyzji anonimizacyjnych podczas zaliczenia.

## Uruchamianie testow

Domyslne testy:

```bash
python -m unittest discover -s P_1/test
```

Stary wzorzec kompatybilny:

```bash
python -m unittest discover -s P_1/test -p '*_test.py'
```

Bezposrednio:

```bash
python P_1/test/fft_test.py
```

## Najwazniejszy przeplyw danych w aplikacji

1. Uzytkownik wybiera PNG.
2. `AnalysisWorker.run` wywoluje `analyze_png_file`.
3. `display_IHDR_chunks_info` recznie czyta sygnature i `IHDR`.
4. `load_all_chunks` recznie przechodzi po wszystkich chunkach.
5. `describe_chunk` tworzy raport tekstowy chunkow.
6. `Cv2ImageLoader` wczytuje piksele do tablicy NumPy.
7. `NumpyRoundTripFftAnalyzer` liczy FFT, faze, IFFT i metryki.
8. GUI pokazuje oryginal, widmo, faze, rekonstrukcje i blad.
9. Po kliknieciu anonimizacji `anonymize_png_chunks` tworzy nowy plik PNG.
10. Raport pokazuje, ktore dane zostaly usuniete i czy obrazowe `IDAT` zostalo zachowane.

## Ograniczenia obecnej implementacji

- Parser EXIF pokazuje tylko podsumowanie, nie wszystkie tagi.
- Kod nie skleja ani nie przepisuje strategii podzialu wielu `IDAT` w jeden chunk; zachowuje dane obrazu bez zmian.
- GUI prezentuje dane wizualnie po normalizacji do `uint8`, ale metryki poprawnosci sa liczone na oryginalnych danych.
- `main.py` jest prostym skryptem demonstracyjnym; glownym interfejsem projektu jest GUI.
