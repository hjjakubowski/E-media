# Notatka obronna do projektu P_1

Ta notatka jest przygotowana pod obrone projektu z pliku `E_media_projekt_1-1.pdf`. Nie jest to skrocony README. To material do nauczenia sie projektu o formatach PNG, FFT, anonimizacji, testowanie i konkretne fragmenty kodu.

Dokument opisuje tylko zagadnienia z PDF, ktore zostaly zaimplementowane w `P_1`. Nie opisuje jako wykonanych rzeczy, ktorych kod nie robi, np. pelnego parsera EXIF, pelnej analizy metod kompresji PNG albo sklejania wielu `IDAT` w jeden chunk.

## Mapa wymagan PDF na implementacje

| Zagadnienie z PDF | Czy jest w P_1 | Glowne pliki |
| --- | --- | --- |
| Reczne dekodowanie informacji z PNG na podstawie bajtow | Tak | `src/chunks.py` |
| Pokazanie atrybutow pliku: rozmiar, glebia koloru, typ koloru | Tak | `src/chunks.py`, `app/gui.py` |
| Prezentacja pliku jako obrazu | Tak | `app/gui.py`, `src/fft_presenters.py` |
| Wyswietlenie widma Fouriera | Tak | `src/fft_services.py`, `src/fft_presenters.py`, `app/gui.py` |
| Pelniejsza informacja FFT, nie tylko obraz oryginalny | Tak: modul logarytmiczny, faza, rekonstrukcja, blad | `src/fft_services.py`, `src/fft_models.py` |
| Testowanie poprawnosci FFT | Tak: FFT -> IFFT -> metryki bledu + testy jednostkowe | `src/fft_services.py`, `test/test_fft.py` |
| Anonimizacja bez ingerencji w obraz | Tak: przepisanie danych obrazu, usuwanie metadanych, zachowanie chunkow renderujacych | `src/chunks.py`, `app/gui.py`, `test/test_chunks.py` |
| Critical chunks PNG | Tak: odczyt, opis, zachowanie przy anonimizacji | `src/chunks.py` |
| Wybrane ancillary chunks | Czesiowo: `tIME`, `tEXt`, podstawowe `eXIf`, oraz polityka renderujacych ancillary | `src/chunks.py` |
| Dane ukryte w konstrukcji pliku, np. bajty po `IEND` | Tak: wykrywanie i usuwanie trailing data | `src/chunks.py`, `test/test_chunks.py` |

## 1. Reczne dekodowanie PNG na podstawie bajtow

### Co mowi wymaganie

PDF wymaga, zeby metadane formatu byly odczytane recznie, czyli na podstawie analizy kolejnych bajtow pliku. W projekcie oznacza to, ze informacje o strukturze PNG nie sa brane z gotowego parsera PNG. Kod sam otwiera plik jako bajty, sam czyta sygnature, sam rozpoznaje chunki, sam liczy CRC i sam interpretuje `IHDR`.

### Jak jest to zaimplementowane

Glowny modul to `src/chunks.py`.

Najwazniejsze elementy:

- `PNG_SIGNATURE` - wzorzec pierwszych 8 bajtow PNG;
- `read_chunk(f)` - reczne czytanie jednego chunka;
- `parse_ihdr(chunk_data)` - reczna interpretacja naglowka `IHDR`;
- `load_all_chunks(image)` - przejscie po calym pliku chunk po chunku;
- `PngChunk` - struktura przechowujaca wynik recznego odczytu.

### PNG jako ciag bajtow

Kazdy poprawny plik PNG zaczyna sie sygnatura:

```text
89 50 4E 47 0D 0A 1A 0A
```

W kodzie:

```python
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
```

Znaczenie:

- `\x89` - bajt spoza ASCII, pomaga wykryc problemy z transmisja tekstowa;
- `PNG` - litery identyfikujace format;
- `\r\n` - koniec linii Windows;
- `\x1a` - historyczny znak konca pliku w DOS;
- `\n` - koniec linii Unix.

Funkcja `load_all_chunks` robi:

```python
signature = f.read(len(PNG_SIGNATURE))
if signature != PNG_SIGNATURE:
    raise ValueError("Invalid PNG signature")
```

To oznacza: przeczytaj dokladnie 8 bajtow i porownaj z oczekiwana sygnatura. Jesli sie nie zgadza, dalsze parsowanie nie ma sensu.

### Struktura chunka PNG

Po sygnaturze PNG sklada sie z chunkow. Kazdy chunk ma:

```text
4 bajty length
4 bajty chunk type
length bajtow data
4 bajty CRC
```

Kod nie zgaduje tych wartosci. Czyta je po kolei w `read_chunk`.

### `read_chunk(f)` krok po kroku

#### Odczyt dlugosci

```python
length_raw = f.read(4)
```

`f.read(4)` to standardowa funkcja Pythona na pliku binarnym. Czyta maksymalnie 4 bajty z aktualnej pozycji kursora pliku. Jesli plik jest uciety, moze zwrocic mniej niz 4 bajty.

Kod sprawdza:

```python
if len(length_raw) != 4:
    raise EOFError("Brak danych na dlugosc chunka")
```

To zabezpiecza przed niepelna struktura PNG.

#### Zamiana 4 bajtow na liczbe

```python
(chunk_length,) = st.unpack(">I", length_raw)
```

Uzyta biblioteka: `struct`.

Co robi `struct.unpack`:

- bierze surowe bajty;
- interpretuje je wedlug formatu;
- zwraca krotke z odczytanymi wartosciami.

Format `">I"`:

- `>` - big-endian, czyli najstarszy bajt jest pierwszy;
- `I` - 4-bajtowa liczba calkowita bez znaku.

PNG uzywa big-endian, dlatego nie mozna uzyc domyslnego endianu maszyny. Gdyby uzyc zlego endianu, dlugosc chunka bylaby bledna.

Przyklad:

```text
00 00 00 0D
```

To wartosc 13, czyli typowa dlugosc danych `IHDR`.

#### Limit bezpieczenstwa

```python
if chunk_length > MAX_CHUNK_LENGTH:
    raise ValueError(...)
```

To chroni aplikacje przed plikiem deklarujacym absurdalnie duzy chunk. Bez tego program moglby probowac wczytac setki megabajtow albo wiecej.

#### Odczyt typu chunka

```python
chunk_type = f.read(4)
```

Typ ma zawsze 4 bajty, np.:

- `IHDR`;
- `PLTE`;
- `IDAT`;
- `IEND`;
- `tEXt`.

Kod sprawdza, czy sa to litery ASCII:

```python
if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in chunk_type):
    raise ValueError(...)
```

Zakresy:

- `65..90` - `A..Z`;
- `97..122` - `a..z`.

#### Odczyt danych

```python
chunk_data = f.read(chunk_length)
```

Program czyta dokladnie tyle bajtow, ile podalo pole `length`.

Jesli dostanie mniej:

```python
if len(chunk_data) != chunk_length:
    raise EOFError("Brak danych chunka")
```

#### Odczyt CRC

```python
crc_raw = f.read(4)
(chunk_crc,) = st.unpack(">I", crc_raw)
```

CRC jest suma kontrolna zapisana w pliku. Tak jak length, jest 4-bajtowa big-endian unsigned int.

#### Obliczenie CRC

```python
calc_crc = zlib.crc32(chunk_type)
calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF
```

Uzyta biblioteka: `zlib`.

Co robi `zlib.crc32`:

- liczy 32-bitowa sume kontrolna CRC-32;
- CRC wykrywa przypadkowe uszkodzenia danych;
- w PNG CRC liczy sie po `chunk_type + chunk_data`, nie po polu `length`.

Dlaczego dwa wywolania:

1. Najpierw CRC po nazwie chunka.
2. Potem kontynuacja CRC po danych chunka.

To jest rownowazne policzeniu CRC po sklejonym `chunk_type + chunk_data`, ale bez tworzenia dodatkowej kopii danych.

Maska:

```python
& 0xFFFFFFFF
```

zapewnia dodatnia 32-bitowa reprezentacje wyniku.

#### Porownanie CRC

```python
if chunk_crc != calc_crc:
    raise ValueError(...)
```

Jesli CRC sie rozni, dane chunka sa uszkodzone albo plik zostal zmodyfikowany.

### `load_all_chunks(image)` jako reczne przejscie po pliku

Funkcja:

1. otwiera PNG;
2. sprawdza sygnature;
3. w petli czyta kolejne chunki przez `read_chunk`;
4. tworzy `PngChunk`;
5. konczy dopiero przy `IEND`;
6. czyta ewentualne bajty po `IEND`.

Najwazniejsze techniczne szczegoly:

```python
offset = f.tell()
```

`f.tell()` zwraca aktualna pozycje w pliku. Dla chunka jest to offset pola `length`. Ta informacja przydaje sie przy rozmowie o ukrywaniu danych w konstrukcji pliku.

```python
data_offset = offset + 8
```

Dane zaczynaja sie po 4 bajtach `length` i 4 bajtach `type`, wiec `offset + 8`.

```python
is_critical = not (chunk_type[0] & 32)
```

To rozpoznaje critical vs ancillary na podstawie pierwszej litery nazwy chunka. W ASCII mala litera ma ustawiony bit 32, wielka go nie ma.

Przy `IEND`:

```python
chunks[-1].trailing_data = f.read()
```

To czyta wszystko, co jest po formalnym koncu PNG. Takie dane nie sa potrzebne do obrazu, ale moga sluzyc do ukrywania informacji. Kod je wykrywa i potem usuwa przy anonimizacji.

## 2. Pokazanie atrybutow pliku PNG

### Co mowi wymaganie

PDF wymaga pokazania atrybutow pliku, np. rozmiaru i glebi koloru. W PNG te informacje sa w critical chunku `IHDR`.

### Jak jest to zaimplementowane

Odpowiedzialne funkcje:

- `parse_ihdr(chunk_data)`;
- `display_IHDR_chunks_info(image)`;
- `describe_chunk(chunk)` dla `IHDR`;
- `analyze_png_file` w GUI, ktore wywoluje te funkcje i zbiera tekstowy raport.

### `IHDR` jako naglowek PNG

`IHDR` ma zawsze 13 bajtow:

```text
width             4 bajty
height            4 bajty
bit_depth         1 bajt
color_type        1 bajt
compression       1 bajt
filter_method     1 bajt
interlace_method  1 bajt
```

Kod sprawdza:

```python
if len(chunk_data) != 13:
    raise ValueError(...)
```

To jest wazne, bo `IHDR` o innej dlugosci oznacza niepoprawny PNG.

### Rozpakowanie `IHDR`

```python
st.unpack(">IIBBBBB", chunk_data)
```

Znaczenie:

- `>` - big-endian;
- `I` - width, 4 bajty;
- `I` - height, 4 bajty;
- `B` - bit depth, 1 bajt;
- `B` - color type, 1 bajt;
- `B` - compression, 1 bajt;
- `B` - filter method, 1 bajt;
- `B` - interlace method, 1 bajt.

### Interpretacja pol

#### `width` i `height`

Rozmiar obrazu w pikselach.

#### `bit_depth`

Liczba bitow na probke, np. 8 albo 16. Dla RGB przy `bit_depth=8` kazdy kanal ma 8 bitow. Piksel RGB ma wtedy 24 bity, ale glebia w `IHDR` opisuje pojedyncza probke/kanal.

#### `color_type`

Kod typu koloru:

- `0` - grayscale;
- `2` - truecolor RGB;
- `3` - indexed-color, czyli obraz paletowy;
- `4` - grayscale z alpha;
- `6` - RGB z alpha.

Kod uzywa mapy `COLOR_TYPES`, zeby pokazac opis tekstowy.

#### `compression`

Dla standardowego PNG wartosc powinna byc `0`, co oznacza metoda deflate/inflate.

#### `filter_method`

Dla standardowego PNG wartosc `0`. PNG filtruje linie obrazu przed kompresja, zeby zwiekszyc skutecznosc kompresji.

#### `interlace_method`

`0` oznacza brak interlace, `1` oznacza Adam7 interlace. Kod pokazuje wartosc, ale nie implementuje recznego dekodowania pikseli z IDAT.

## 3. Critical chunks PNG

### Co mowi wymaganie

Dla PNG PDF wymaga wczytania i wyswietlenia zawartosci critical chunks. Critical chunks sa konieczne do poprawnego odtworzenia obrazu.

### Critical chunks obslugiwane przez kod

Kod rozpoznaje i opisuje:

- `IHDR` - naglowek;
- `PLTE` - paleta kolorow;
- `IDAT` - skompresowane dane obrazu;
- `IEND` - koniec pliku PNG.

### `IHDR`

Opisany szczegolowo w poprzedniej sekcji. Kod pokazuje wszystkie pola `IHDR`.

### `PLTE`

`PLTE` przechowuje palete. Kazdy kolor to 3 bajty:

```text
R G B
```

Jesli `PLTE` ma 768 bajtow, to:

```text
768 / 3 = 256 kolorow
```

Kod w `decode_plte_summary` sprawdza, czy dlugosc jest podzielna przez 3. Potem petla co 3 bajty buduje liste kolorow.

Dlaczego to jest wazne:

- dla obrazow typu `indexed-color` piksele w `IDAT` sa indeksami do palety;
- bez `PLTE` nie da sie poprawnie zinterpretowac kolorow obrazu paletowego;
- dlatego `PLTE` jest critical.

### `IDAT`

`IDAT` zawiera skompresowane dane obrazu. Kod nie dekompresuje recznie `IDAT`, bo PDF pozwala wczytywac dane pikseli gotowa biblioteka do FFT. Natomiast parser PNG:

- czyta wszystkie bajty `IDAT`;
- sprawdza CRC;
- pokazuje SHA-256;
- pokazuje hex danych;
- przepisuje `IDAT` bez zmian przy anonimizacji.

Dlaczego `IDAT` moze byc wiele:

PNG pozwala podzielic skompresowany strumien obrazu na wiele chunkow `IDAT`. Kolejnosc tych chunkow ma znaczenie. Kod zachowuje je w oryginalnej kolejnosc, dzieki czemu nie zmienia danych obrazu.

### `IEND`

`IEND` formalnie konczy PNG. Ma dlugosc 0.

Kod pokazuje:

- czy `IEND` jest pusty;
- ile bajtow znajduje sie po `IEND`;
- hex tych bajtow.

Dane po `IEND` nie sa potrzebne do wyswietlenia obrazu, ale moga byc nosnikiem ukrytej informacji. Dlatego kod je raportuje i usuwa przy anonimizacji.

## 4. Wybrane ancillary chunks

### Co mowi wymaganie

PDF dla wyzszych poziomow wspomina o dodatkowych segmentach, czyli ancillary chunks. Kod implementuje wybrane dekodery i polityke anonimizacji.

### `tIME`

`decode_time` rozumie 7-bajtowy format czasu:

```text
year month day hour minute second
```

Uzywa `struct.unpack(">HBBBBB", data)`.

Technicznie:

- `H` - 2 bajty na rok;
- `B` - 1 bajt na kazda pozostala wartosc.

`tIME` jest metadana, wiec przy anonimizacji moze zostac usuniety.

### `tEXt`

`decode_text` dzieli dane po pierwszym bajcie zerowym:

```python
parts = data.split(b"\x00", 1)
```

To daje:

- keyword;
- tekst.

`tEXt` moze zawierac autora, opis, komentarz, narzedzie zapisu. Jest to klasyczna metadana, wiec kod usuwa ja przy anonimizacji.

### `eXIf`

`decode_exif_summary` robi podstawowe sprawdzenie EXIF:

- endian `II` albo `MM`;
- magic number 42;
- offset do IFD;
- liczba tagow.

Uzyte funkcje z biblioteki:

- `struct.unpack("<H", ...)` albo `struct.unpack(">H", ...)` - odczyt 2-bajtowej wartosci zalezne od endianu;
- `struct.unpack("<I", ...)` albo `struct.unpack(">I", ...)` - odczyt 4-bajtowego offsetu.

Kod nie dekoduje wszystkich tagow EXIF. Trzeba to powiedziec na obronie wprost: projekt pokazuje podstawowe rozpoznanie bardziej zlozonego chunka, ale nie jest pelnym parserem EXIF.

### Ancillary chunks zachowywane dla renderowania

Niektore ancillary chunks nie sa "prywatna metadana", tylko wplywaja na wyglad:

- `gAMA` - gamma;
- `sRGB` - przestrzen barw;
- `iCCP` - profil ICC;
- `cHRM` - chromatycznosc;
- `tRNS` - przezroczystosc;
- `sBIT` - istotne bity;
- `bKGD` - sugerowane tlo.

Kod zachowuje je przy anonimizacji, bo wymaganie ogolne mowi o braku ingerencji w obraz.

## 5. Anonimizacja PNG

### Co mowi wymaganie

PDF wymaga opcji kasowania zbednych informacji o pliku, tak aby plik zostal zanonimizowany bez ingerencji w obraz.

### Jak jest to zaimplementowane

Glowne funkcje:

- `should_keep_chunk_for_anonymization`;
- `anonymize_png_chunks`;
- `write_chunk`;
- `anonymize_analysis_result`;
- przycisk `Anonimizuj wybrany plik` w GUI.

### Decyzja projektowa

Kod nie usuwa wszystkiego jak leci. Dzieli chunki na:

1. critical - zachowac;
2. ancillary wplywajace na renderowanie - zachowac;
3. ancillary bedace metadanymi lub dodatkowymi informacjami - usunac;
4. bajty po `IEND` - usunac.

To jest decyzja, ktora trzeba umiec uzasadnic:

- samo `IDAT` przechowuje dane obrazu, wiec musi zostac;
- `PLTE` moze byc niezbedne dla kolorow, wiec musi zostac;
- `tEXt` moze zdradzac autora albo program, wiec usuwamy;
- `tIME` moze zdradzac czas modyfikacji, wiec usuwamy;
- `gAMA` moze zmienic wyswietlanie, wiec zachowujemy.

### `write_chunk` i poprawny zapis CRC

Przy anonimizacji nie wystarczy skopiowac fragmentow tekstowo. Trzeba zapisac poprawny chunk:

1. `length`;
2. `type`;
3. `data`;
4. nowe CRC.

CRC jest liczone tak samo jak przy odczycie:

```python
calc_crc = zlib.crc32(chunk_type)
calc_crc = zlib.crc32(chunk_data, calc_crc) & 0xFFFFFFFF
```

Nawet jesli dane chunka sie nie zmieniaja, zapis przez `write_chunk` gwarantuje poprawna strukture pliku.

### Dlaczego obraz sie nie zmienia

Najwazniejsze:

- `IDAT` jest przepisywany bez zmian;
- kolejnosc zachowanych chunkow jest zachowana;
- `PLTE` jest zachowany;
- ancillary chunks istotne dla renderowania sa zachowane;
- usuwane sa informacje zbedne dla pikseli albo dane po `IEND`.

W testach sprawdzano tez praktycznie, ze po anonimizacji realnego PNG macierz pikseli wczytana przez OpenCV pozostaje taka sama.

## 6. Wczytywanie pikseli do FFT

### Co pozwala PDF

PDF pozwala korzystac z gotowych bibliotek do wczytywania danych pikseli obrazu. Reczne czytanie dotyczy metadanych/naglowka PNG.

### Uzyta biblioteka: OpenCV

Kod:

```python
cv.imread(image_path, cv.IMREAD_UNCHANGED)
```

Co robi `cv.imread`:

- otwiera plik obrazu;
- dekoduje format PNG;
- dekompresuje dane `IDAT`;
- stosuje filtry PNG;
- zwraca macierz pikseli jako `np.ndarray`.

Flaga `cv.IMREAD_UNCHANGED` oznacza:

- nie wymuszaj konwersji do 3 kanalow;
- zachowaj alpha, jesli istnieje;
- zachowaj glebie bitowa, np. `uint16`;
- zachowaj grayscale jako obraz 2D.

OpenCV zwraca kolory jako BGR/BGRA, nie RGB/RGBA. Dlatego kod robi konwersje.

### `cv.cvtColor`

Kod:

```python
cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
cv.cvtColor(image, cv.COLOR_BGR2RGB)
```

Co robi:

- przestawia kolejnosc kanalow;
- nie zmienia wartosci pikseli poza kolejnoscia;
- `BGR` staje sie `RGB`;
- `BGRA` staje sie `RGBA`.

Dlaczego to wazne:

- Matplotlib i GUI standardowo oczekuja RGB;
- bez konwersji czerwony i niebieski bylyby zamienione.

## 7. Transformacja Fouriera

### Po co FFT w projekcie

FFT pozwala zobaczyc, jakie czestotliwosci przestrzenne wystepuja w obrazie. W obrazie:

- wolne zmiany jasnosci to niskie czestotliwosci;
- ostre krawedzie i paski to wysokie czestotliwosci;
- regularny wzor daje wyrazne piki w widmie.

### Uzyta funkcja: `np.fft.fft2`

Kod:

```python
freq = np.fft.fft2(channel_org)
```

Co robi `fft2`:

- liczy dwuwymiarowa dyskretna transformate Fouriera;
- wejscie to macierz pikseli `height x width`;
- wyjscie to macierz liczb zespolonych tego samego rozmiaru;
- kazdy element opisuje jedna skladowa czestotliwosciowa.

Matematycznie dla obrazu `f(x, y)`:

```text
F(u, v) = suma_x suma_y f(x, y) * exp(-2*pi*i*(u*x/M + v*y/N))
```

Nie trzeba recznie implementowac tego wzoru, bo NumPy robi to wydajnym algorytmem FFT.

### Liczby zespolone w FFT

Wynik FFT jest zespolony:

```text
a + bi
```

Z tego bierzemy:

- modul: `sqrt(a^2 + b^2)`;
- faze: `atan2(b, a)`.

Kod:

```python
np.abs(freq_shift)
np.angle(freq_shift)
```

### `np.fft.fftshift`

Kod:

```python
freq_shift = np.fft.fftshift(freq)
```

Co robi:

- zamienia cwiartki widma miejscami;
- przenosi skladowa zerowa, czyli DC, do centrum obrazu;
- dzieki temu wykres widma jest intuicyjny.

Bez `fftshift` najnizsza czestotliwosc bylaby w rogu, co jest mniej czytelne.

### Logarytmiczny modul widma

Kod:

```python
spectrum_log = np.log10(np.abs(freq_shift) + 1.0)
```

Szczegoly:

- `np.abs` liczy modul liczby zespolonej;
- `+ 1.0` chroni przed `log10(0)`;
- `np.log10` kompresuje zakres.

Dlaczego logarytm:

- skladowa DC moze byc bardzo duza;
- mniejsze czestotliwosci bylyby niewidoczne w skali liniowej;
- logarytm pozwala zobaczyc slabsze elementy widma.

### Faza

Kod:

```python
phase = np.angle(freq_shift)
```

Co robi `np.angle`:

- dla kazdej liczby zespolonej liczy kat;
- wynik jest w radianach;
- zakres to zwykle `[-pi, pi]`.

Faza jest potrzebna, bo sam modul nie jest pelna informacja o FFT. Dwa obrazy moga miec podobny modul widma, ale rozna faze i wygladac inaczej.

## 8. Transformacja odwrotna i metryki

### `np.fft.ifft2`

Kod:

```python
channel_rec = np.fft.ifft2(freq).real
```

Co robi:

- liczy odwrotna transformate Fouriera;
- z widma wraca do dziedziny obrazu;
- wynik powinien byc taki sam jak oryginalny kanal.

Dlaczego `.real`:

- matematycznie wynik dla obrazu rzeczywistego powinien byc rzeczywisty;
- przez bledy numeryczne moze pojawic sie bardzo mala czesc urojona;
- `.real` bierze czesc rzeczywista.

### Blad rekonstrukcji

Kod:

```python
err = channel_rec - channel_org
```

Jesli FFT i IFFT zadzialaly poprawnie, `err` powinien byc zerowy albo bardzo bliski zera.

### MSE

```python
mse = mean(err ** 2)
```

Sredni blad kwadratowy. Kwadrat powoduje, ze duze bledy sa karane mocniej.

### RMSE

```python
rmse = sqrt(mse)
```

Pierwiastek z MSE. Ma ta sama skale co piksele.

### MAE

```python
mae = mean(abs(err))
```

Sredni blad bezwzgledny. Latwiejszy do intuicyjnej interpretacji niz MSE.

### Maksymalny blad bezwzgledny

```python
max_abs = max(abs(err))
```

Najwazniejszy dla decyzji `passed`. Jesli nawet jeden piksel/kanal ma za duzy blad, test powinien to wykryc.

### PSNR

Kod:

```python
20 * log10(data_range) - 10 * log10(mse)
```

PSNR to Peak Signal-to-Noise Ratio. Im wyzszy, tym mniejszy blad.

Dla `mse == 0`:

```python
psnr = inf
```

bo rekonstrukcja jest idealna.

### `data_range`

Dla `uint8`:

```text
data_range = 255
```

Dla `uint16`:

```text
data_range = 65535
```

Kod uzywa:

```python
np.iinfo(channel.dtype).max
```

`np.iinfo` zwraca informacje o typie calkowitym: minimum, maksimum, liczbe bitow.

## 9. Testowanie FFT

### Test automatyczny: round-trip

Metoda:

```text
obraz -> FFT -> IFFT -> porownanie z obrazem
```

To testuje, czy transformata i odwrotna transformata sa ze soba spojne.

### Test czarnego obrazu

Plik: `P_1/data/Black.png`.

Co powinno sie stac:

- wszystkie piksele maja wartosc 0;
- FFT jest zerowe;
- log-widmo jest zerowe, bo `log10(0 + 1) = 0`;
- faza nie niesie informacji, NumPy zwraca 0 dla zer;
- rekonstrukcja jest czarna;
- mapa bledu jest czarna;
- `MSE`, `RMSE`, `MAE`, `max_abs_error` sa 0 albo bliskie 0;
- `passed=True`.

O co moze zapytac prowadzacy:

- Dlaczego widmo czarnego obrazu jest zerowe? Bo obraz nie ma zadnej skladowej jasnosci ani zmian przestrzennych.
- Dlaczego dodajemy `+1` przed logarytmem? Zeby uniknac logarytmu z zera.
- Co sprawdza ten test? Stabilnosc dla obrazu bez kontrastu i brak dzielenia przez zero w normalizacji.

### Test obrazu w bialo-czarne pasy

Plik: `P_1/data/bws.png`.

Co powinno sie stac:

- obraz ma okresowy wzor;
- w widmie pojawiaja sie wyrazne piki;
- srodek widma to skladowa DC, czyli srednia jasnosc;
- piki poza srodkiem odpowiadaja czestotliwosci pasow;
- im gesciej ulozone pasy, tym dalej od srodka beda piki;
- IFFT powinna odtworzyc ostre pasy;
- mapa bledu powinna byc prawie czarna.

O co moze zapytac prowadzacy:

- Dlaczego pasy daja piki? Bo regularny wzor ma konkretna dominujaca czestotliwosc.
- Dlaczego potrzebny jest `fftshift`? Zeby skladowa DC byla w centrum wykresu.
- Co oznacza odleglosc piku od centrum? Czestotliwosc wzoru.

### Testy jednostkowe

`test/test_fft.py` sprawdza:

- grayscale round-trip;
- BGR -> RGB;
- BGRA -> RGBA z zachowaniem alpha;
- `uint16` bez obciecia do `uint8`;
- metryki ponizej tolerancji;
- poprawne srednie PSNR.

`test/test_chunks.py` sprawdza:

- wykrywanie danych po `IEND`;
- usuwanie `tEXt` i trailing data;
- zachowanie `gAMA`;
- pelny opis palety `PLTE`.

## 10. Prezentacja pliku i GUI

### PySide6

GUI korzysta z PySide6. Najwazniejsze klasy:

- `QApplication` - glowna aplikacja Qt i petla zdarzen;
- `QMainWindow` - glowne okno;
- `QStackedWidget` - przelaczanie miedzy strona wyboru i strona wynikow;
- `QFileDialog` - systemowe okno wyboru plikow;
- `QLabel` - wyswietlanie tekstu i obrazow;
- `QListWidget` - lista analizowanych plikow;
- `QPlainTextEdit` - konsola z raportem;
- `QThreadPool` i `QRunnable` - praca w tle;
- `Signal` i `Slot` - komunikacja miedzy workerem a GUI.

### Dlaczego worker w tle

FFT i parsowanie plikow moga trwac. Gdyby wykonywac je w glownym watku GUI, okno mogloby przestac reagowac. `AnalysisWorker` wykonuje prace w tle i emituje sygnaly z wynikiem.

### `QImage` i `QPixmap`

`numpy_to_qimage` zamienia tablice NumPy na `QImage`.

`QPixmap.fromImage(image)` tworzy obiekt gotowy do wyswietlenia w `QLabel`.

Dla danych nie-`uint8` kod normalizuje wartosci do `0..255`, bo ekran wyswietla obraz jako 8-bitowe skladowe koloru. To jest tylko wizualizacja. Metryki FFT sa liczone na oryginalnych wartosciach.

## 11. Matplotlib jako alternatywna prezentacja

`src/fft_presenters.py` korzysta z Matplotlib.

Najwazniejsze funkcje:

- `plt.figure(figsize=(13, 8))` - tworzy figure;
- `plt.subplot(2, 3, n)` - wybiera miejsce w siatce wykresow;
- `plt.imshow(...)` - wyswietla tablice jako obraz;
- `plt.colorbar(...)` - dodaje skale kolorow;
- `plt.suptitle(...)` - tytul calej figury z metrykami;
- `plt.tight_layout()` - poprawia rozmieszczenie;
- `plt.show()` - pokazuje okno.

`imshow`:

- dla grayscale z `cmap="gray"` pokazuje pojedynczy kanal;
- dla RGB/RGBA pokazuje kolory;
- dla fazy uzywa `cmap="twilight"`, bo faza jest cykliczna;
- dla bledu uzywa `cmap="hot"`, bo jasne obszary latwo wskazuja blad.

## 12. Typowe pytania prowadzacego i odpowiedzi

### Dlaczego metadane PNG sa czytane recznie, skoro OpenCV tez wczytuje PNG?

OpenCV jest uzyte tylko do pikseli potrzebnych do FFT. Metadane i chunki sa czytane recznie w `src/chunks.py`: sygnatura, length, type, data, CRC, IHDR, PLTE, IDAT, IEND i wybrane ancillary.

### Po co sprawdzac CRC?

CRC potwierdza, ze typ i dane chunka nie sa uszkodzone. PNG definiuje CRC dla kazdego chunka. Kod liczy CRC samodzielnie przez `zlib.crc32` i porownuje z wartoscia z pliku.

### Dlaczego `IDAT` nie jest recznie dekompresowany?

PDF pozwala uzyc gotowych bibliotek do wczytania danych pikseli. Reczne wymaganie dotyczy metadanych/naglowka. Kod czyta i opisuje `IDAT` jako chunk, sprawdza CRC i zachowuje go przy anonimizacji, ale piksele do FFT bierze przez OpenCV.

### Dlaczego nie usuwamy wszystkich ancillary chunks?

Bo ogolne wymaganie mowi o anonimizacji bez ingerencji w obraz. Niektore ancillary chunks wplywaja na renderowanie, np. gamma, profil koloru albo przezroczystosc. Kod usuwa metadane, ale zachowuje ancillary chunks istotne dla wygladu.

### Co oznacza `passed=True` w FFT?

Oznacza, ze po wykonaniu FFT i IFFT maksymalny blad bezwzgledny jest mniejszy lub rowny tolerancji. To potwierdza poprawny round-trip numeryczny.

### Dlaczego pokazujemy faze FFT?

Sam modul widma nie jest pelna informacja. Faza zawiera informacje o przesunieciu struktur obrazu. Dwa obrazy moga miec podobny modul, ale rozna faze.

### Dlaczego dla obrazu w pasy sa piki w widmie?

Pasy sa regularnym wzorem, czyli maja dominujaca czestotliwosc przestrzenna. FFT rozklada obraz na takie czestotliwosci, dlatego widac piki.

### Dlaczego dla czarnego obrazu widmo jest zerowe?

Bo wszystkie piksele maja wartosc 0. Nie ma ani sredniej jasnosci, ani zmian przestrzennych. FFT z samych zer daje same zera.

## 13. Co student powinien umiec powiedziec na obronie

1. PNG zaczyna sie 8-bajtowa sygnatura.
2. Po sygnaturze ida chunki: length, type, data, CRC.
3. `IHDR` ma 13 bajtow i zawiera rozmiar, glebie bitowa, typ koloru, kompresje, filtr i interlace.
4. Critical chunks sa rozpoznawane po wielkiej pierwszej literze typu chunka.
5. CRC liczymy po `chunk_type + chunk_data`.
6. `PLTE` to lista kolorow RGB po 3 bajty.
7. `IDAT` to skompresowane dane obrazu, ktore sa zachowywane bez zmian.
8. Dane po `IEND` nie sa potrzebne do obrazu i sa usuwane przy anonimizacji.
9. OpenCV wczytuje piksele, ale nie sluzy do recznego czytania metadanych.
10. OpenCV daje BGR/BGRA, dlatego kod konwertuje do RGB/RGBA.
11. FFT daje liczby zespolone.
12. Modul widma pokazuje sile czestotliwosci.
13. Faza pokazuje przesuniecie skladowych.
14. `fftshift` przenosi skladowa DC do centrum.
15. Logarytm widma ulatwia zobaczenie slabszych skladowych.
16. IFFT sluzy do sprawdzenia, czy FFT jest poprawnie odwracalna.
17. MSE, RMSE, MAE, max error i PSNR opisuja blad rekonstrukcji.
18. Czarny obraz powinien miec zerowe widmo i zerowy blad.
19. Obraz w pasy powinien miec wyrazne piki widma.
20. Anonimizacja usuwa metadane, ale zachowuje to, co potrzebne do obrazu.

## 14. Granice implementacji, o ktorych lepiej mowic wprost

- Kod nie jest pelnym parserem wszystkich mozliwych ancillary chunks.
- Kod nie jest pelnym parserem EXIF; pokazuje tylko podsumowanie.
- Kod nie dekompresuje recznie `IDAT`, bo do FFT uzywa OpenCV, co PDF dopuszcza.
- Kod nie przebudowuje strategii podzialu obrazu na chunki `IDAT`; zachowuje je w oryginalnej postaci.
- GUI normalizuje dane do wyswietlania, ale nie zmienia danych uzywanych do metryk.
- Projekt jest skoncentrowany na PNG, nie BMP ani WAV.

## 15. Minimalna sciezka prezentacji projektu

1. Uruchomic GUI.
2. Wczytac `Black.png`.
3. Pokazac `IHDR`, chunki, widmo, faze, rekonstrukcje i blad.
4. Wyjasnic, dlaczego czarny obraz ma zerowe widmo.
5. Wczytac `bws.png`.
6. Pokazac piki widma i wyjasnic zwiazek z pasami.
7. Wczytac plik z metadanymi, np. zawierajacy `tEXt`.
8. Pokazac opis chunkow.
9. Kliknac anonimizacje.
10. Pokazac raport: co usunieto, co zachowano, czy `IDAT` jest zachowany.
11. Uruchomic testy jednostkowe.
12. Wyjasnic FFT -> IFFT jako metode weryfikacji.
