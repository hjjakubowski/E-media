# Projekt 2 - RSA dla PNG

## Zakres

Projekt szyfruje i deszyfruje pliki PNG algorytmem RSA. Obsługiwany zakres wejścia:

- PNG 8-bit,
- color type 0, 2 albo 6,
- brak interlace,
- jeden lub wiele chunkow IDAT,
- filtry PNG 0..4.

Zaszyfrowany plik jest zapisywany jako poprawny PNG 16-bitowy.

## Modyfikacja naglowka i metadanych

RSA powieksza dane: blok jawny ma `key_bytes - 1` bajtow, a blok szyfrogramu ma
`key_bytes` bajtow. Z tego powodu szyfrogram nie miesci sie w oryginalnym
8-bitowym obrazie o tej samej liczbie pikseli.

Dlatego zaszyfrowany PNG zmienia `IHDR.bit_depth` z 8 na 16. Pozwala to uzyc
tych samych wymiarow obrazu, ale podwoic pojemnosc danych obrazu.

Projekt dodaje prywatny chunk pomocniczy `rsAP`. Zawiera on tylko dane techniczne
potrzebne do deszyfrowania:

- tryb szyfrowania,
- typ zaszyfrowanego ladunku,
- rozmiary danych jawnych i szyfrogramu,
- rozmiar klucza,
- IV dla trybu CHAIN.

Istniejace chunki metadanych PNG sa zachowywane. Zmieniany jest `IHDR`, `IDAT`
oraz dodawany jest `rsAP`.

## Metoda 1: szyfrowanie danych po dekompresji

Standardowa komenda `encrypt` wykonuje:

1. odczyt PNG,
2. polaczenie wszystkich chunkow IDAT,
3. dekompresje zlib,
4. usuniecie filtrow PNG 0..4,
5. szyfrowanie bajtow pikseli RSA,
6. zapis szyfrogramu jako danych obrazu 16-bit,
7. ponowna kompresje zlib.

Ten wariant daje poprawny, otwieralny PNG po zaszyfrowaniu.

## Metoda 2: szyfrowanie skompresowanych danych IDAT

Komenda `encrypt-compressed` szyfruje bezposrednio skompresowane bajty IDAT.
Nie wykonuje dekompresji przed RSA.

Po deszyfrowaniu komenda `decrypt-compressed` odtwarza oryginalny skompresowany
strumien IDAT. Ta metoda jest przydatna do porownania wymaganego w tresci
projektu.

Metody nie sa rownowazne:

- metoda 1 szyfruje dane pikseli i tworzy nowy strumien zlib,
- metoda 2 szyfruje juz skompresowany strumien zlib i po deszyfrowaniu odtwarza
  go bajt po bajcie.

## ECB i CHAIN

Tryb ECB szyfruje kazdy blok niezaleznie. Ten sam blok jawny daje ten sam blok
szyfrogramu, dlatego w obrazach z powtarzalnymi obszarami moze zostac widoczna
struktura.

Tryb CHAIN miesza blok jawny z poprzednim blokiem szyfrogramu oraz IV. Dzieki
temu identyczne bloki jawne nie powinny dawac identycznych blokow szyfrogramu.

Komenda `report` generuje zaszyfrowane obrazy ECB i CHAIN oraz raport tekstowy
z liczba powtorzonych blokow.

## Porownanie z gotowym RSA

Komenda `compare-library` uzywa `pycryptodome` i tej samej pary kluczy RSA.
Biblioteczne RSA uzywa paddingu PKCS#1 v1.5, dlatego wynik szyfrowania rozni sie
od surowego deterministycznego RSA z projektu.

Oczekiwany wniosek:

- oba warianty poprawnie deszyfruja dane,
- szyfrogramy nie sa identyczne,
- biblioteczny RSA jest probabilistyczny dzieki losowemu paddingowi,
- surowy RSA w ECB jest deterministyczny.

## Przykladowe komendy

```bash
python3 Projekt_2/src/main.py keygen --bits 512
python3 Projekt_2/src/main.py encrypt --input Projekt_2/data/PWr_gray.png --output Projekt_2/output/PWr_gray_ecb.png --mode ecb
python3 Projekt_2/src/main.py decrypt --input Projekt_2/output/PWr_gray_ecb.png --output Projekt_2/output/PWr_gray_decrypted.png
python3 Projekt_2/src/main.py encrypt-compressed --input Projekt_2/data/PWr_gray.png --output Projekt_2/output/PWr_gray_compressed_ecb.png --mode ecb
python3 Projekt_2/src/main.py compare-compression --input Projekt_2/data/PWr_gray.png --mode ecb
python3 Projekt_2/src/main.py report --input Projekt_2/data/PWr_gray.png
python3 Projekt_2/src/main.py compare-library --input Projekt_2/data/PWr_gray.png
python3 Projekt_2/src/main.py gui
```

GUI uruchamia te same funkcje co CLI. Pozwala wybrac PNG, wygenerowac klucze,
zaszyfrowac obraz w trybie ECB lub CHAIN, odszyfrowac wynik, porownac metody
kompresji, wygenerowac raport ECB/CHAIN oraz wykonac porownanie z bibliotecznym
RSA.
