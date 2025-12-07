# preprocessing_all_in.md

## Opis przetwarzania i pozyskiwania danych

W ramach projektu "Dane Bez Twarzy" proces przetwarzania danych wejściowych, mający na celu ich anonimizację, opiera się na hybrydowym podejściu łączącym techniki oparte na wyrażeniach regularnych (RegEx) oraz modelach z uczeniem maszynowym (NER - Named Entity Recognition).

### 1. Warstwa RegEx (anonymizer/regex_layer.py)

Pierwszym etapem przetwarzania jest zastosowanie warstwy RegEx. Warstwa ta jest odpowiedzialna za identyfikację i zastępowanie danych wrażliwych o stałym, predefiniowanym formacie. Przykłady takich danych to:
- PESEL (`[pesel]`)
- Adresy e-mail (`[email]`)
- Numery telefonów (`[phone]`)
- Numery kont bankowych (IBAN) (`[bank-account]`)
- Numery kart kredytowych (`[credit-card-number]`)
- Numery dowodów osobistych (`[document-number]`)

Wyrażenia regularne pozwalają na szybką i precyzyjną detekcję tych wzorców, zapewniając wysoką skuteczność dla danych o ustrukturyzowanym charakterze.

### 2. Warstwa NER (anonymizer/ner_layer.py)

Po wstępnej anonimizacji za pomocą RegEx, tekst jest przekazywany do warstwy Named Entity Recognition. Warstwa ta wykorzystuje zaawansowany model językowy (HerBERT, dostrojony do zadania token-classification dla języka polskiego) do identyfikacji encji nazwanych w tekście. W przeciwieństwie do RegEx, NER jest w stanie rozpoznawać dane wrażliwe na podstawie kontekstu, co jest kluczowe dla:
- Imion i nazwisk (`[name]`, `[surname]`)
- Miejscowości i adresów (`[city]`, `[address]`)
- Nazw firm (`[company]`)
- Dat (`[date]`)
- Wieku i płci (`[age]`, `[sex]`)
- Informacji o relacjach rodzinnych (`[relative]`)
- Informacji zdrowotnych (`[health]`)

Model HerBERT (`allegro/herbert-base-cased` lub jego lokalna wersja `models/herbert_ner_v2`) jest ładowany z biblioteki `transformers` i działa w trybie offline, jeśli wskazana jest lokalna ścieżka.

### Podsumowanie

Połączenie tych dwóch warstw pozwala na kompleksową anonimizację szerokiego zakresu danych osobowych i wrażliwych. Dane są pozyskiwane z plików treningowych `nask_train/original.txt` i `nask_train/anon.txt` i są przetwarzane w taki sposób, aby zachować strukturę i długość linii (format 1:1), zastępując wrażliwe informacje odpowiednimi tagami w nawiasach kwadratowych, np. `[pesel]`, `[name]`.
