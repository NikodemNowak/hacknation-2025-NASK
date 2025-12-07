# synthetic_generation_all_in.md

## Generowanie danych syntetycznych

Moduł generowania danych syntetycznych w projekcie "Dane Bez Twarzy" ma za zadanie zastępować zanonimizowane tagi (np. `[name]`, `[city]`) realistycznymi, ale fikcyjnymi informacjami. Dzięki temu możliwe jest tworzenie zbiorów danych, które zachowują strukturalne i semantyczne właściwości oryginalnych danych, jednocześnie chroniąc prywatność.

### Mechanizm działania

Generator syntetyczny (`anonymizer/synthetic.py`) wykorzystuje hybrydowe podejście:
1.  **Słowniki wewnętrzne**: Dla niektórych typów tagów (np. imiona, nazwiska, miasta, ulice) system może korzystać z predefiniowanych list popularnych danych, aby podstawiać realistyczne wartości. To zapewnia szybkość i przewidywalność dla często występujących kategorii.
2.  **Model językowy (PLLuM)**: W przypadku bardziej złożonych lub kontekstowych danych, a także w celu zapewnienia większej różnorodności i realizmu, wykorzystywany jest duży model językowy (Large Language Model - LLM) - PLLuM (`CYFRAGOVPL/pllum-12b-nc-chat-250715`).
    *   **PLLuMClient** (`anonymizer/pllum_client.py`) służy do komunikacji z API modelu PLLuM, co pozwala na generowanie danych syntetycznych o wysokiej jakości, zgodnych z kontekstem zdania i realiami języka polskiego.

### Walka z fleksją

Generowanie syntetycznych danych w języku polskim stwarza wyzwanie związane z odmianą (fleksją). Wiele słów w języku polskim zmienia swoją formę w zależności od przypadku, rodzaju i liczby.
*   **Przykład problemu**: Jeśli oryginalne zdanie brzmi "Mieszkam w {city}.", proste podstawienie nazwy miasta bez uwzględnienia odmiany (np. "Mieszkam w Radom.") jest niepoprawne gramatycznie.
*   **Sukces**: System dąży do generowania poprawnych gramatycznie fraz, np. "Mieszkam w Radomiu." (miejscownik).
*   **Porażka**: Mimo zaawansowanych mechanizmów, w pełni automatyczne i zawsze poprawne generowanie syntetyczne z uwzględnieniem wszystkich zasad fleksji jest zadaniem bardzo trudnym. PLLuM, jako zaawansowany LLM, jest w stanie w wielu przypadkach poprawnie odmienić podstawiane dane, bazując na kontekście, w którym tag został umieszczony. Jednakże, sporadycznie mogą pojawić się sytuacje, gdzie generowane słowo nie jest w pełni zgodne gramatycznie, zwłaszcza dla rzadkich form lub w nietypowych konstrukcjach zdaniowych.

### Dbałość o sens

Rozwiązanie dba o sens generowanych danych poprzez:
*   **Kontekstualizację**: PLLuM, jako model kontekstowy, analizuje otoczenie tagu, co pozwala na generowanie bardziej spójnych i realistycznych danych syntetycznych.
*   **Podobieństwo inputów**: Celem jest, aby wygenerowane syntetyczne inputy były jak najbardziej podobne do danych wejściowych pod względem struktury i typu informacji, zachowując jednocześnie ich fikcyjny charakter. Randomizacja jest kontrolowana, aby unikać absurdalnych podstawień.

### Log z przykładami (Showcase)

Poniżej przedstawiono przykłady działania mechanizmu generowania syntetycznego:

*   **Szablon (zanonimizowany)**: "Spotkałem się z [name] w [city]."
*   **Wynik (syntetyczny)**: "Spotkałem się z Kasią w Gdańsku."

*   **Szablon (zanonimizowany)**: "Moim PESELem jest [pesel] i pracuję w [company]."
*   **Wynik (syntetyczny)**: "Moim PESELem jest 89012345678 i pracuję w "Innowacyjne Rozwiązania sp. z o.o."."

(Należy wstawić tutaj 3-5 par ciekawszych przykładów po wygenerowaniu danych)
