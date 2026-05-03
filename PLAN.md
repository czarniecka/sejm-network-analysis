# Analiza Sejmu RP — obecna kadencja

**API:** [https://api.sejm.gov.pl/sejm.html](https://api.sejm.gov.pl/sejm.html)

> **Założenie ogólne:** Usuwamy polityków nieaktywnych.
---
## Pytania badawcze
### 1. Lojalność głosowania — czy politycy głosują przeciwko swojej partii?

**Pytania:** Czy jacyś politycy głosują częściej przeciwko swojej partii? 
Pytania poboczne: Jacy politycy są najbardziej zgodni? Czy pokrywa się to z partiami politycznymi?

| Element | Opis |
|---|---|
| **Node** | Poseł |
| **Edge** | Istnieje między dwoma posłami, jeśli głosowali tak samo (obaj ZA lub obaj PRZECIW lub obaj WSTRZYMANIE) |
| **Waga krawędzi** | % takich samych głosowań spośród tych, na których **obydwoje byli obecni** |
| **Kierunkowość** | Nieskierowana, ważona |

**Założenia filtrowania:**
- Politycy byli razem na co najmniej **X** wspólnych głosowaniach
- Ich wzajemna zgodność wynosi co najmniej **Y%**

**Do zbadania:**
- Wykres zgodności w zależności od progu Y (jak zmienia się liczba krawędzi wraz ze wzrostem wymaganej zgodności)
- Porównanie struktury sieci z oficjalnym podziałem na kluby (community detection)
- Identyfikacja "buntowników" — posłów z niską zgodnością z własnym klubem i wysoką z innymi
- Mosty między partiami: posłowie łączący różne społeczności

---
### 2. Skąd się rodzą politycy?

**Pytanie:** Gdzie się rodzą politycy? Czy widać podział na Polskę A i Polskę B z podziałem na partie?

| Element        | Opis                                        |
| -------------- | ------------------------------------------- |
| **Node typ A** | Poseł (atrybut: klub/partia)                |
| **Node typ B** | Miejscowość / gmina / województwo urodzenia |
| **Edge**       | Poseł → miejscowość urodzenia               |

**Projekcje jednomodalne:**
- **Sieć posłów** — połączeni, jeśli urodzili się w tej samej miejscowości lub powiecie
- **Sieć miejscowości** — połączone, jeśli urodził się tam poseł tej samej partii

**Do zbadania:**
- Rozkład geograficzny urodzin per partia
- Indeks dywersyfikacji regionalnej każdego klubu
- Węzły-huby: miejscowości z wieloma posłami różnych partii vs monopartyjne

---

### 3. Wykształcenie polityków

**Pytanie:** Jakie wykształcenie mają politycy? Czy koreluje to z przynależnością partyjną?

| Element | Opis |
|---|---|
| **Node typ A** | Poseł |
| **Node typ B** | Uczelnia / kierunek / poziom wykształcenia |
| **Edge** | Poseł ↔ uczelnia (studiował tam) |

**Projekcje jednomodalne:**
- **Sieć posłów** — połączeni, jeśli skończyli tę samą uczelnię lub kierunek
- **Sieć uczelni** — połączone, jeśli kończyli je posłowie z tego samego klubu

**Do zbadania:**
- Które uczelnie tworzą kliki w obrębie jednej partii?
- Czy kierunek wykształcenia koreluje z przynależnością partyjną?
- Stopień węzła uczelni jako miara "wpływu politycznego" danej uczelni

> **Uwaga:** API Sejmu podaje `educationLevel` (wyższe / średnie / etc.), ale nie nazwę uczelni.

---
### 4. Migracje polityków

**Pytanie:** Jak daleko od miejsca urodzenia kandydują politycy? Które regiony importują / eksportują polityków?

**Dane:** porównanie pól `birthLocation` i `districtName` z API.

| Element | Opis |
|---|---|
| **Node** | Region / województwo (urodzenia **lub** okręgu wyborczego) |
| **Edge (skierowany)** | `birthLocation` → `districtName` |
| **Waga krawędzi** | Liczba posłów na danej trasie migracji |
| **Self-loop** | Poseł kandyduje tam, gdzie się urodził (brak migracji) |

**Do zbadania:**
- **In-degree** regionu = ile polityków dany region "importuje" spoza
- **Out-degree** regionu = ile lokalnych polityków "eksportuje" do innych okręgów
- Czy migracja jest lokalna (sąsiednie województwa) czy daleka?
- Porównanie wzorców migracji per partia

---
### 5. Porównanie sieci ze znanymi modelami

**Dla każdej z powyższych sieci zbadać:**

- Rozkład stopni węzłów — czy power-law (sieć bezskalowa) czy rozkład Poissona (sieć losowa)?
- Współczynnik grupowania vs losowy graf Erdos–Renyi o tych samych parametrach
- Średnia długość najkrótszej ścieżki — efekt małego świata (small world)?
- Średnica sieci
- Liczba komponentów spójnych

| Model | Charakterystyka |
|---|---|
| Sieć losowa (ER) | Rozkład Poissona, niski clustering |
| Sieć bezskalowa (BA) | Rozkład potęgowy, huby |
| Small world (WS) | Wysoki clustering + krótkie ścieżki |

---
## Endpointy API

```

GET /sejm/term10/MP # lista wszystkich posłów

GET /sejm/term10/MP/{id} # dane posła: birthLocation, districtName, club, educationLevel

GET /sejm/term10/votings # lista głosowań

GET /sejm/term10/votings/{sitting}/{voting}/votes # kto jak głosował

```