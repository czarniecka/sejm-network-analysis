# Notatka do prezentacji plakatu — Parliamentary Network Analysis (Sejm RP X kadencja)

---

## 1. Metodologia i Dane — jak to zbudowaliśmy

- **Źródło:** oficjalne API Sejmu RP X kadencji (dane od początku kadencji 2023–present)
- **Skala:** setki głosowań, 459 aktywnych posłów, 10 klubów parlamentarnych
- **Preprocessing:** odfiltrowano nieaktywnych posłów, znormalizowano dane o nieobecnościach — badamy rzeczywiste poglądy, nie to kto częściej chorował lub był na delegacji

### Model sieci (graf)
- **Wierzchołki (Nodes):** konkretni posłowie, przypisani do swoich klubów
- **Krawędzie (Edges):** łączą posłów, grubość (waga) = siła ich zgody w głosowaniach
- Graf nieskierowany, ważony: `G = (V, E)`

### Wskaźnik podobieństwa `S_ij`
- Stosunek identycznych głosów do wszystkich wspólnych głosowań
- `S_ij = Σ I(v_ik = v_jk) / |V_ij|`
- Licznik: liczba sytuacji, gdzie poseł A i B głosowali tak samo (obaj "Za" lub obaj "Przeciw")
- Mianownik: liczba wspólnych głosowań
- Wynik zawsze w przedziale [0, 1] — pozwala obiektywnie zmierzyć dystans ideologiczny **bez patrzenia na przynależność partyjną**

---

## 2. Statystyki zbioru danych

- **459** aktywnych posłów
- **105 101** par poselskich z co najmniej 50 wspólnymi głosowaniami (próg antystatystyczny)
- **10** klubów parlamentarnych śledzonych w trakcie kadencję

---

## 3. Właściwości sieci

- **Small-world structure (σ = 3.26):** posłowie są silnie skupieni wewnątrz partii, ale ścieżki między dowolnymi dwoma posłami z różnych obozów są zaskakująco krótkie
- **Assortative by club (r = 0.46):** posłowie dobierają się głównie w obrębie swoich klubów — mamy na to dowód matematyczny
- **Równowaga strukturalna Heidera:** 99% trójkątów głosowań spełnia zasadę "wróg mojego wroga jest moim przyjacielem" — polaryzacja w polskim Sejmie jest **geometrycznie stabilna i przewidywalna**, nie chaotyczna

> **Kluczowa narracja:** polaryzacja istnieje, ale ma regularną, geometryczną strukturę.

---

## 4. Analiza progów — trzy fazy Sejmu

**Eksperyment:** stopniowe zwiększanie progu podobieństwa `S_ij` i obserwowanie, kiedy znikają "mosty" między posłami różnych obozów.

### Faza 1: Szeroki Konsensus (próg 30%)
- Pozostaje **104 760 krawędzi** — sieć niemal w pełni połączona
- Wniosek: w głosowaniach technicznych, proceduralnych i oczywistych ustawach Sejm jest niemal jednomyślny — podziały partyjne praktycznie nie istnieją

### Faza 2: Międzypartyjny "Kręgosłup" (próg 50%–90%)
- Liczba krawędzi spada powoli: **52 169 → 46 878**
- Wniosek: istnieje trwały "kręgosłup" porozumienia ponad podziałami — nawet przy wymogu 90% zgodności blisko połowa wszystkich par poselskich współpracuje ze sobą

### Faza 3: Kolaps i Twarda Polaryzacja (próg 99%)
- Gwałtowny spadek do **14 757 krawędzi** (~14% wszystkich par)
- Wniosek: to moment "prawdy" — ujawnia się żelazna dyscyplina partyjna, sieć rozpada się na izolowane kluby
- **99% to magiczna granica** — dopiero ekstremalne wymagania co do lojalności pokazują prawdziwe granice między partiami

---

## 5. Analiza Czasowa — Frekwencja Klubów

- **Koalicja (KO, PSL-TD, Polska 2050, Lewica):** frekwencja stabilnie powyżej 90%
  - Interpretacja: przy niewielkiej przewadze mandatowej każdy głos ma znaczenie — dyscyplina obecności jest bezpośrednio powiązana z możliwością sprawowania władzy
- **Opozycja i małe kluby:** systematycznie niższa i bardziej chwiejna frekwencja
  - Interpretacja "Small Club Effect": gdy wynik głosowania jest przesądzony przez stabilną większość, "koszt" nieobecności posła opozycji jest niski — ich głosy rzadko są decydujące (pivotal)
- **Konfederacja:** najbardziej zmienna frekwencja, z ostrymi, powtarzającymi się spadkami (sięgającymi 75% w 2025–2026)

---

## 6. Analiza Czasowa — Spójność Wewnętrzna Konfederacji

### Pik: Luty–Marzec 2024 (Konsolidacja)
- Konfederacja osiąga lokalne maksimum, niemal dorównując dyscyplinie partii koalicyjnych
- Przyczyny:
  - Jednoznaczne wsparcie protestów rolników przeciwko Zielonemu Ładowi — wspólny, twardy cel
  - Wybory samorządowe wymusiły wyciszenie sporów wewnętrznych
  - Narracja "Trzeciej Siły" wzmocniona przez chaos wokół mandatów Wąsika i Kamińskiego

### Najgłębszy kryzys: Czerwiec–Październik 2024 (Rozłam)
- Spójność spada poniżej 92% — najostrzejszy kryzys wewnętrzny w całym zbiorze danych
- Przyczyny:
  - Głosowania nad aborcją obnażyły fundamentalne różnice między skrzydłem wolnościowym (Nowa Nadzieja) a frakcją religijno-konserwatywną (Korona)
  - Kluczowi liderzy (Braun, Tyszka) wyjechali do Brukseli po wyborach do PE — osłabienie bieżącej dyscypliny
  - Afera korupcyjna w Trzeciej Drodze (sprawa Gomoły) wywołała wzajemne oskarżenia frakcji

### Stabilizacja: 2025–2026 (Profesjonalizacja)
- Spójność wraca powyżej 95% i stabilizuje się
- Przyczyny:
  - Efekt Mentzena: wczesne ogłoszenie kampanii prezydenckiej wymusiło lojalność całego ugrupowania
  - Aktywność Krzysztofa Bosaka jako Wicemarszałka "ucywilizowała" klub
  - Celowe głosowanie blokowe jako strategia przejęcia elektoratu po spadkach sondażowych PiS

---

## 7. Detekcja Społeczności — Algorytm Leiden

### Zasada działania
- Zaawansowany algorytm Leiden grupuje posłów wyłącznie na podstawie topologii głosowań — **algorytm nie wiedział, kto należy do jakiej partii**

### Wyniki

**Próg 70% — parlament to dwa bloki, nie wiele partii**
- Leiden wykrywa **3 społeczności** mimo 13 klubów
- NMI = 0.675 — partie wyjaśniają tylko 67% struktury
- 100 mismatchów (22%) — potencjalni mosty lub dysydenci blokowi
- Interpretacja: koalicja głosuje jak jeden podmiot — Lewica i PSL-TD mają różne programy, ale w głosowaniach są nierozróżnialne

**Próg 99% — każda partia to osobna wyspa**
- Leiden wykrywa **8 społeczności**, **NMI = 0.971**
- Topologia sieci w 97% odtwarza przynależność klubową bez etykiet partyjnych
- Konfederacja nie głosuje jak PiS nawet będąc razem w opozycji — na tym poziomie są odrębni
- **Tylko 15 posłów (4,2%)** to strukturalni outlierzy

### Posłowie "Pomostowi" (Bridge MPs)
- 15 osób, których algorytm przypisał do innej wspólnoty niż ich oficjalny klub
- Są to "rebelianci" głosujący wbrew własnemu klubowi lub osoby, które zmieniły barwy partyjne
- Ich pozycja (Betweenness Centrality) wskazuje, że pełnią rolę łączników między zwaśnionymi blokami

**Konkretne osoby — posłowie koalicji najbardziej zgodni z opozycją (fig18):**
- M. Stożek, M. Zawisa, M. Konieczny, A. Zandberg — wszyscy z **Razem** (ok. 39–41% zgody z opozycją)
- B. Wołoszański [KO] — 36.5%, K. Piekarska [KO], D. Jażłowiecka [KO] — ~35%
- Liczne nazwiska z **PSL-TD**: Grzyb, Nowogórska, Sawicki, Maliszewski, Tomczak, Żelazowska, Rzepa

**Posłowie opozycji najbardziej zgodni z koalicją:**
- **Z. Ziobro [PiS] — 39.4%** — ironicznie najbardziej "pomostowy" z całej opozycji
- L. Mejza, M. Pawłowska, R. Romanowski, Z. Rau — wszyscy PiS (~35–37%)
- **S. Mentzen [Konfederacja] — 34.9%** — jedyny przedstawiciel Konfederacji w zestawieniu

### Główny wniosek
> Polska scena polityczna ma **dwupoziomową strukturę**: na poziomie bloków (koalicja/opozycja) granice są twarde; na poziomie partii wewnątrz bloku granice istnieją, ale są słabsze — widoczne dopiero przy progu 99%.

---

## 8. Analiza Spektralna Laplasjanu

- **Metoda:** wektory własne znormalizowanego Laplasjanu grafu — pozycje węzłów wyznaczone **wyłącznie z topologii głosowań**, bez etykiet partyjnych

### Co pokazują poszczególne wektory własne:

- **Ev2 — wektor Fiedlera (λ = 0.530):** optymalny binarny podział grafu — wyraźnie oddziela koalicję rządzącą od opozycji wzdłuż osi X; mała wartość własna potwierdza strukturalnie głęboki i stabilny podział
- **Ev3 (λ = 0.946):** izoluje wyjątkowość Konfederacji — podczas gdy partie koalicyjne tworzą gęsty klaster, Ev3 wyciąga Konfederację daleko w górę, wskazując na wysoce odrębny wzorzec głosowań
- **Ev4 (λ = 0.995):** separacja outlierów i "rebeliantów" — partia Razem (niebieskie punkty na górze) i indywidualni posłowie PiS (na dole) wyraźnie się wyodrębniają

---

## 9. Analiza Betweenness — Kto jest "mostem"?

### Przy progu 70% — mosty między blokami koalicja–opozycja
- **Tomasz Rzymkowski** (Demokracja, bt = 0.024) i **Sławomir Zawiślak** (Konfederacja KP, bt = 0.011) — dominują z ogromną przewagą
- To posłowie strukturalnie łączący koalicję z opozycją — głosują nieregularnie, raz z jednym blokiem, raz z drugim
- Reszta top 12: głównie Lewica, Demokracja i niezależni — małe kluby siedzące na granicy bloków

### Przy progu 99% — mosty wewnątrz partii
- Betweenness mierzy tu kto łączy podgrupy wewnątrz własnej partii (wartości wyższe: 0.02–0.05)
- **Bogusław Wołoszański** (KO, bt = 0.045) — łączy podgrupy wewnątrz KO
- **Agnieszka Buczyńska** i **Paulina Hennig-Kloska** — mosty w Polska 2050/Centrum
- W top 12 są **3 posłowie PiS** (Wojtyszek, Warwas, Soin) — łączą ugrupowania wewnątrz PiS

---

## 10. DODATKOWE ANALIZY — Poza Plakatem

### Wiek posłów (fig14)
- **Konfederacja i Razem** to zdecydowanie **najmłodsze kluby** — mediana ok. 38–42 lat
- **PSL-TD** to najstarszy klub, z najszerszym rozkładem wiekowym (posłowie od ~35 do ~75 lat)
- **KO i PiS** mają mediany w okolicach 50–55 lat
- Słaba negatywna korelacja wieku z poziomem buntowniczości: **r = -0.189** — młodsi posłowie nieznacznie częściej głosują wbrew partii

### Płeć posłów (fig15)
- **Konfederacja:** ~5% kobiet — zdecydowanie najmniej ze wszystkich klubów
- **PSL-TD:** ~15% kobiet, **PiS:** ~18% kobiet
- **KO i Lewica:** ~40% kobiet
- **Razem:** najbardziej zbilansowany klub — ok. 50/50
- Ogólny rebel rate: mężczyźni 0.011 vs kobiety 0.006 — różnica **istotna statystycznie (p = 0.011)**
- W Konfederacji to jednak **mężczyźni** mają zdecydowanie wyższy rebel rate (~0.047 vs ~0.025 kobiet) — wewnętrzne spory napędzają głównie mężczyźni

### Rebelianci (fig3) — *(rozszerza info z plakatu)*
- **Top rebel: M. Jakubiak [niezależny] — 67% głosowań wbrew linii** (oderwany od wszystkich)
- P. Matysiak [niezależny] — 27.6%, S. Zawiślak [Konfederacja KP] — 26.5%, P. Kukiz — 20.7%
- T. Rzymkowski [Demokracja] — 14.1%, K. Berkowicz [Konfederacja] — 9.9%
- **Silna negatywna korelacja rebel rate z centralności eigenvektorowej: ρ = -0.78 (p < 10⁻¹⁰⁰)**
- Wniosek: im ważniejszy poseł w sieci (centralny), tym **nigdy** nie buntuje się przeciw partii — rebelianci są zawsze na obrzeżach sieci

### Tematy głosowań — Co dzieli, co jednoczy (fig22)
**Najbardziej polaryzujące tematy (koalicja vs. opozycja głosuje odwrotnie):**
- Ustawy ograniczające/regulujące (media, kryptowaluty, kodeksy karne) — zgoda koalicja–opozycja ok. 0.02–0.06
- Przepisy o odpowiedzialności, projektach budżetowych, sprawach konstytucyjnych

**Najbardziej jednoczące tematy (Sejm głosuje niemal jednomyślnie):**
- Obrona cywilna i bezpieczeństwo państwa (~0.84 zgodności)
- Uchwały parlamentarne, projekty uchwał (~0.78–0.79)
- Sprawy proceduralne i techniczne

**Wniosek:** Sejm jest podzielony w sprawach ideologicznych i politycznych, ale jednoczy się wokół bezpieczeństwa i procedur — spójne z wynikami analizy progowej.

### Trójkąty i Frustracja Strukturalna (fig25) — *(rozszerza info z plakatu)*
- Sieć zawiera próbkę **21 919 292 trójkątów** — zbalansowanych: **14 773 062**, niezbalansowanych: **143 262**
- **Globalny wskaźnik frustracji: 0.96%** (tylko 1% trójkątów łamie zasadę Heidera)
- **Frustracja wg klubu:**
  - **Konfederacja: 11.2%** — zdecydowanie poza normą; jako jedyna nie pasuje do schematu koalicja–opozycja
  - Wszystkie pozostałe kluby: ok. 0.6–0.8%
- **Top posłowie wg frustration index:**
  - K. Berkowicz [Konfederacja] — **46%**, S. Mentzen — **44.8%**, R. Wilk — **43.7%**, B. Foltyn — **35.1%**
  - Dalej: G. Piączek [Konfederacja] 3.4%, M. Pawłowska [PiS] 3%, K. Bosak 1.4%
- **Wniosek:** Konfederacja jest "anomalią" w binary świecie Sejmu — jej posłowie często głosują raz z koalicją, raz z opozycją, rozsadzając trójkąty zbalansowane

---

## 11. Kluczowe Wnioski do Zapamiętania

1. **Polaryzacja nie jest totalna** — większość czasu posłowie są ze sobą dość zgodni (głosowania proceduralne i techniczne)
2. **99% to magiczna granica** — dopiero ekstremalne wymagania co do lojalności ujawniają prawdziwe podziały między partiami
3. **Dwupoziomowa struktura:** koalicja vs. opozycja to poziom podstawowy; różnice wewnątrz bloków ujawniają się dopiero przy najwyższych progach
4. **Polaryzacja jest geometrycznie stabilna** — 99% trójkątów spełnia zasadę Heidera; to nie chaos, lecz przewidywalna struktura
5. **Głosowanie w Sejmie odzwierciedla przynależność partyjną niemal idealnie** — NMI = 0.971 bez użycia etykiet partyjnych
6. **Konfederacja jest strukturalnie najbardziej odrębna** — odróżniona nawet od PiS jako osobna "wyspa" przy każdym progu analizy
