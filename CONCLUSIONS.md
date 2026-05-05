# Struktura i polaryzacja

## Sieć głosowań — kolorowanie wg klubu (`fig1_network_70/99`)

Kolorowanie wg klubu pokazuje, czy posłowie tego samego klubu skupiają się razem, czy są rozproszeni.

### Pytania badawcze

**Czy polska scena polityczna jest spolaryzowana strukturalnie — tj. czy koalicja i opozycja tworzą dwa oddzielne klastry w sieci?**

> **Wniosek:** Wyraźne skupiska kolorów wskazują na silną homofilię partyjną (KO i PiS). KO łączy się z innymi partiami, a PiS nie.

**Jak zmienia się struktura sieci przy wyższym progu (99%)? Czy pozostaje jeden komponent, czy sieć rozpada się na podgrupy?**

> **Wniosek:** Patrz wykres `fig10_edges_vs_threshold`.

---

## Centralność pośrednictwa (`fig1_betweenness_70/99`)

Węzły o wysokiej betweenness to „mosty" informacyjne — posłowie, których usunięcie najbardziej rozspójniłoby sieć.

### Pytania badawcze

**Kto jest strukturalnie kluczowy dla przepływu konsensusu w parlamencie? Czy to liderzy frakcji, czy „rebel bridgers"?**

> **Wniosek:** Wskazani posłowie łączą koalicję jako element spójności.

**Czy posłowie o wysokiej betweenness należą do jednego klubu, czy rozkładają się między koalicją a opozycją?**

---

## Równowaga strukturalna (`fig25/26`)

- **Czy polska polityka jest „zbalansowana" w sensie Heidera** („wróg mojego wroga jest moim przyjacielem")?  
  99% trójkątów jest zbalansowanych — potwierdza to, że dwa wyraźne obozy są stabilne.

- **Które trójkąty są niezbalansowane i kto je tworzy?**  
  Potencjalne anomalie polityczne wymagają dalszej analizy.

---

## Najbardziej „oryginalny" wynik dla pracy

> **Sieć głosowań jest małym światem (σ = 3.26), wysoko asortatywna wg klubu (r = 0.46), ale jednocześnie 99% trójkątów jest zbalansowanych — co oznacza, że parlamentarna polaryzacja jest strukturalnie stabilna i przewidywalna, nie chaotyczna.**

To daje narrację: **polaryzacja istnieje, ale ma regularną, „geometryczną" strukturę.**

---

## Wykrywanie społeczności — algorytm Leiden

### Próg 70% — parlament to dwa bloki, nie wiele partii

Leiden wykrywa **3 społeczności** mimo że jest 13 klubów. Oznacza to, że z perspektywy topologii sieci koalicja głosuje jak jeden podmiot, podobnie opozycja. Lewica i PSL-TD mają różne programy, ale w głosowaniach są nierozróżnialne — to silny wynik.

- **NMI = 0.675** — partie wyjaśniają tylko 67% struktury, bo Leiden upraszcza do bloków, ignorując granice klubowe wewnątrz każdego bloku.
- **100 mismatchów (22%)** — posłowie, których klub jest w innym bloku niż ich sieciowe sąsiedztwo — potencjalni mosty lub dysydenci blokowi.

### Próg 99% — każda partia to osobna wyspa

Leiden wykrywa **8 społeczności**, **NMI = 0.971** — topologia sieci niemal idealnie odtwarza przynależność klubową bez korzystania z etykiet partyjnych.

To znaczy, że przy bardzo silnej zgodności głosowań partie są naprawdę spójnymi, odizolowanymi grupami. Konfederacja nie głosuje jak PiS nawet jeśli oboje są opozycją — na tym poziomie są odrębni.

**Tylko 15 posłów (4.2%)** to strukturalni outlierzy — warto sprawdzić, kto to jest; mogą to być osoby które zmieniły klub lub głosowały nielojalnie na tyle konsekwentnie, że sieciowo „należą" do innej partii.

### Główny wniosek

> **Polska scena polityczna ma dwupoziomową strukturę:** na poziomie bloków (koalicja/opozycja) granice są twarde i wyraźne w sieci; na poziomie partii wewnątrz bloku granice istnieją, ale są słabsze — co Leiden wykrywa dopiero przy progu 99%.

Można to sformułować jako:

> *"Voting behavior in the Polish Sejm is organized primarily along coalition–opposition lines, with intra-bloc party distinctions emerging only at the highest agreement thresholds."*

---

## Analiza betweenness — dwa różne obrazy

### Próg 70% — mosty między blokami

**Tomasz Rzymkowski** (Demokracja, bt = 0.024) i **Sławomir Zawiślak** (Konfederacja_KP, bt = 0.011) dominują z ogromną przewagą nad resztą. To posłowie, którzy strukturalnie łączą koalicję z opozycją — prawdopodobnie głosują nieregularnie, raz z jednym blokiem, raz z drugim. Przy progu 70% (gęsta sieć) właśnie tacy „szwędający się" posłowie stają się mostami między skupiskami.

Reszta top 12 to głównie Lewica, Demokracja i niezależni — małe kluby/niezależni siedzący na granicy bloków.

### Próg 99% — mosty wewnątrz partii

Przy 99% betweenness mierzy coś innego: **kto łączy podgrupy wewnątrz własnej partii**. Wartości są dużo wyższe (0.02–0.05), co znaczy że te osoby są naprawdę krytyczne strukturalnie.

- **Bogusław Wołoszański** (KO, bt = 0.045) — łączy podgrupy wewnątrz KO.
- **Agnieszka Buczyńska** i **Paulina Hennig-Kloska** — mosty w Polska2050/Centrum (mały klub, jeden człowiek może domykać spójność).
- W top 12 przy 99% są **3 posłowie PiS** (Wojtyszek, Warwas, Soin) — łączą podgrupowania wewnątrz PiS.
