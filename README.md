# kmer-counter

k-mer spectra and sequence complexity from the command line: word counts for
k = 1 to 8 over nucleotide or protein sequence, Shannon entropy against the
ceiling the sequence could actually reach, linguistic complexity, DUST and
Wootton-Federhen low-complexity regions along the sequence, word composition
against expectation, perfect tandem repeats, and alignment-free distances
between spectra. Pure Python with no third-party dependencies, and a
self-contained HTML dashboard that opens from a local disk with no network
connection and no JavaScript.

## Install

Install the prerelease from PyPI:

```bash
python -m pip install --pre norvara-kmer-counter
```

The Python package and command keep the shorter project name:

```python
import kmer_counter
```

```bash
kmer-counter --version
```

## Quickstart

```bash
kmer-counter count -i examples/aav2.fasta -i examples/huntingtin.fasta -k 6 --no-bars
```

```
> aav2.fasta
  FASTA  1 record  4,679 bp  DNA  GC=53.79%
  words    k=6  windows=4,674  distinct=2,476 of 4,096 observable  saturation=60.45%
  entropy  11.04 bits of 11.94 bits possible  (92.46%)  linguistic=0.5586  repeated=73.88% (floor 12.39%)
  regions  0 regions covering 0.00%  (dust>=20.00, window 64)
  repeats  longest tandem CGCT x 3.2 at 21-33  (300 in all)
  ! at k=6 there are 4,674 windows for 4,096 possible words, so at least 579 of them (12.4%) must belong to a repeated word whatever the sequence; read the share of windows in a repeated word against that floor, not against zero
> huntingtin.fasta
  FASTA  1 record  3,144 aa  protein
  words    k=6  windows=3,139  distinct=3,110 of 3,139 observable  saturation=99.08%
  entropy  11.58 bits of 11.62 bits possible  (99.68%)  linguistic=0.6070  repeated=1.05% (floor 0.00%)
  regions  13 regions covering 7.92%  (complexity<=2.20, window 12)
  repeats  longest run Q x 23.0 at 18-40  (41 in all)
  ! at k=6 the sequence has 3,139 windows against 64,000,000 possible words, so its entropy is capped by its length at 11.62 bits and is not comparable with that of a longer sequence
```

The alphabet is read from the sequence, so the DNA genome and the protein are
measured in their own alphabets in one command. Word-abundance bars are printed
unless `--no-bars` is given:

```
  most abundant words of 4
    QQQQ         20  ########################
    PPPP         15  ##################
    EEEE          5  ######
```

`profile` scores complexity along each sequence and merges the windows that
score badly into regions:

```bash
kmer-counter profile -i examples/huntingtin.fasta
```

```
> huntingtin.fasta
  protein  1 record  3,144 aa  window=12  step=1  complexity<=2.20
  13 regions covering 249 residues (7.92%)
  NP_002102.4
            12-86          75 residues  peak=0.00  mean=0.96
           291-304         14 residues  peak=1.90  mean=2.05
```

That first region is huntingtin's polyglutamine and polyproline tract, which
is what protein low-complexity calling is for.

`compare` measures the distance between spectra without aligning anything:

```bash
kmer-counter compare -i examples/aav2.fasta -i examples/phix174.fasta -k 4
```

```
  cosine distance between 2 spectra at k=4
               aav2.fa…  phix174…
  aav2.fasta     0.0000    0.2158
  phix174.fasta  0.2158    0.0000
  closest: aav2.fasta and phix174.fasta at 0.2158
```

Two viral genomes of similar size sit a fifth of the way apart in 4-mer
composition; the same file compared with itself sits at zero.

## What it measures

- **The spectrum** is the count of every word of length k in a window that
  advances one residue at a time. `distinct` is how many different words
  occurred, `observable` how many could have, and `saturation` the ratio: one
  means every window held a word seen nowhere else.
- **Entropy** is the Shannon entropy of the word counts, in bits, reported
  beside the ceiling the sequence could reach at that word length. The ceiling
  is the highest entropy that many windows could carry over that word space,
  allocated as evenly as whole windows allow: a sequence with 3,139 windows
  cannot show 64 million words however complex it is, and 4,674 windows over
  4,096 words cannot be divided evenly either.
- **Linguistic complexity** is the product over word lengths of words seen
  against words observable, after Trifonov. Unlike entropy it is sensitive to
  words that are absent rather than merely rare.
- **Low-complexity regions** come from the DUST score for nucleotide sequence
  and Wootton-Federhen complexity for protein - the measures BLAST and SEG mask
  with - taken on a sliding window and merged where they overlap.
- **Composition** gives every word of one, two and three residues with the
  count its residue frequencies imply, so CpG depletion or a codon-wobble
  signature is visible as a ratio rather than inferred from a raw count.
- **Repeats** are perfect homopolymer runs and perfect tandem repeats of units
  up to six residues, reported with their positions.
- **Distances** between spectra are cosine, Jensen-Shannon and Bray-Curtis.

## Options

```
count
-i, --input PATH         file to read, or '-' for standard input;
                         repeat for several files
    --input-format       auto | fasta | fastq (default: auto)
    --alphabet           auto | dna | rna | protein (default: auto)
    --limit N            stop after the first N records of each input
-k, --k K                word length to report, 1 to 8; every shorter length
                         is counted too (default: 4)
    --canonical          count a word and its reverse complement as one word
    --window N           window for the complexity profile (default: 64 for
                         nucleotide, 12 for protein)
    --step N             advance between reported windows (default: 1)
    --threshold SCORE    score at which a window is low-complexity
                         (default: 20 for DUST, 2.2 bits for protein)
    --tracks N           how many records keep a full window profile for the
                         report (default: 4)
    --top N              how many abundant words to report (default: 10)
    --min-run N          shortest homopolymer run to report (default: 5)
    --max-period N       longest tandem unit to search for (default: 6)
    --no-bars            omit the word-abundance bars from the summary

profile                  the window options above, plus
    --regions PATH       the low-complexity regions as a TSV
    --bed PATH           the regions as BED, 0-based and half-open

compare
-k, --k K                word length to compare at (default: 4)
    --measure            cosine | jensen-shannon | bray-curtis
    --canonical          count a word and its reverse complement as one word
```

## Outputs

| Flag | Output |
| --- | --- |
| `--format table` (default) | the readable per-input summary on stdout |
| `--format tsv` | one row of summary metrics per input, on stdout |
| `--format json` \| `none` | every metric as JSON, or nothing |
| `--output PATH` \| `-` | the word table (`count`), the scored windows (`profile`), the pairwise distances (`compare`) |
| `--report PATH` \| `-` | the self-contained HTML dashboard (`count`) |
| `--regions PATH`, `--bed PATH` | the low-complexity regions (`profile`) |

Any combination may be requested in a single pass. Compressed input is
recognised from its magic bytes, so gzip, bzip2 and xz files need no flag and
no particular file name.

The JSON carries every summary metric, the per-length table, the regions, the
repeats, the per-record rows, the dinucleotide signature and the composition
tables, with two bounded exceptions that the document names itself rather than
leaving to be noticed: a spectrum of more than 4,096 distinct words is cut to
the abundant ones and flagged `words_complete: false`, and a composition table
of more than 4,096 words - protein triples are 8,000 - is listed under
`tables_omitted`. Both are written in full by `--output`.

Either file output may be written to standard output as `-`, which makes the
rows pipeable; the summary then has to be silenced with `--format none`, since
two documents cannot share one stream.

Every input is opened, and every output destination resolved and checked,
before any input is read, so a run that cannot finish does not start. An output
is refused if it names an input under any name - a hard link or a symlink is
the same file, and comparing names alone would let a run truncate its own input
- and two outputs are refused if they name one file, since the second would
silently replace the first. With several inputs the per-input report names are
generated rather than typed, so they are resolved and checked in the same
preflight.

## Reading the output

- **Windows overlap.** A sequence of L residues has L - k + 1 windows and every
  residue but the ends sits in k of them, so the counts are not independent
  observations. That is what makes a spectrum sensitive to repeats, and it is
  also why the z-scores beside the composition tables are indicative rather
  than a test.
- **Entropy is capped by length.** Once the word space is larger than the
  number of windows, the entropy is bounded by log2 of the window count, not by
  composition, and two sequences of different lengths cannot be compared at
  that word length. The tool says so rather than leaving it implicit.
- **Length forces some repetition.** With more windows than there are possible
  words, some word has to recur: 4,674 windows over 4,096 six-mers force at
  least 579 of them, an eighth, into a repeated word whatever the sequence is.
  That floor is printed beside the observed share, because the share means
  nothing read against zero. It is a floor and not a verdict: the AAV2 genome's
  observed 73.88% at k = 6 is far above its 12.39% floor, so most of it is the
  sequence rather than the arithmetic.
- **Ambiguity is skipped, not guessed.** A window holding an N, a gap, a stop or
  a residue outside the twenty is not counted, and the number of such windows is
  reported. A sequence that is a quarter N has a spectrum taken over three
  quarters of its windows.
- **Thresholds belong with their windows.** The DUST threshold of 20 assumes a
  window of 64, where the score runs from 0 to 31; in a window of 12 the highest
  possible score is 5 and no window can ever be called. Change one and the tool
  tells you the other no longer fits.
- **Only perfect repeats are reported.** A tandem repeat with one substitution
  in it is reported as two repeats, and a region that is 95% periodic is not
  reported at all. Everything reported can be checked against the sequence by
  position.
- **Distances are compositional.** Two sequences built from the same words in a
  different order are identical to every measure here, so a small distance is
  evidence of similar composition and not of homology.
- **The alphabet is read from the first records,** because the whole file does
  not fit in a pipe. A sequence counts as nucleotide when at least 90% of its
  letters are A, C, G, T, U or N. Name it with `--alphabet` when that is wrong;
  prose written only in letters is valid protein and cannot be detected as
  anything else.

Anything a reader needs in order to read a number correctly is printed in the
terminal summary after a `!`, shown in the report, and carried in the JSON under
`qualifications`, because a program reading the JSON cannot see a warning
printed beside it.

## The HTML report

```bash
kmer-counter count -i examples/aav2.fasta -k 6 --report aav2.html --format none
```

One document, no network, no script: the word-abundance and abundance-census
charts, the complexity table, the score and entropy tracks along each sequence
with the called regions shaded, the pair odds-ratio grid, the triples furthest
from expectation, the repeats and the per-record table. Every table in it can
be downloaded from the page itself as the same tab-separated file `--output`
would have written. Where a chart column holds more than one window the band
shows the range and the line the mean, so a single low-complexity window in a
megabase is not averaged out of sight.

## Examples

`examples/` holds four real NCBI records, downloaded unmodified:
`aav2.fasta` (NC_001401.2), `phix174.fasta` (NC_001422.1), `aav2_vp1.fasta`
(YP_680426.1) and `huntingtin.fasta` (NP_002102.4). Re-fetch them with
`python examples/fetch_examples.py`.

## References

- Morgulis A, Gertz EM, Schäffer AA, Agarwala R (2006). A fast and symmetric
  DUST implementation to mask low-complexity DNA sequences. *Journal of
  Computational Biology* 13:1028-1040.
- Wootton JC, Federhen S (1993). Statistics of local complexity in amino acid
  sequences and sequence databases. *Computers & Chemistry* 17:149-163.
- Karlin S, Burge C (1995). Dinucleotide relative abundance extremes: a genomic
  signature. *Trends in Genetics* 11:283-290.
- Trifonov EN (1990). Making sense of the human genome. In *Structure and
  Methods*, Adenine Press.

## License

PolyForm Noncommercial License 1.0.0. See [LICENSE](LICENSE).
