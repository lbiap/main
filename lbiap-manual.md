---
title: "LBiaP manual"
output:
  pdf_document: default
  html_document: default
date: "2026-03-24"
---

# LBiaP 0.38 — User Guide

The **Lexical Bundles Identification and Analysis Program (LBiaP)** will
helps you measure the amount of independently-observed **lexical
bundles**—repeated multi-word sequences (n-grams)—in a folder of text
files. It’s designed for **corpus-driven discourse analysis**: you set
frequency/dispersion (range) thresholds, press a button, and LBiaP
produces CSV reports and browsable HTML pages that highlight bundles
right inside your texts.

You can find a glossary in Section "7) Concepts primer (descriptive
linguistics + corpus-driven discourse)"

------------------------------------------------------------------------

## 1) What you’ll need (once)

-   **Python 3.8+**

    -   Windows/macOS: get it from python.org and check “Add Python to
        PATH” during install (Windows).

-   **Libraries** Open “Command Prompt” (Windows) or “Terminal” (macOS)
    and run:

    ``` bash
    pip install pandas nltk
    ```

-   **NLTK tokenizer data** (one-time download):

    ``` bash
    python -c "import nltk; nltk.download('punkt')"
    ```

-   **Tkinter GUI**

    -   Comes with standard Python on Windows/macOS.

    -   On some Linux systems you may need:
        `sudo apt-get install python3-tk`.

    -   A **folder named `csv/`** (LBiaP writes working n-gram lists
        here).

## 2) Preparing your corpus

-   Put your texts as **`.txt` files** into a single folder. Each file
    is treated as one “document.”
-   Use **UTF-8** encoding.
-   It’s OK if your texts contain apostrophes/contractions; LBiaP
    handles those.

------------------------------------------------------------------------

## 3) Starting the app (no terminal knowledge required)

-   **Windows**: Double-click the `.py` file. If it opens in an editor,
    right-click → **Open with** → **Python**.

-   **macOS**: Double-click; if it opens in Python Launcher, choose
    **Run**. If that doesn’t work, use Terminal:

    ``` bash
    python3 lbiap.py
    ```

You’ll see a small window titled **“LBiaP 0.38”** with two buttons and
several entry fields.

```
mermaid
flowchart TD
    subgraph "User's Original Corpus Folder"
        OLDDIR["📁 Selected Corpus Folder<br/>(olddir)"]
        TXTFILES["📄 *.txt files<br/>(your language texts)"]
    end
    
    subgraph "Script Location"
        SCRIPT["🐍 LBiaP"]
        CSVFOLDER["📁 csv/ folder<br/>(temporary n-gram data)"]
    end
    
    OLDDIR -->|User selects this| SCRIPT
    SCRIPT -->|Copies, transforming text| PROCESSED_TXT
    
    subgraph "Processing Folder (Created)"
        HTML_FILES["📄 *.html files<br/>(readable versions)"]
        PROCESSED_TXT["📄 *.txt files<br/>(cleaned & tokenized)"]
        NEWDIR["📁 corpus_copyfix/<br/>(inside selected folder)"]
    end
    
    SCRIPT -->|Creates| CSVFOLDER
    CSVFOLDER -->|"3grams.csv to 9grams.csv"| NGRAM_FILES["📊 Functional taxonomy from extant articles."]
    
    subgraph "Output Files (in original folder)"
        RESULTS_INIT["📊 {filename}_results_initial.csv"]
        RESULTS_FINAL["📊 {filename}_results_final.csv"]
        FREQVALS["📊 {filename}_freqvals.csv"]
        FUNC_TAXONOMY["📊 {filename}_functionaltaxonomy_results.csv"]
        DELETED_COLS["📝 {filename}_deleted_columns.txt"]
        GRAM_CHECKS["📊 {n}gramcheck.csv (x7 files)"]
    end
    
    PROCESSED_TXT -->|Analysis produces| RESULTS_INIT
    RESULTS_INIT -->|Control for interlocking bundles| RESULTS_FINAL
    RESULTS_FINAL -->|Summary stats| FREQVALS
    NGRAM_FILES -->|Functional taxonomy| FUNC_TAXONOMY
    RESULTS_FINAL -->|Notes which columns got deleted| DELETED_COLS
        
    style OLDDIR fill:#e1f5fe
    style NEWDIR fill:#fff3e0
    style CSVFOLDER fill:#f3e5f5
    style RESULTS_FINAL fill:#c8e6c9
    style FREQVALS fill:#c8e6c9
```

------------------------------------------------------------------------

## 4) The interface at a glance

**Buttons**

-   **Select Corpus** — choose the folder that contains your `.txt`
    files.
-   **Generate LB Data** — runs the program.

**Frequency thresholds** (left column labels; right column boxes you can
edit)

-   **9-grams … 3-grams**: minimum total occurrences across the
    concatenated corpus for each n-gram length. Defaults (from top to
    bottom): `5, 5, 5, 5, 10, 20, 40`. Higher = stricter (fewer,
    stronger bundles).

**Range**

-   Minimum **number of files** in which a bundle must appear
    (dispersion). Default: `5`.

**Filename**

-   A short name used as a prefix for exported files (e.g.,
    `myproject`).

> Quick start: keep the defaults, set **Range = 3–5** depending on
> corpus size, set **Filename = your_project_name**, and go.

------------------------------------------------------------------------

## 5) What happens when you click “Generate LB Data”

Broadly speaking:

1.  **A safe copy of your corpus**

    -   LBiaP duplicates your folder into `corpus_copyfix/` so your
        originals stay untouched.

2.  **Tokenization & cleanup**

    -   Changes text to lowercase and uses NLTK’s tokenizer.
    -   Temporarily rewrites contractions (e.g., `"I'm"` →
        `mcontraction`) so multi-word counting is accurate.

3.  **N-gram candidate mining (9 → 3 words)**

    -   Counts all n-grams (9 down to 3) that meet your **frequency**
        cutoffs; writes lists to `csv/9grams.csv`, …, `csv/3grams.csv`.

4.  **Document-level counting**

    -   Builds a big table (one row per file in the copy, one column per
        candidate bundle) and counts occurrences.

5.  **Range (dispersion) filter**

    -   Keeps only bundles that appear in at least the number of files
        specified in the range field. .

6.  **Interlocking control**

    -   See Cortes & Lake (2023) for the role of interlocking bundles in
        vastly overinflating frequency counts.

7.  **Final reports & HTML**

    -   Exports CSV summaries (see next section) and creates HTML
        versions of your texts inside `corpus_copyfix/` with found
        bundles **bold + underlined** for quick browsing.

------------------------------------------------------------------------

## 6) Outputs you’ll get (and how to read them)

All exports appear next to your script (and inside `corpus_copyfix/` for
HTML).

-   `csv/9grams.csv … csv/3grams.csv` Candidate bundle lists by length
    after initial frequency filtering.

-   `frame2.csv` Internal working table of counts (one row per file,
    columns = candidates).

-   `FILENAME_results_initial.csv` Counts per file after early
    filtering, before interlocking bundle cleanup.

-   `FILENAME_deleted_columns.txt` Bundles removed during stricter
    frequency/range passes (helpful audit trail).

-   `FILENAME_results_final.csvT.csv` Final **presence/occurrence**
    matrix (transposed). Columns are files; rows are bundles (with
    contractions restored).

-   `FILENAME_freqvals.csv` The key summary table:

    -   `frequency` — total occurrences across the corpus.
    -   `range` — how many distinct files contain the bundle.
    -   `words` — bundle length in words. Use this to sort by importance
        (e.g., high range + high frequency).

-   `FILENAME_functionaltaxonomy_results.csv` *(if `bundledb.csv`
    provided)* Merges your final bundle list with your taxonomy, sorted
    by `Subcat`.

-   `corpus_copyfix/*.html` One HTML per text file with bundles
    highlighted; left sidebar links let you click through documents.

> Interpreting results:
>
> -   **Frequency** shows salience.
> -   **Range** shows **dispersion**, a strong indicator that a pattern
>     is characteristic of the register/genre rather than idiosyncratic
>     to one file.

------------------------------------------------------------------------

## 7) Concepts primer (descriptive linguistics + corpus-driven discourse)

-   **Lexical bundle**: a recurrent, fixed or semi-fixed multi-word
    sequence (e.g., *on the other hand*, *it is important to*). Bundles
    are surface strings—not necessarily idioms—and can be structurally
    incomplete (e.g., *the end of the*).

-   **N-gram**: a sequence of *n* tokens (3-grams, 4-grams, … 9-grams
    here).

-   **Frequency vs. Range (Dispersion)**:

    -   *Frequency* = total times a bundle occurs.
    -   *Range* = number of **different files** where it occurs. Range
        guards against a pattern that is frequent in just one text.

-   **Corpus-driven** vs. **corpus-based**:

    -   *Corpus-driven* minimizes prior assumptions; patterns “emerge”
        from data (LBiaP’s thresholds let patterns surface).
    -   *Corpus-based* typically tests pre-specified hypotheses.

-   **Why contractions are “expanded”**: LBiaP temporarily encodes
    contractions (e.g., `"'s"` → `scontraction`) to prevent tokenization
    from splitting bundles unpredictably. They’re restored in the final
    outputs.

-   **Interlocking bundles**: Longer bundles often contain shorter ones
    (e.g., *at the end of the* contains *the end of*). LBiaP marks
    detected sequences in the working copies to avoid inflated counts
    from overlaps.

-   **Functional taxonomy**: Many analyses group bundles into roles such
    as **Stance/Attitude** (e.g., *I think that*), **Text-organizing**
    (*on the other hand*), **Referential** (*the end of the*). Supplying
    `bundledb.csv` lets LBiaP attach such labels to your results.

------------------------------------------------------------------------

## 8) A basic workflow for discourse analysis

1.  **Run LBiaP** with sensible thresholds.
2.  Open **`FILENAME_freqvals.csv`** and sort by `range` (desc), then
    `frequency` (desc).
3.  Skim **`corpus_copyfix/*.html`** to see bundles **in context**
    (crucial for function).
4.  If you supplied **`bundledb.csv`**, read
    **`FILENAME_functionaltaxonomy_results.csv`** to see distribution by
    functional class.
5.  Iterate: tweak thresholds to focus on shorter vs. longer bundles, or
    to tighten dispersion.

------------------------------------------------------------------------

## 9) Troubleshooting (common fixes)

-   **“LookupError: Resource punkt not found”** Run:
    `python -c "import nltk; nltk.download('punkt')"`
-   **“No such file or directory: 'csv/…'”** Create a folder named
    **`csv`** next to the script before running.
-   **Weird characters / encoding errors** Save your `.txt` files as
    **UTF-8**.
-   **Very slow on huge corpora** Raise frequency thresholds, lower
    n-gram maximum (if you edit the code), or run on a smaller subset
    first. LBiaP already parallelizes tokenization with multiple
    threads.
-   **No GUI appears** Confirm Tkinter is installed (Linux:
    `python3-tk`). On macOS, try launching from Terminal with
    `python3 lbiap.py`.

------------------------------------------------------------------------

## 10) Selected References

-   Bestgen, Y. (2018). Evaluating the frequency threshold for selecting
    lexical bundles by means of an extension of the Fisher's exact test.
    Corpora, 13(2), 205-228.

-   Bestgen, Y. (2020). Comparing lexical bundles across corpora of
    different sizes: The Zipfian problem. Journal of Quantitative
    Linguistics, 27(3), 272-290.

-   Cortes, V. (2008). A comparative analysis of lexical bundles in
    academic history writing in English and Spanish. Corpora, 3(1),
    43-57.

-   Cortes, V. (2013). The purpose of this study is to: Connecting
    lexical bundles and moves in research article introductions. Journal
    of English for academic purposes, 12(1), 33-43.

-   Cortes, V. (2022). Lexical bundles in EAP. In The Routledge handbook
    of corpora and English language teaching and learning (pp. 220-233).
    Routledge.

-   Cortes, V. (2023). Lexical bundles and phrase frames. In Conducting
    genre-based research in applied linguistics (pp. 105-126).
    Routledge.

-   Cortes, V., & Lake, W. (2023). LBiaP: A solution to the problem of
    attaining observation independence in lexical bundle studies.
    International Journal of Corpus Linguistics, 28(2), 263-277.

------------------------------------------------------------------------
