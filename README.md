# Synthetic Speech Data Pipeline

A data preparation pipeline for building Egyptian Arabic TTS (Text-to-Speech) training datasets. The pipeline covers the full lifecycle from raw text collection through normalization to audio synthesis, targeting the Egyptian dialect specifically.

---

## Overview

This notebook was built to address a gap that anyone working on Arabic speech knows well: most publicly available TTS corpora cover Modern Standard Arabic, not Egyptian colloquial. The pipeline pulls text from three Egyptian Arabic sources, cleans and normalizes it into a form a TTS model can actually handle, and then synthesizes audio using Microsoft Edge TTS as the synthesis backend.

The normalization layer is the core of the work. Raw text from social media or Wikipedia contains numbers, dates, currency symbols, phone numbers, emojis, and abbreviations that a TTS model cannot reliably verbalize. Each of those surface forms needs to be expanded into spoken Egyptian Arabic before it ever reaches a synthesis engine.

---

## Data Sources

Three Excel files are loaded and concatenated into a single corpus:

- `TE_Sports.xlsx` — Egyptian sports commentary and news text
- `8.Arabic_Egyptian_Wikipedia.xlsx` — Wikipedia articles filtered to Egyptian Arabic
- `2.Egyptian Tweets.xlsx` — Social media text from Egyptian Twitter

All sources expose a `Text` column. After concatenation, the pipeline drops null rows and deduplicates, leaving a clean unique-sentence corpus.

---

## Normalization Pipeline

The normalization step converts raw Egyptian Arabic text into a fully verbalized form safe for TTS input. The processing order is intentional — more specific patterns are applied before more general ones to avoid partial-match collisions.

### Processing Order

1. Emoji removal
2. Arabic-Indic digit normalization (٠١٢٣ → 0123)
3. Date expansion (`15/3/2024` → `خمستاشر مارس ألفين وأربعة وعشرين`)
4. Time expansion (`1:30` → `واحدة ونص`)
5. Percentage expansion (`20%` → `عشرين فى المية`)
6. Currency expansion (`100 EGP`, `$12.50`, `€30` → verbalized with correct plural forms)
7. Numeric range expansion (`5-10` → `من خمسة لحد عشرة`)
8. Phone number digit-by-digit verbalization
9. Abbreviation expansion (`د.` → `دكتور`, `كم.` → `كيلومتر`)
10. Arabic letter unification (diacritics removed, variant forms normalized)
11. Remaining standalone number expansion

### Number Verbalization

Numbers are verbalized in Egyptian dialect, not Modern Standard Arabic. The distinction matters: a model trained on MSA pronunciations will produce unnatural output when deployed against Egyptian speech. The lookup tables cover:

- Units and teens (0–19) in colloquial form (`تلتة`, `اتناشر`)
- Tens with correct Egyptian endings (`تلاتين`, `ستين`)
- Hundreds (`مية`, `ميتين`, `تلتمية`)
- Thousands, millions, and billions with proper dual and plural forms

### Currency

Currency handling includes correct Arabic grammatical agreement. The `get_plural_form` function selects between singular, dual, and plural based on the numeric value, which is required for grammatically well-formed Arabic speech output.

Supported currencies: EGP, USD, EUR, and their common written variants (`ج`, `£`, `$`, `€`).

---

## TTS Synthesis Backends

### Edge TTS (active)

The production synthesis path uses `edge-tts`, Microsoft's neural TTS service accessed via Python. Two Egyptian Arabic voices are used in alternation:

- `ar-EG-SalmaNeural` (female)
- `ar-EG-ShakirNeural` (male)

Alternating voices produces a more varied dataset, which generally improves acoustic model training downstream.

### NAMAA Egyptian TTS (inactive)

The notebook includes code for loading the NAMAA-Egyptian-TTS model (`NAMAA-Space/NAMAA-Egyptian-TTS` on Hugging Face, built on the Chatterbox architecture). This was the preferred backend because it was trained on Egyptian Arabic specifically, but it was blocked by a dependency conflict in the Colab environment at the time of writing. The code is preserved for when the incompatibility is resolved.

### Spark-TTS (scaffolding)

Spark-TTS (`SparkAudio/Spark-TTS-0.5B`) is installed and the model is loaded via Unsloth's `FastModel` wrapper. It is available in the environment but not wired into the synthesis loop in the current version of the notebook.

---

## Checkpointing

The synthesis loop writes a `checkpoint.json` file after every successfully synthesized sample. On restart, already-processed IDs are skipped. This matters in practice: Colab sessions disconnect, and re-running the synthesis from scratch wastes quota and time.

Each result is also appended line-by-line to a `manifest.jsonl` file containing the prompt ID, original text, audio path, voice used, and status. This manifest is the artifact that downstream training scripts consume.

---

## Output Structure

```
Olimi AI/
    audio/
        p_0001.mp3
        p_0002.mp3
        ...
    prompts/
        manifest.jsonl       # one JSON object per line, synthesis results
        checkpoint.json      # set of completed prompt IDs
    normalized_data.xlsx     # normalized text before synthesis
```

---

## Environment

The notebook is designed to run in Google Colab with Google Drive mounted. Drive is used for both model storage and output persistence across sessions.

Key dependencies:

- `unsloth` — model loading wrapper for Spark-TTS
- `edge-tts` — Microsoft neural TTS client
- `transformers`, `sentencepiece` — standard Hugging Face stack
- `peft`, `trl`, `bitsandbytes` — fine-tuning stack (present for Spark-TTS fine-tuning experiments)
- `omegaconf`, `einx`, `torchcodec` — Spark-TTS runtime requirements

---

## Known Issues and Notes

- The NAMAA model dependency conflict (`chatterbox-tts` package version) needs resolution before that backend can be used. It is the higher-priority backend given its Egyptian dialect specialization.
- The current synthesis run is capped at the first 100 samples (`prompts = prompts[:100]`). Remove or increase this limit for full dataset generation.
- A 300ms sleep is inserted between synthesis calls to avoid rate limiting from the Edge TTS service. For large batches, consider increasing this or implementing exponential backoff on failures.
- The `normalize_marks` function (expanding `?` and `!` to their spoken Arabic equivalents) is commented out. It was found to produce unnatural output in context and needs further work before enabling.
- The Arabic letter unification step was iterated on during development. An earlier version unified `أ/إ/آ → ا` and `ة → ه`, but this was found to degrade audio quality. The current version takes the opposite approach for `ة` and removes only diacritics.
