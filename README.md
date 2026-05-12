This repository contains the scripts we used for direct in-context lemma generation with Mistral-Large-Instruct-2407.


## Features
- Few-shot prompt generation (1–5 shots)
- Wordform and sentence-based input formats

## Input format
The system expects tokenized sentences, one per line.  
Each word must be separated by whitespace or newline depending on configuration.

## Output format
The system outputs TSV files in the format:
word \t lemma
Each sentence is separated by a blank line.

## Configuration
Main parameters in `main.py`:

- `prompt_type`: basic_prompt or full_prompt, where:
    basic prompt is an instruction to perform lemmatization, without any further explanation;
    full prompt is bsic prompt + a set of explicit instructions
- `shots`: 1_shot to 5_shot
- `input_type`: wordform or sentence
- `corpus`: dataset name (e.g. BDT, PUD)

If you use the data in your work, please refer to our paper:
@inproceedings{toporkov-etal-2025-lemma,
    title = "Lemma Dilemma: On Lemma Generation Without Domain- or Language-Specific Training Data",
    author = "Toporkov, Olia  and
      Akbik, Alan  and
      Agerri, Rodrigo",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.988/",
    doi = "10.18653/v1/2025.findings-emnlp.988",
    pages = "18219--18232",
    ISBN = "979-8-89176-335-7",
}
