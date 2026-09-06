# Detektimi i lajmeve të rreme: BERT vs LLaMA 3

Punim diplome — FIEK, Universiteti i Prishtinës.
Krahasim eksperimental midis modelit transformer **BERT** (i regjur / fine-tuned)
dhe modelit të madh gjuhësor **LLaMA 3.2 3B Instruct** (i përdorur me prompting,
pa fine-tuning) për detektimin binar FAKE/REAL të deklaratave në datasetin
**LIAR**.

## Sistemet e krahasuara

| Sistemi | Lloji | Trajnim mbi LIAR |
|---|---|---|
| BERT baseline | `bert-base-uncased`, fine-tuned, 3 epoka | po |
| BERT V2 | i njëjti, fine-tuned me *class weights* | po |
| LLaMA 3.2 3B zero-shot | `meta-llama/llama-3.2-3b-instruct` (OpenRouter) | jo (prompting) |
| LLaMA 3.2 3B few-shot | i njëjti model, 6 shembuj në kontekst | jo (prompting) |

Të gjitha vlerësohen mbi të njëjtin test set LIAR (1 267 deklarata: 818 FAKE / 449 REAL).

## Struktura e projektit

```
data/
  raw/liar_dataset/         # LIAR origjinal (.tsv) — i gitignoruar
  processed/                # datasete binare + tri-klasëshe, tokenizimi BERT — i gitignoruar
bert_scripts/               # përgatitja, tokenizimi, regjimi dhe vlerësimi i BERT
  prepare_bert_dataset.py, analyze_tokenization.py, evaluate_bert.py,
  analyze_bert_errors.py, analyze_bert_original_labels.py, test_bert.py
prepare_dataset.py          # përpunimi paraprak i LIAR
check_duplicates.py         # duplikatet / deklaratat konfliktuoze
label_analysis.py           # analiza e etiketave + hartimi binar
train_bert.py               # regjimi i BERT baseline
train_bert_v2.py            # regjimi i BERT V2 (class weights)
evaluate_bert_v2.py         # vlerësimi i BERT V2
validate_processed_dataset.py
llama_openrouter_zeroshot_full.py   # LLaMA zero-shot mbi test set-in e plotë
llama_fewshot_full.py               # LLaMA few-shot mbi test set-in e plotë
test_llama_zeroshot_small.py, test_llama_zeroshot_100.py,
test_llama_fewshot_100.py, test_openrouter.py   # eksperimente paraprake / smoke tests
analysis/
  build_comparison.py       # bashkon parashikimet e 4 modeleve → results/comparison/
notebooks/
  final_comparison_analysis.ipynb   # analiza krahasuese e konsoliduar (metrika, figura, McNemar)
results/
  bert/                     # metrika, matrica, analizë gabimesh/confidence për BERT
  llama/                    # parashikimet e LLaMA (zero-shot / few-shot, CSV)
  comparison/               # tabela, figura dhe parashikime të bashkuara për të 4 sistemet
models/                     # checkpoint-et e BERT — i gitignoruar
```

## Riprodhimi

Kërkohet Python 3 me paketat në `requirements.txt`.
Për thirrjet e LLaMA-s duhet një skedar `.env` me `OPEN_ROUTER_API_KEY=...`.

```bash
# BERT (nga bert_scripts/ dhe skriptet në rrënjë)
python prepare_dataset.py
python bert_scripts/prepare_bert_dataset.py
python train_bert.py
python train_bert_v2.py
python bert_scripts/evaluate_bert.py
python evaluate_bert_v2.py

# LLaMA (kërkon .env me OPEN_ROUTER_API_KEY)
python llama_openrouter_zeroshot_full.py
python llama_fewshot_full.py

# Analiza krahasuese (metrika + figura + McNemar → results/comparison/)
python analysis/build_comparison.py
python -m nbconvert --to notebook --execute --inplace notebooks/final_comparison_analysis.ipynb
```

## Rezultatet kryesore

| Model | Saktësia | Makro F1 | MCC | FAKE recall | REAL recall |
|---|---|---|---|---|---|
| Baseline klasë-shumicë | 0.646 | — | 0.00 | — | — |
| BERT baseline | 0.646 | 0.616 | 0.232 | 0.72 | 0.51 |
| BERT V2 | 0.630 | 0.618 | 0.253 | 0.62 | 0.64 |
| LLaMA 3.2 3B zero-shot | 0.462 | 0.443 | 0.159 | 0.22 | 0.91 |
| LLaMA 3.2 3B few-shot | 0.358 | 0.269 | 0.033 | 0.01 | 1.00 |

BERT-i i regjur e tejkalon LLaMA-n e nxitur në mënyrë statistikisht domethënëse
(McNemar p < 0.0001). Dallimi BERT baseline vs V2 nuk është domethënës (p = 0.21).
Few-shot prompting-u e degradoi LLaMA-n (kolaps në klasën REAL). Detajet e plota në
`notebooks/final_comparison_analysis.ipynb` dhe `results/comparison/`.
