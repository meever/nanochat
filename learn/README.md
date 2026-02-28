# nanochat — The Book

A self-contained learning book that explains every component of the nanochat LLM training codebase, from tokenization through RLHF.

## Reading

Open **[book.html](book.html)** in any browser. No server needed — it's a single self-contained file with KaTeX math and Mermaid diagrams loaded from CDN.

## Chapters

| # | Title | Key topic |
|---|-------|-----------|
| 1 | The Big Picture | Architecture overview, pipeline, scaling philosophy |
| 2 | Transformer Basics | Self-attention, causal mask, multi-head, MLP |
| 3 | The GPT Model | RoPE, GQA, value embeddings, sliding windows |
| 4 | Tokenizer | BPE, special tokens, conversation rendering |
| 5 | Dataset and Dataloader | Parquet shards, BOS-aligned packing |
| 6 | Training Loop | Scaling laws, LR schedule, mixed precision |
| 7 | Optimizer | AdamW, Muon, Newton-Schulz, parameter groups |
| 8 | Inference Engine | KV cache, sampling, tool-use state machine |
| 9 | Evaluation | BPB, CORE, ChatCORE, benchmarks |
| 10 | Chat SFT and RL | SFT data mixture, REINFORCE, DAPO |
| 11 | Running and Modifying | CLI reference, multi-GPU, troubleshooting |
| 12 | End-to-End Walkthrough | Trace a token through the entire system |

## Building

Rebuild the book after editing chapters:

```powershell
pwsh learn/build-book.ps1
```

Chapters live in `learn/chapters/` as markdown files (`NN-slug.md`). The build script assembles them into `learn/book.html`.
