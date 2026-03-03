# Benchmark Data Sources

SmarterRouter aggregates model performance data from multiple independent sources to make intelligent routing decisions. This document explains each source, what data they provide, and how to configure them.

## Table of Contents
- [Overview](#overview)
- [HuggingFace Open LLM Leaderboard](#huggingface-open-llm-leaderboard)
- [LMSYS Chatbot Arena](#lmsys-chatbot-arena)
- [ArtificialAnalysis](#artificialanalysis)
- [Multi-Source Merging](#multi-source-merging)
- [Troubleshooting](#troubleshooting)

---

## Overview

Benchmark data is used to score model capabilities in three categories:
- **Reasoning**: Logic, math, analysis (MMLU, GPQA, Math)
- **Coding**: Programming ability (HumanEval, LiveCodeBench)
- **General**: Balanced overall performance

SmarterRouter can fetch data from multiple sources. Enable them via `ROUTER_BENCHMARK_SOURCES` in your `.env`:

```env
ROUTER_BENCHMARK_SOURCES=huggingface,lmsys,artificial_analysis
```

**Order matters:** Sources are queried sequentially. If multiple sources provide data for the same model, later sources overwrite earlier ones (last write wins). This lets you prioritize your preferred sources by ordering.

---

## HuggingFace Open LLM Leaderboard

**Source:** HuggingFace Open LLM Leaderboard dataset  
**What it provides:**
- MMLU (Multi-discipline Multi-choice Language Understanding)
- HellaSwag (commonsense reasoning)
- Winogrande (pronoun disambiguation)
- TruthfulQA (truthfulness)
- MMLU-Pro (harder version of MMLU)

**Coverage:** Broad coverage of open-weight models (Llama, Mistral, Mixtral, Qwen, Gemma, etc.)

**Update frequency:** Daily

**Configuration:** No API key required (public dataset)

**Pros:**
- Completely free, no rate limits
- Wide model coverage
- Standardized testing methodology

**Cons:**
- Limited to models that submit to leaderboard
- Doesn't include proprietary models (OpenAI, Anthropic, etc.)
- Some benchmarks are older (may not reflect latest models)

**How it works:** SmarterRouter downloads the results from HuggingFace's datasets server and matches models using a combination of explicit mapping and fuzzy name matching.

---

## LMSYS Chatbot Arena

**Source:** LMSYS Chatbot Arena leaderboard  
**What it provides:**
- ELO ratings from crowd-sourced human voting
- MT-Bench scores (multi-turn conversation quality)
- Individual benchmark scores (GGP, math, etc.)

**Coverage:** Popular models including OpenAI, Anthropic, Google, and top open models

**Update frequency:** Weekly

**Configuration:** No API key required (public data)

**Pros:**
- Real-world user preference data
- Includes proprietary models
- ELO ratings reflect overall chat quality

**Cons:**
- Less granular than standardized benchmarks
- Human voting introduces noise
- Smaller model coverage than HuggingFace

**How it works:** Fetches the latest leaderboard data and merges with other sources.

---

## ArtificialAnalysis

**Source:** ArtificialAnalysis.ai independent evaluations  
**What it provides:**
- Proprietary **Intelligence Index** (0-100) - overall capability
- **Coding Index** (0-100) - programming proficiency
- **Math Index** (0-100) - mathematical reasoning
- Standard benchmarks: MMLU-Pro, GPQA, LiveCodeBench, Math-500
- Real-world **speed metrics**: tokens/sec, time-to-first-token
- **Pricing information** (for cloud APIs)
- Model metadata: creator, release date, context window

**Coverage:** Major cloud providers (OpenAI, Anthropic, Google) and some open models

**Update frequency:** Daily

**Configuration:** Requires free API key (1,000 requests/day)

```env
# Get your key from https://artificialanalysis.ai/insights
ROUTER_ARTIFICIAL_ANALYSIS_API_KEY=your-key-here
ROUTER_ARTIFICIAL_ANALYSIS_CACHE_TTL=86400  # 24h cache (respect rate limits)
```

**Optional:** Model mapping file to handle naming differences (see `artificial_analysis_models.example.yaml`)

**Pros:**
- Comprehensive: benchmarks + speed + pricing in one source
- Proprietary indices provide alternative scoring
- Real-world throughput metrics help with speed-based routing
- Covers models other sources miss

**Cons:**
- API key required (free tier limited)
- Model naming may not match your local Ollama tags
- Smaller model coverage than HuggingFace

**How it works:**
1. Fetches all models from ArtificialAnalysis API
2. Maps AA model identifiers to your Ollama model names (using mapping file or auto-generation)
3. Converts AA scores to standard benchmark format
4. Stores AA-specific indices in `extra_data` column for future use
5. Speed metrics stored in `throughput` field

**Data mapping:**
- `artificial_analysis_intelligence_index` → stored in `extra_data` (0-100)
- `artificial_analysis_coding_index` → stored in `extra_data` (0-100)
- `artificial_analysis_math_index` → stored in `extra_data` (0-100)
- `mmlu_pro` → standard benchmark column `mmlu`
- `livecodebench` → standard benchmark column `humaneval` (coding proxy)
- `math_500` → standard benchmark column `math`
- `gpqa` → standard benchmark column `gpqa`
- `median_output_tokens_per_second` → standard benchmark column `throughput`

---

## Multi-Source Merging

When you enable multiple sources, SmarterRouter merges the data intelligently:

1. Each source fetches its data independently
2. Data is keyed by `ollama_name` (your Ollama model tag)
3. For each model, all non-null fields are collected
4. **Last source wins** for conflicting fields
5. Model's scored capability (`reasoning_score`, `coding_score`, `general_score`) are recalculated from the merged benchmark data

**Example:** With `ROUTER_BENCHMARK_SOURCES=huggingface,artificial_analysis`
- HuggingFace provides MMLU, HellaSwag, etc. for `llama3:70b`
- ArtificialAnalysis provides MMLU-Pro, LiveCodeBench, speed metrics for `openai/gpt-4o`
- Later source (AA) overwrites any overlapping fields (like `mmlu` if both provide it)
- Final benchmark row contains the best/merged data from both sources

**Best practice:** Order sources by priority:
```env
# Prefer HuggingFace for open models, but fill gaps with ArtificialAnalysis
ROUTER_BENCHMARK_SOURCES=huggingface,lmsys,artificial_analysis
```

Or prioritize ArtificialAnalysis for its speed data:
```env
ROUTER_BENCHMARK_SOURCES=huggingface,artificial_analysis
```

---

## Troubleshooting

### "ArtificialAnalysis provider returned 0 models"
- Check your API key is set correctly: `echo $ROUTER_ARTIFICIAL_ANALYSIS_API_KEY`
- Verify you haven't exceeded the 1,000 requests/day limit
- Check logs for mapping errors - your local model names may not match AA's naming
- Create a model mapping file to explicitly connect your models

### "No benchmark data for my model"
- Some models aren't in public leaderboards
- Use `ROUTER_ARTIFICIAL_ANALYSIS_MODEL_MAPPING_FILE` to map your model tag to an equivalent AA model
- Example: Your `llama3.1:70b` might map to AA's "Llama-3.1-70B"

### "Rate limit exceeded"
- Free tier is 1,000 requests/day
- Cache TTL defaults to 24h (86,400 seconds) to minimize re-fetching
- Increase cache TTL to reduce frequency: `ROUTER_ARTIFICIAL_ANALYSIS_CACHE_TTL=172800` (48h)
- Consider upgrading to a paid plan for higher limits

### "Model mapped to wrong Ollama name"
- Check your mapping file (if used)
- Enable debug logging: `ROUTER_LOG_LEVEL=DEBUG` to see mapping attempts
- Add explicit mapping entry for the problematic model

---

## Advanced: Direct Database Queries

You can query benchmark data directly to see which sources contributed:

```sql
-- See all benchmarks for a model
SELECT * FROM model_benchmarks WHERE ollama_name = 'llama3:70b';

-- Check which sources have data (implicit from which fields are non-NULL)
SELECT 
  ollama_name,
  mmlu IS NOT NULL AS has_huggingface,
  elo_rating IS NOT NULL AS has_lmsys,
  extra_data IS NOT NULL AS has_artificial_analysis,
  throughput IS NOT NULL AS has_speed_data
FROM model_benchmarks;
```

The `extra_data` JSON column contains all ArtificialAnalysis-specific fields that don't fit the standard schema.

---

## Performance Impact

Benchmark sync runs:
- On router startup (if database is empty or `force=true`)
- Periodically via the polling interval (controlled separately)
- When manually triggered via `/admin/reprofile`

Each source typically adds <1 second to startup (cached). Full resync from all sources takes 2-5 seconds. ArtificialAnalysis may take longer due to API rate limiting.

**Tip:** The first sync after enabling a new source may take longer as it fetches all available models. Subsequent syncs are incremental (only new/updated models).

---

## Future Sources

We're evaluating additional benchmark sources:
- **OpenCompass** - Comprehensive Chinese/English leaderboard
- **MT-Bench** - Multi-turn conversation benchmarks
- **Custom internal benchmarks** - Enterprise users may want to add proprietary evaluations

架构 allows easy addition of new `BenchmarkProvider` implementations. If you have a source you'd like to see supported, [open an issue](https://github.com/peva3/SmarterRouter/issues).
