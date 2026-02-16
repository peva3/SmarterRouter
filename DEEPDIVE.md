# LLM Router Proxy: Architectural Deep Dive

This document provides a technical overview of the LLM Router Proxy, detailing the design philosophy, component interactions, and the rationale behind specific architectural choices.

---

## 1. The Core Philosophy

The primary goal of this project is to solve the "Paradox of Choice" in local LLM deployments. As the number of available models grows (Mistral, Llama, Qwen, DeepSeek, etc.), users often find it difficult to know which model is best for a specific prompt. 

Most users default to their largest model, which is slow, or their fastest model, which might be too simple for complex tasks. This router acts as an intelligent middleware that makes that decision automatically, balancing capability, speed, and resource constraints.

---

## 2. Component Architecture

### 2.1 Backend Abstraction Layer (`router/backends/`)
We didn't want to build a tool that only works with Ollama. The backend layer uses a Protocol-based abstraction (similar to a Java Interface) to ensure that the core routing logic is decoupled from the specific LLM engine.

- **Ollama Backend**: The primary target for local users. Supports VRAM management and model unloading.
- **llama.cpp Backend**: Designed for high-performance deployments using the standard `llama.cpp` server.
- **OpenAI-Compatible Backend**: Allows the router to act as a bridge to external APIs (OpenAI, Anthropic, or even other instances of this router).

**Why this matters:** It future-proofs the system. If a new high-performance engine emerges tomorrow, we only need to implement one Python class to support it.

### 2.2 The Routing Engine (`router/router.py`)
The "Brain" of the system. It handles the scoring and selection process using a multi-weighted algorithm.

- **Query Difficulty Prediction**: Before choosing a model, the router analyzes the prompt. It looks for logic indicators, code structures, and instruction density to decide if the task is "Easy" or "Hard."
- **Scoring Heuristics**: It combines three distinct data points:
    1. **Static Benchmarks**: External data from HuggingFace/LMSYS (how the model performs in general).
    2. **Runtime Profiles**: Local data from our profiler (how the model performs on *your* hardware).
    3. **Name Affinity**: Heuristic matching for specific tasks (e.g., routing `.py` requests to `*coder` models).
- **Quality vs. Speed Tuner**: The `ROUTER_QUALITY_PREFERENCE` setting acts as a global bias. A low value prioritizes throughput; a high value prioritizes benchmark scores and model size.

### 2.3 The Profiling Pipeline (`router/profiler.py` & `router/judge.py`)
Model evaluation is often subjective. We moved from a basic "did it respond?" check to a sophisticated evaluation pipeline.

- **Standardized Prompts**: We use a curated set of prompts inspired by **MT-Bench**. This ensures models are tested on reasoning, coding, and creativity in a consistent way.
- **LLM-as-Judge**: This is a critical feature for high-quality deployments. If enabled, the router uses a powerful model (the "Judge") to grade the responses of smaller models. 
- **Capability Detection**: The profiler doesn't just look at scores; it probes for Vision and Tool-Calling support, ensuring requests requiring these features aren't routed to models that will fail them.

**Rationale:** Local hardware varies wildly. A model that is fast on an A100 might be unusable on a laptop. Local profiling is the only way to get an accurate "Speed" score for your specific environment.

### 2.4 Resource & VRAM Management
Running multiple models locally is a VRAM nightmare. 

- **Proactive Unloading**: Before the router tells a backend to load a new model, it can issue an unload command for the current model. This prevents "Out of Memory" errors during model switching.
- **Pinned Models**: You can "pin" a small, efficient model (like Phi-3 mini) to VRAM. The router will prioritize this model for simple queries, providing near-instant responses for the 80% of tasks that don't need a massive model.

---

## 3. Data Flow: Anatomy of a Request

1. **Ingress**: A user sends an OpenAI-style `/v1/chat/completions` request to the router.
2. **Analysis**:
    - The router identifies if the request needs Vision or specific Tools.
    - The difficulty predictor tags the request as Easy, Medium, or Hard.
3. **Selection**:
    - The `RouterEngine` pulls all profiled models from the database.
    - It filters out models that lack required capabilities (e.g., Vision).
    - It calculates a weighted score for each remaining model.
    - The model with the highest score is selected.
4. **Execution**:
    - The router checks if the model is loaded.
    - If a different model is in VRAM, it triggers an unload.
    - It forwards the request to the backend.
5. **Egress**:
    - The response is streamed back to the user.
    - An optional signature is appended (e.g., "Model: deepseek-r1:7b").
6. **Feedback (Optional)**:
    - If the user provides a rating via `/v1/feedback`, that score is saved to the database and will influence that model's selection in the future.

---

## 4. Database & Storage (`router/models.py`)

We chose **SQLite** via **SQLAlchemy** for storage.

- **Why SQLite?** Zero configuration. It's a single file (`router.db`) that makes the router truly "plug-and-play."
- **Audit Logging**: Every routing decision and response time is logged. This allows for future "Post-Mortem" analysis to see if the router is making the right choices.
- **Schema**:
    - `ModelProfile`: Local performance data.
    - `ModelBenchmark`: External leaderboard data.
    - `ModelFeedback`: User ratings.
    - `BenchmarkSync`: Tracking when we last updated data from HuggingFace.

---

## 5. Security & Production Readiness

While often used locally, we've added features to make the router safe for multi-user environments:

- **Rate Limiting**: Protects your GPU from being overwhelmed by too many concurrent requests.
- **Admin Keys**: Protects sensitive endpoints like `/admin/reprofile` while keeping the main chat API accessible.
- **Sanitization**: All prompts are stripped of control characters and validated against length limits to prevent injection or memory-exhaustion attacks.
- **Cascading Fallbacks**: If the "best" model happens to be down or fails mid-generation, the router can automatically retry with the "second best" model, improving overall system reliability.
