# Architecture Overview

This document describes the SmarterRouter architecture and data flow.

## System Architecture

```mermaid
graph TB
    subgraph Client["Client Applications"]
        OWUI["OpenWebUI"]
        CLI["CLI Tools"]
        Other["Other Apps"]
    end

    subgraph Router["SmarterRouter"]
        subgraph API["API Layer"]
            CHAT[/v1/chat/completions\]
            MODELS[/v1/models\]
            HEALTH[/health\]
            ADMIN[/admin/*\]
        end

        subgraph Core["Core Services"]
            ROUTER["Router Engine"]
            PROFILER["Model Profiler"]
            CACHE["Semantic Cache"]
            SECURITY["Security Layer"]
        end

        subgraph Data["Data Layer"]
            SQLITE[(SQLite DB)]
            REDIS[(Redis Cache)]
            PROVIDER[(Provider DB)]
        end

        subgraph Backends["LLM Backends"]
            OLLAMA["Ollama"]
            LLAMACPP["llama.cpp"]
            OPENAI["OpenAI-compatible"]
            EXTERNAL["External Providers"]
        end

        subgraph GPU["GPU Monitoring"]
            NVIDIA["NVIDIA"]
            AMD["AMD"]
            INTEL["Intel"]
            APPLE["Apple Silicon"]
        end
    end

    OWUI --> CHAT
    CLI --> CHAT
    Other --> CHAT

    CHAT --> ROUTER
    CHAT --> SECURITY
    MODELS --> ROUTER
    HEALTH --> Core
    ADMIN --> Core

    ROUTER --> CACHE
    ROUTER --> PROFILER
    ROUTER --> Data
    ROUTER --> Backends

    PROFILER --> Backends
    PROFILER --> GPU

    SECURITY --> Data

    CACHE --> REDIS
    CACHE --> SQLITE

    Data --> PROVIDER

    style Router fill:#e1f5fe
    style Core fill:#fff3e0
    style Data fill:#e8f5e9
    style Backends fill:#fce4ec
    style GPU fill:#f3e5f5
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Middleware
    participant Security
    participant Cache as Semantic Cache
    participant Router as Router Engine
    participant Backend as LLM Backend
    participant DB as Database

    Client->>API: POST /v1/chat/completions
    API->>Middleware: Apply middleware
    Middleware->>Security: Rate limiting & validation
    Security->>Security: Prompt injection check
    Security->>Security: Content moderation

    alt Cache Hit
        Security->>Cache: Check semantic cache
        Cache->>Cache: Similarity search
        Cache-->>Security: Cached response
        Security-->>Client: Return cached response
    else Cache Miss
        Security->>Router: Route query
        Router->>DB: Load model profiles
        Router->>DB: Load benchmarks
        Router->>Router: Analyze prompt
        Router->>Router: Score models
        Router->>Router: Select best model

        Router->>Backend: Send request
        Backend->>Backend: Generate response
        Backend-->>Router: Response

        Router->>Cache: Store in cache
        Router-->>Security: Response
        Security-->>Client: Return response
    end
```

## Component Breakdown

### 1. API Layer

- **REST API**: FastAPI-based endpoints
- **OpenAI Compatibility**: Drop-in replacement for OpenAI API
- **Admin Endpoints**: Management and monitoring

### 2. Core Services

#### Router Engine

- Analyzes prompts for complexity, tools, vision requirements
- Scores models using benchmarks + profiles + feedback
- Applies penalties/bonuses for size, provider, diversity
- Selects optimal model for each request

#### Model Profiler

- Profiles models using test prompts
- Measures capabilities across categories (reasoning, coding, creativity)
- Estimates VRAM requirements
- Updates database with results

#### Semantic Cache

- Caches routing decisions and responses
- Uses cosine similarity for cache hits
- Adaptive thresholds based on hit rates
- Persistent storage with SQLite

#### Security Layer

- Rate limiting (per IP, per endpoint)
- Prompt injection detection
- Content moderation
- API key validation
- Admin IP whitelist

### 3. Data Layer

- **SQLite**: Primary database for profiles, benchmarks, feedback
- **Redis**: Optional distributed cache
- **Provider DB**: External benchmark data from HuggingFace, LMSYS, ArtificialAnalysis

### 4. LLM Backends

Abstracted interface supporting:
- **Ollama**: Local model management
- **llama.cpp**: High-performance inference
- **OpenAI-compatible**: Any OpenAI API-compatible service
- **External Providers**: OpenAI, Anthropic, Google, etc.

### 5. GPU Monitoring

Auto-detects GPU vendor and monitors:
- **NVIDIA**: nvidia-smi
- **AMD**: rocm-smi + sysfs
- **Intel**: sysfs (i915/xe drivers)
- **Apple Silicon**: unified memory

## Data Flow Detail

### Chat Completion Request

1. **Request Validation**
   - Parse JSON body
   - Validate model name (if specified)
   - Check rate limits

2. **Security Checks**
   - Prompt injection detection
   - Content moderation (optional)
   - Request size validation

3. **Caching**
   - Check semantic cache for similar prompts
   - Cache hit: return cached response
   - Cache miss: continue to routing

4. **Prompt Analysis**
   - Tokenize and analyze complexity
   - Detect vision requirements (images)
   - Detect tool requirements (functions)
   - Calculate complexity score

5. **Model Selection**
   - Load available models
   - Get benchmarks for each model
   - Get user feedback scores
   - Calculate combined scores
   - Apply penalties/bonuses
   - Select best model

6. **Backend Request**
   - Prepare request for selected backend
   - Apply circuit breaker / retry logic
   - Stream response back to client

7. **Post-Processing**
   - Append model signature (if enabled)
   - Cache response
   - Log routing decision

### Background Tasks

```mermaid
graph LR
    subgraph Tasks["Background Tasks"]
        SYNC[Benchmark Sync]
        POLL[Model Polling]
        CACHE_CLEAN[Cache Cleanup]
        DLQ[Dead Letter Queue]
    end

    SYNC -->|Every 4h| PROVIDER[(Provider DB)]
    POLL -->|Every 5m| OLLAMA[Ollama Backend]
    CACHE_CLEAN -->|Hourly| SQLITE[(SQLite)]
    DLQ -->|Retry| SYNC

    style Tasks fill:#fff3e0
```

## Configuration Flow

```mermaid
graph TD
    ENV[Environment Variables] -->|ROUTER_*| CONFIG[config.py]
    CONFIG --> SETTINGS[Settings Object]
    SETTINGS -->|Injected| APP[FastAPI App]
    SETTINGS -->|Injected| ROUTER[Router Engine]
    SETTINGS -->|Injected| CACHE[Cache]
    SETTINGS -->|Injected| BACKENDS[Backends]

    style ENV fill:#e8f5e9
    style SETTINGS fill:#e1f5fe
```

## Scoring Algorithm

```mermaid
graph TD
    PROMPT[Prompt Analysis] -->|1.0| COMPLEXITY{Complexity Score}
    PROMPT -->|0.2| CATEGORY{Category}

    COMPLEXITY -->|Penalties| SIZE[Size Penalties]
    COMPLEXITY -->|Bonuses| SIZEB[Size Bonuses]
    CATEGORY -->|Boost| CATB[Category Boost]

    BENCHMARKS[Benchmarks] -->|0.5| WEIGHTED[Weighted Score]
    PROFILES[Profiles] -->|0.3| WEIGHTED
    FEEDBACK[User Feedback] -->|0.2| WEIGHTED

    WEIGHTED --> COMBINED[Combined Score]
    SIZE --> COMBINED
    SIZEB --> COMBINED
    CATB --> COMBINED

    COMBINED --> PROVIDER[Provider Bonus]
    PROVIDER --> DIVERSITY[Diversity Penalty]
    DIVERSITY --> FINAL[Final Score]

    FINAL --> SELECT[Select Best Model]

    style FINAL fill:#fff3e0
    style SELECT fill:#e8f5e9
```

## Scalability Considerations

### Horizontal Scaling

- Stateless design enables multiple instances
- Redis backend for distributed caching
- Database connection pooling
- Load balancer distributes requests

### Performance Optimizations

- Async/await throughout
- Connection pooling for HTTP clients
- Batched database operations
- Vectorized similarity calculations (numpy)
- LRU caching for profiles and benchmarks

### Resource Management

- Automatic VRAM monitoring
- Circuit breakers prevent cascading failures
- Rate limiting protects resources
- Background tasks don't block requests

## Security Architecture

```mermaid
graph TB
    subgraph Security["Security Layers"]
        IP[IP Whitelist]
        RATE[Rate Limiting]
        AUTH[API Key Auth]
        INPUT[Input Validation]
        PROMPT[Prompt Injection]
        CONTENT[Content Moderation]
    end

    CLIENT[Client] --> IP
    IP --> RATE
    RATE --> AUTH
    AUTH --> INPUT
    INPUT --> PROMPT
    PROMPT --> CONTENT
    CONTENT --> APP[Application]

    style Security fill:#ffebee
```

## Deployment Options

### Docker Compose (Single Node)

```mermaid
graph LR
    subgraph Docker["Docker Compose"]
        ROUTER[SmarterRouter]
        OLLAMA[Ollama]
        REDIS[Redis]
    end

    CLIENT --> ROUTER
    ROUTER --> OLLAMA
    ROUTER --> REDIS
```

### Kubernetes (Production)

```mermaid
graph TB
    subgraph K8s["Kubernetes Cluster"]
        INGRESS[Ingress]
        ROUTER1[SmarterRouter Pod 1]
        ROUTER2[SmarterRouter Pod 2]
        ROUTER3[SmarterRouter Pod 3]
        SVC[Service]
        PVC[PersistentVolume]
    end

    CLIENT --> INGRESS
    INGRESS --> SVC
    SVC --> ROUTER1
    SVC --> ROUTER2
    SVC --> ROUTER3
    ROUTER1 --> PVC
    ROUTER2 --> PVC
    ROUTER3 --> PVC
```

See [kubernetes.md](kubernetes.md) for detailed Kubernetes deployment instructions.

## Monitoring & Observability

### Metrics

- Request rate, latency, errors (RED metrics)
- Cache hit/miss rates
- GPU VRAM utilization
- Model selection distribution
- Backend health status

### Logging

- Structured JSON logging
- Request correlation IDs
- Sanitized user input
- Error context enrichment

### Health Checks

```
GET /health
```

Returns:
- Database connectivity
- Backend status
- GPU status
- Cache status
- Background task counts
- DLQ counts (if enabled)
