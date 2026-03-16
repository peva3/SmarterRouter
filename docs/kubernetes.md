# Kubernetes Deployment Guide

This guide covers deploying SmarterRouter to Kubernetes using Helm charts or raw manifests.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start (Helm)](#quick-start-helm)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [GPU Support](#gpu-support)
- [Scaling](#scaling)
- [Monitoring](#monitoring)

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.x (for Helm deployment)
- GPU nodes (optional, for GPU-accelerated models)

## Quick Start (Helm)

### 1. Add the Helm repository

```bash
helm repo add smarterrouter https://charts.smarterrouter.io
helm repo update
```

### 2. Install with default values

```bash
helm install smarterrouter smarterrouter/smarterrouter \
  --namespace smarterrouter \
  --create-namespace
```

### 3. Configure with custom values

Create `values.yaml`:

```yaml
# Backend configuration
backend:
  ollama:
    enabled: true
    url: "http://ollama:11434"

# API Keys (use Kubernetes secrets in production)
secrets:
  adminApiKey: "your-admin-key-here"
  openaiApiKey: ""  # Set via external secret

# GPU support
gpu:
  enabled: false
  vendor: nvidia  # nvidia, amd, intel

# Resource limits
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# Persistence for database
persistence:
  enabled: true
  size: 10Gi
  storageClass: "standard"

# Ingress configuration
ingress:
  enabled: true
  host: smarterrouter.example.com
  tls:
    enabled: true
    secretName: smarterrouter-tls

# Autoscaling
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

Install with custom values:

```bash
helm install smarterrouter smarterrouter/smarterrouter \
  -f values.yaml \
  --namespace smarterrouter
```

## Manual Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: smarterrouter
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: smarterrouter-config
  namespace: smarterrouter
data:
  ROUTER_OLLAMA_URL: "http://ollama:11434"
  ROUTER_PORT: "11434"
  ROUTER_LOG_LEVEL: "INFO"
  ROUTER_BENCHMARK_SOURCES: "huggingface,lmsys"
  ROUTER_SIGNATURE_ENABLED: "true"
  ROUTER_ENABLE_PROMPT_CACHING: "true"
  ROUTER_ENABLE_RESPONSE_CACHING: "true"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: smarterrouter-secrets
  namespace: smarterrouter
type: Opaque
stringData:
  ROUTER_ADMIN_API_KEY: "your-admin-key-here"
  ROUTER_OPENAI_API_KEY: ""
  ROUTER_ANTHROPIC_API_KEY: ""
  ROUTER_ENCRYPTION_KEY: "your-32-char-encryption-key"
```

### PersistentVolumeClaim

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: smarterrouter-data
  namespace: smarterrouter
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smarterrouter
  namespace: smarterrouter
  labels:
    app: smarterrouter
spec:
  replicas: 2
  selector:
    matchLabels:
      app: smarterrouter
  template:
    metadata:
      labels:
        app: smarterrouter
    spec:
      containers:
        - name: smarterrouter
          image: smarterrouter/smarterrouter:2.2.0
          ports:
            - containerPort: 11434
              name: http
          envFrom:
            - configMapRef:
                name: smarterrouter-config
            - secretRef:
                name: smarterrouter-secrets
          volumeMounts:
            - name: data
              mountPath: /app/data
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
          livenessProbe:
            httpGet:
              path: /health
              port: 11434
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 11434
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: smarterrouter-data
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: smarterrouter
  namespace: smarterrouter
spec:
  selector:
    app: smarterrouter
  ports:
    - port: 11434
      targetPort: 11434
      name: http
  type: ClusterIP
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: smarterrouter
  namespace: smarterrouter
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - smarterrouter.example.com
      secretName: smarterrouter-tls
  rules:
    - host: smarterrouter.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: smarterrouter
                port:
                  number: 11434
```

### HorizontalPodAutoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: smarterrouter
  namespace: smarterrouter
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: smarterrouter
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## Configuration

### Environment Variables from ConfigMap/Secret

All `ROUTER_*` environment variables can be set via ConfigMap (non-sensitive) or Secret (sensitive). See the [Configuration Reference](configuration.md) for all options.

### Common Configurations

#### External LLM Backend

```yaml
# ConfigMap
data:
  ROUTER_OLLAMA_URL: "http://ollama-service.ollama-ns.svc.cluster.local:11434"
```

#### Redis Cache (Distributed)

```yaml
# ConfigMap
data:
  ROUTER_CACHE_BACKEND: "redis"
  ROUTER_REDIS_URL: "redis://redis-master.redis.svc.cluster.local:6379"
```

#### Rate Limiting

```yaml
# ConfigMap
data:
  ROUTER_RATE_LIMIT_ENABLED: "true"
  ROUTER_RATE_LIMIT_REQUESTS_PER_MINUTE: "100"
  ROUTER_RATE_LIMIT_CHAT_REQUESTS_PER_MINUTE: "60"
```

## GPU Support

### NVIDIA GPU

Enable GPU support by adding the NVIDIA device plugin:

```yaml
spec:
  containers:
    - name: smarterrouter
      resources:
        limits:
          nvidia.com/gpu: 1
```

### AMD GPU

```yaml
spec:
  containers:
    - name: smarterrouter
      resources:
        limits:
          amd.com/gpu: 1
```

### Intel GPU

```yaml
spec:
  containers:
    - name: smarterrouter
      resources:
        limits:
          gpu.intel.com/i915: 1
```

### Node Selector for GPU Nodes

```yaml
spec:
  nodeSelector:
    accelerator: nvidia-tesla-t4
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

## Scaling

### Horizontal Pod Autoscaling

The HPA automatically scales based on CPU/memory usage:

```bash
# View current replicas
kubectl get hpa smarterrouter -n smarterrouter

# View metrics
kubectl get hpa smarterrouter -n smarterrouter -w
```

### Vertical Pod Autoscaling (VPA)

For automatic resource tuning:

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: smarterrouter
  namespace: smarterrouter
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: smarterrouter
  updatePolicy:
    updateMode: "Auto"
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: smarterrouter
  namespace: smarterrouter
  labels:
    release: prometheus
spec:
  selector:
    matchLabels:
      app: smarterrouter
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

### Grafana Dashboard

Import dashboard ID `smarterrouter-1` or use the provided ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: smarterrouter-dashboard
  namespace: smarterrouter
  labels:
    grafana_dashboard: "1"
data:
  smarterrouter.json: |
    {
      "dashboard": {
        "title": "SmarterRouter Metrics",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(smarterrouter_requests_total[5m])"
              }
            ]
          },
          {
            "title": "Cache Hit Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "smarterrouter_cache_hits_total / (smarterrouter_cache_hits_total + smarterrouter_cache_misses_total)"
              }
            ]
          },
          {
            "title": "GPU VRAM Usage",
            "type": "graph",
            "targets": [
              {
                "expr": "smarterrouter_gpu_vram_used_bytes"
              }
            ]
          }
        ]
      }
    }
```

## Troubleshooting

### Pod not starting

```bash
# Check pod status
kubectl get pods -n smarterrouter

# View logs
kubectl logs -n smarterrouter deployment/smarterrouter

# Check events
kubectl get events -n smarterrouter --sort-by='.lastTimestamp'
```

### Database issues

```bash
# Check PVC
kubectl get pvc -n smarterrouter

# Check PV
kubectl get pv

# Exec into pod to check database
kubectl exec -it -n smarterrouter deployment/smarterrouter -- ls -la /app/data/
```

### GPU not detected

```bash
# Check node labels
kubectl get nodes -o yaml | grep -A5 "nvidia.com/gpu"

# Check GPU plugin
kubectl get pods -n kube-system | grep nvidia

# Check pod resources
kubectl describe pod -n smarterrouter -l app=smarterrouter
```

## Cleanup

### Helm

```bash
helm uninstall smarterrouter -n smarterrouter
kubectl delete namespace smarterrouter
```

### Manual

```bash
kubectl delete -f manifests/
kubectl delete namespace smarterrouter
```
