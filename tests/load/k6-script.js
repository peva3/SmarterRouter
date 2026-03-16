// Load testing with k6
// Item #52: Load testing suite using k6
//
// Usage:
//   k6 run --vus 100 --duration 5m k6-script.js
//
// Or with stages:
//   k6 run --stage 2m:100,5m:100,2m:0 k6-script.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const cacheHitRate = new Rate('cache_hits');
const responseTime = new Trend('response_time');
const errorCounter = new Counter('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'], // 95% of requests under 2s
    http_req_failed: ['rate<0.01'],     // Error rate under 1%
    'response_time': ['p(99)<3000'],    // 99th percentile under 3s
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:11436';

// Test data
const simplePrompts = [
  'What is 2+2?',
  'Hello',
  'Explain Python',
  'What is AI?',
  'Write hello world',
];

const complexPrompts = [
  'Write a Python function to implement quicksort with full documentation',
  'Explain the differences between REST and GraphQL APIs',
  'Generate a SQL query to find the top 10 customers by revenue',
  'Create a React component for a todo list with TypeScript',
];

export function setup() {
  // Health check before starting
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'health check status is 200': (r) => r.status === 200,
    'health check has models': (r) => r.json('ollama') !== undefined,
  });
  
  return { timestamp: Date.now() };
}

export default function () {
  const scenario = Math.random();
  
  if (scenario < 0.6) {
    // 60%: Simple chat completions
    simpleChat();
  } else if (scenario < 0.8) {
    // 20%: Complex chat completions
    complexChat();
  } else if (scenario < 0.9) {
    // 10%: List models
    listModels();
  } else if (scenario < 0.95) {
    // 5%: Health check
    healthCheck();
  } else {
    // 5%: Embeddings
    embeddings();
  }
  
  sleep(Math.random() * 3 + 1); // Sleep 1-4 seconds
}

function simpleChat() {
  const payload = JSON.stringify({
    model: '',
    messages: [
      { role: 'user', content: randomChoice(simplePrompts) }
    ],
    stream: false,
  });
  
  const res = http.post(
    `${BASE_URL}/v1/chat/completions`,
    payload,
    { 
      headers: { 'Content-Type': 'application/json' },
      timeout: '60s',
    }
  );
  
  responseTime.add(res.timings.duration);
  
  const success = check(res, {
    'simple chat status is 200': (r) => r.status === 200,
    'simple chat has content': (r) => r.json('choices') !== undefined,
  });
  
  if (!success) {
    errorCounter.add(1);
  }
}

function complexChat() {
  const payload = JSON.stringify({
    model: '',
    messages: [
      { role: 'user', content: randomChoice(complexPrompts) }
    ],
    stream: false,
  });
  
  const res = http.post(
    `${BASE_URL}/v1/chat/completions`,
    payload,
    { 
      headers: { 'Content-Type': 'application/json' },
      timeout: '120s',
    }
  );
  
  responseTime.add(res.timings.duration);
  
  const success = check(res, {
    'complex chat status is 200': (r) => r.status === 200,
    'complex chat has content': (r) => r.json('choices') !== undefined,
  });
  
  if (!success) {
    errorCounter.add(1);
  }
}

function listModels() {
  const res = http.get(`${BASE_URL}/v1/models`, { timeout: '10s' });
  
  responseTime.add(res.timings.duration);
  
  const success = check(res, {
    'list models status is 200': (r) => r.status === 200,
    'list models has data': (r) => r.json('data') !== undefined,
  });
  
  if (!success) {
    errorCounter.add(1);
  }
}

function healthCheck() {
  const res = http.get(`${BASE_URL}/health`, { timeout: '5s' });
  
  responseTime.add(res.timings.duration);
  
  check(res, {
    'health status is 200': (r) => r.status === 200,
  });
}

function embeddings() {
  const payload = JSON.stringify({
    input: 'Hello world',
    model: '',
  });
  
  const res = http.post(
    `${BASE_URL}/v1/embeddings`,
    payload,
    { 
      headers: { 'Content-Type': 'application/json' },
      timeout: '30s',
    }
  );
  
  responseTime.add(res.timings.duration);
  
  const success = check(res, {
    'embeddings status is 200': (r) => r.status === 200,
    'embeddings has data': (r) => r.json('data') !== undefined,
  });
  
  if (!success) {
    errorCounter.add(1);
  }
}

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function teardown(data) {
  console.log('Load test completed');
  console.log(`Started at: ${new Date(data.timestamp).toISOString()}`);
  console.log(`Ended at: ${new Date().toISOString()}`);
}
