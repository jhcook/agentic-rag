# ADR-0001: Apple Silicon M2 Pro Deployment
Date: 2025-11-28
Status: Accepted

## Context
Target hardware is Apple Silicon M2 Pro with 10 cores and 16 GB RAM.
ARM architecture requires container images built for Apple Silicon.

## Decision
Adopt a services architecture.
Use ARM-compatible containers and cross-compile dependencies where necessary.

## Consequences
- Positive: Scalability, isolation, optimized for Apple Silicon performance.
- Negative: Increased complexity in CI/CD and multi-arch builds.
- Trade-off: Higher setup cost, but future-proof for ARM adoption.

## Alternatives considered
- Monolith: simpler, but underutilizes hardware and limits scalability.
- Hybrid: rejected due to added operational overhead.

## Links
- .agent/policies/architecture-policy.yaml
- .agent/qa/ci-matrix.yaml
