# Research Benchmark: Copilot Evaluation Report
**Model**: gpt-5-mini
**Dataset**: Plancraft (val.small)
**Date**: 2026-03-05

## 1. Executive Summary
| Architecture | Success Rate | Avg Steps |
|--------------|--------------|-----------|
| Single | 73.6% | 6.5 |
| Independent | 71.8% | 8.0 |
| Centralized | 61.8% | 10.6 |
| Decentralized | 77.3% | 7.4 |
| Hybrid | 67.3% | 9.8 |

## Single Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate | Avg Steps |
|------------|-------|---------|------|-----------|
| Low (1-2) | 38 | 38 | 100.0% | 1.6 |
| Medium (3-8) | 30 | 26 | 86.7% | 6.3 |
| High (9+) | 42 | 17 | 40.5% | 10.8 |

### Top Failures
- `incorrect_stop`: 24
- `max_steps_reached`: 5

## Independent Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate | Avg Steps |
|------------|-------|---------|------|-----------|
| Low (1-2) | 38 | 38 | 100.0% | 1.6 |
| Medium (3-8) | 30 | 25 | 83.3% | 6.6 |
| High (9+) | 42 | 16 | 38.1% | 13.4 |

### Top Failures
- `incorrect_stop`: 21
- `max_steps_reached`: 10

## Centralized Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate | Avg Steps |
|------------|-------|---------|------|-----------|
| Low (1-2) | 38 | 32 | 84.2% | 1.8 |
| Medium (3-8) | 30 | 20 | 66.7% | 10.8 |
| High (9+) | 42 | 16 | 38.1% | 15.1 |

### Top Failures
- `incorrect_stop`: 27
- `max_steps_reached`: 15

## Decentralized Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate | Avg Steps |
|------------|-------|---------|------|-----------|
| Low (1-2) | 38 | 38 | 100.0% | 1.5 |
| Medium (3-8) | 30 | 27 | 90.0% | 6.0 |
| High (9+) | 42 | 20 | 47.6% | 11.7 |

### Top Failures
- `incorrect_stop`: 16
- `max_steps_reached`: 9

## Hybrid Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate | Avg Steps |
|------------|-------|---------|------|-----------|
| Low (1-2) | 38 | 38 | 100.0% | 2.9 |
| Medium (3-8) | 30 | 20 | 66.7% | 11.3 |
| High (9+) | 42 | 16 | 38.1% | 15.1 |

### Top Failures
- `incorrect_stop`: 24
- `max_steps_reached`: 12
