#!/bin/bash

echo "Waiting for all 5 architectures to complete their benchmarks..."
while [ $(ls -1 plancraft/agents/benchmark/output/*.json 2>/dev/null | grep val.small | wc -l) -lt 5 ]; do
    sleep 10
done

echo "All 5 architectures completed! Generating benchmark report..."
python3 plancraft/agents/scripts/benchmark_runner.py
echo "Done!"
