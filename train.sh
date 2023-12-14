#!/bin/bash

# Define the list of datasets, attacks, and aggregators
datasets=("MNIST" "Fashion-MNIST" "CIFAR10")
attacks=("single_direction" "partial_single_direction")
aggregators=("average" "filterl2" "ex_noregret")

# Loop over each combination
for dataset in "${datasets[@]}"; do
  for attack in "${attacks[@]}"; do
    for aggregator in "${aggregators[@]}"; do
      # Run the simulation command
      python src/simulate.py --dataset "$dataset" --attack "$attack" --agg "$aggregator"
      
      # Add any additional logic or commands here if needed
    done
  done
done