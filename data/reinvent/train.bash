#!/bin/bash

names=('ESR1' 'HCRTR1' 'JAK1' 'P2RX3' 'KDM1A' 'IDH1' 'RIOK1' 'NR4A1' 'GRIK1' 'FTO' 'SPIN1' 'CCR9')

for name in "${names[@]}"
do
    mkdir "$name"
    cd "$name" || exit 1
    reinvent "../toml/${name}_train.toml"
    reinvent "../toml/${name}_sample.toml"
    cd ..
done