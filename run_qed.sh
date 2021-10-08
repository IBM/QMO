#!/bin/bash
seqs=()
while read -r smiles; do
    seqs+=("$smiles")
done < data/qed_test.csv

if [[ ! -e logs/qed ]]
then
    mkdir -p logs/qed
fi
for s in "${seqs[@]}"
do
    python QMO/run.py -t=20 -k=50 -w=0.25 --beta=10 -q=50 --base_lr=0.2 --adam \
    --flip-weight -s=123456789 --early-stop=0.9 --sim=0.4 --score=qed $s \
    > logs/qed/${s}.log
done
