#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
    python main.py \
    --policy "TD3" \
    --env "HalfCheetah-v3" \
    --seed $i \
    --save_model

    python main.py \
    --policy "TD3" \
    --env "Hopper-v3" \
    --seed $i \
    --save_model

    python main.py \
    --policy "TD3" \
    --env "Walker2d-v3" \
    --seed $i \
    --save_model

    python main.py \
    --policy "TD3" \
    --env "Ant-v3" \
    --seed $i \
    --save_model

done

