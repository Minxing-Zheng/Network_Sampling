#!/bin/bash

model_type="BA ER WS SBM Graphon"
density_type="low mid high"
stat_type="degree clustering community eigenvalue"

for a in $model_type;
  do
  for b in $density_type;
    do
      for c in $stat_type;
        do
          for seed in {1..10};
            do
              echo "$a,$b,$c,$seed"
      done;
    done;
  done;
done;
