#!/bin/bash

# Terrain-aware comparison
echo "Generating terrain-aware comparison..."
python plot_three_methods.py \
    --proposed new_results/eval_dump_trained_localization_model_64beacons_1000instances_fixed_power13.npz \
    --plain new_results/eval_dump_trained_localization_model_64beacons_1000instances_fixed_no_rssi2dist.npz \
    --mlat ../multilateration/new_results/eval_dump_mlat_64beacons_100instances_64N_16A.npz \
    --outdir ../../thesis/images \
    --tag ""

# Free-space comparison
echo "Generating free-space comparison..."
python plot_three_methods.py \
    --proposed new_results/eval_dump_trained_localization_model_64beacons_1000instances_fixed_power13_free.npz \
    --plain new_results/eval_dump_trained_localization_model_64beacons_1000instances_fixed_no_rssi2dist_free.npz \
    --mlat ../multilateration/new_results/eval_dump_mlat_64beacons_1000instances_free_64N_16A.npz \
    --outdir ../../thesis/images \
    --tag free

echo "Done!"
