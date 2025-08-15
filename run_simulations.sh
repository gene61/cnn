
OBJECTS=( "Drone1" "Drone2" "Drone3" "No_Drone")

ALPHA_VALUES=($(seq 0.0010 0.00005 0.070 | awk '{printf "%.3f\n", $1}' | shuf))

# echo "Generated ALPHA_VALUES:"
printf "%s\n" "${ALPHA_VALUES[@]}"
printf "ALPHA_VALUES count: %d\n" "${#ALPHA_VALUES[@]}"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="simulation_results_$TIMESTAMP"
mkdir -p $OUTPUT_DIR

# Number of repetitions per alpha value
REPEATS=3

# Progress tracking
TOTAL_SIMS=$(( ${#OBJECTS[@]} * ${#ALPHA_VALUES[@]} * REPEATS ))
COMPLETED=0

# Run simulations for each object and alpha combination
for OBJECT in "${OBJECTS[@]}"; do
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        echo "Running simulations for $OBJECT with alpha=$ALPHA"
        
        # Run the simulation 10 times
        for ((i=1; i<=$REPEATS; i++)); do
            echo "Run $i of $REPEATS"
            
            # Run the simulation and capture output
            OUTPUT_FILE="$OUTPUT_DIR/${OBJECT}_alpha_${ALPHA}_run${i}.log"
            python3 diffraction_modell.py --object $OBJECT --alpha_0 $ALPHA > $OUTPUT_FILE 2>&1
            
            # Check if simulation succeeded
            if [ $? -eq 0 ]; then
                echo "Success: $OBJECT with alpha=$ALPHA run $i completed successfully"
            else
                echo "Error: $OBJECT with alpha=$ALPHA run $i failed"
            fi
            ((COMPLETED++))
            PERCENT=$(( COMPLETED * 100 / TOTAL_SIMS ))
            printf "Progress: %d%% (%d/%d)\n" $PERCENT $COMPLETED $TOTAL_SIMS
        done
    done
done

echo "All simulations completed. Results saved in $OUTPUT_DIR"

