#!/bin/bash
# This script runs the evaluation scripts for the different models, relying on openai evals library
# MODIFY THIS SCRIPT ACCORDINGLY WITH YOUR PATHS AND SIMILAR.

# IF USING A VIRTUAL ENVIRONMENT, ACTIVATE IT HERE
source ~/venv/<your_venv>/bin/activate

# SPECIFY THE PATH OF THE FOLDER WHERE THIS SCRIPT IS LOCATED
FOLDER_PATH=<your_path>

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
# create general log file:
mkdir -p ${FOLDER_PATH}/run_evals/logs/${TIMESTAMP}
GENERAL_OUT=${FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/general.out
touch $GENERAL_OUT

# load the API key
# OPEN THE .env FILE AND ADD THE API KEY!
source ${FOLDER_PATH}/.env  

#MODELS=( "gpt-4o-2024-08-06" "gpt-4o-2024-05-13" "gpt-4o-mini-2024-07-18" "gpt-4-turbo-2024-04-09" "gpt-4-0125-preview" "gpt-4-1106-preview" "gpt-4-0613" "gpt-4-0314" "gpt-3.5-turbo-0125" "gpt-3.5-turbo-1106" "gpt-3.5-turbo-0613" "gpt-3.5-turbo-0301" )
MODELS=( "gpt-3.5-turbo-0125" )

# SET THIS TO 1 TO SKIP EVALUATIONS IF THE FILE ALREADY EXISTS; SET TO 0 TO RUN ALL EVALUATIONS AND OVERWRITE EXISTING FILES
SKIP_IF_FILE_FOUND=1  

EVALS=( "neubaroco" "moral_permissibility" "causal_judgment" "commonsense_qa_2" )

# create empty log file:
EVALS_OUT=${FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/evals.out
EVALS_ERR=${FOLDER_PATH}/run_evals/logs/${TIMESTAMP}/evals.err
touch $EVALS_OUT
touch $EVALS_ERR

i=0
for eval in "${EVALS[@]}"; do
  i=$((i+1))
  for model in "${MODELS[@]}"; do

    if [ $SKIP_IF_FILE_FOUND -eq 1 ]; then
      if [ -f ${FOLDER_PATH}/../raw_results/$eval/$model.jsonl ]; then
        echo "Skipping eval $eval for model $model because file already exists" >> $GENERAL_OUT
        continue
      fi
    fi
    echo "Running eval $eval for model $model" >> $GENERAL_OUT
    oaieval $model $eval \
      --record_path=${FOLDER_PATH}/../raw_results/$eval/$model.jsonl \
      --registry_path ${FOLDER_PATH}/registry >> $EVALS_OUT 2>> $EVALS_ERR
  done
done
echo Run $i evals >> $GENERAL_OUT
