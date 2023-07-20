#!/bin/bash

INFILE="$1"
OUTFILE="$2"


# Check if the input file exists
if [[ ! -f "$INFILE" ]]; then
    echo "Error: File not found!"
    exit 1
fi

# Function to join elements of an array into a string
function join_by { local IFS="$1"; shift; echo "$*"; }

# Create an array to store sentences
sentences=("tokens;lemmas;pos;mwe_tag;mwe_parent_offset;mwe_strength;sup_label")
tokens=()
lemmas=()
pos=()
mwe_tag=()
mwe_parent_offset=()
mwe_strength=()
sup_label=()


# Read the file line by line
while IFS=$'\t' read -r -a fields; do
    # if at end of line
    if [[ ${#fields[@]} -eq 0 ]]; then
        # Fill missing elements with 0
        # Add the current sentence to the sentences array
        tokens=("[$(join_by ',' "${tokens[@]}")]")
        lemmas=("[$(join_by ',' "${lemmas[@]}")]")
        pos=("[$(join_by ',' "${pos[@]}")]")
        mwe_tag=("[$(join_by ',' "${mwe_tag[@]}")]")
        mwe_parent_offset=("[$(join_by ',' "${mwe_parent_offset[@]}")]")
        mwe_strength=("[$(join_by ',' "${mwe_strength[@]}")]")
        sup_label=("[$(join_by ',' "${sup_label[@]}")]")
        joined=("$tokens;$lemmas;$pos;$mwe_tag;$mwe_parent_offset;$mwe_strength;$sup_label")
        sentences+=($joined)

        tokens=()
        lemmas=()
        pos=()
        mwe_tag=()
        mwe_parent_offset=()
        mwe_strength=()
        sup_label=()

    else
        # Extract relevant fields and add them to the current sentence
        tokens+=("${fields[1]}")
        lemmas+=("${fields[2]}")
        pos+=("${fields[3]}")
        mwe_tag+=("${fields[4]}")
        mwe_parent_offset+=("${fields[5]}")
        mwe_strength+=("${fields[6]}")
        sup_label+=("${fields[7]}")
    fi
done < "$INFILE"

# Add the last sentence to the sentences array if there is any
if [[ ${#tokens[@]} -gt 0 ]]; then
    tokens=("[$(join_by ',' "${tokens[@]}")]")
    lemmas=("[$(join_by ',' "${lemmas[@]}")]")
    pos=("[$(join_by ',' "${pos[@]}")]")
    mwe_tag=("[$(join_by ',' "${mwe_tag[@]}")]")
    mwe_parent_offset=("[$(join_by ',' "${mwe_parent_offset[@]}")]")
    mwe_strength=("[$(join_by ',' "${mwe_strength[@]}")]")
    sup_label=("[$(join_by ',' "${sup_label[@]}")]")
    joined=("$tokens;$lemmas;$pos;$mwe_tag;$mwe_parent_offset;$mwe_strength;$sup_label")
    sentences+=($joined)
fi
    
# Write the sentences to the CSV file
printf '%s\n' "${sentences[@]}" > "$OUTFILE"

echo "Conversion completed. The output CSV file is: $OUTFILE"