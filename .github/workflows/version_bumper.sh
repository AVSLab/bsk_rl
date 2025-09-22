#!/bin/bash

regex='([0-9]+\.[0-9]+\.)([0-9]+)'

# Check if argument is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 <path-to-pyproject.toml>"
    exit 1
fi

file="$1"

# Check if file exists
if [[ ! -f "$file" ]]; then
    echo "Error: File $file not found!"
    exit 1
fi

# Extract the version line
version_line=$(grep '^version =' "$file")
if [[ $version_line =~ $regex ]]; then
    # Extract and increment the last number
    last_number=${BASH_REMATCH[2]}
    incremented_number=$((last_number + 1))
    updated_version=${BASH_REMATCH[1]}$incremented_number

    # Update the version in the pyproject.toml file
    sed -i "s/version = \"${BASH_REMATCH[1]}${last_number}\"/version = \"$updated_version\"/" "$file"

    echo "Version updated to $updated_version in $file"
else
    echo "Error: Version not found in $file or not in X.Y.Z format"
    exit 1
fi


# Expose the update versions to GitHub Actions
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "updated_version=$updated_version"
  } >> "$GITHUB_OUTPUT"
fi
