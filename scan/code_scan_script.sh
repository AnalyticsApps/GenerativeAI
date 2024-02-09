#!/bin/bash

# Get the input text from the command line argument
input_text="$1"

mkdir gen_code
echo "$input_text" > gen_code/generated.txt

package_details=$( cat gen_code/generated.txt | awk '/pip install/ {print $0}' )
echo "$package_details" > gen_code/requirements.txt

python_code=$(awk '/```python/{flag=1; next} /```/{flag=0} flag' <<< "$input_text")
echo "$python_code" > gen_code/code.py
sed -i '/pip install/d' gen_code/code.py

snyk auth > gen_code/auth.log
synk_gen_code=$( snyk code test gen_code/ )

echo "$synk_gen_code"
