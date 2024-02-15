python ./predict_hunflair2.py --input ./annotations/raw/bioid_text.txt --output ./annotations/hunflair2/bioid.txt --entity_types species
# python ./predict_hunflair2.py --input ./annotations/raw/tmvar_v3_text.txt --output ./annotations/hunflair2/tmvar_v3.txt --entity_types gene
python ./predict_hunflair2.py --input ./annotations/raw/medmentions_text.txt --output ./annotations/hunflair2/medmentions.txt --entity_types disease chemical
