# Preprocess data using PackTron

for i in {00000..00001}; do
	packtron-preprocess \
		--input datasets/c4-train.${i}-of-01024.json.gz \
		--output-prefix datasets/c4-${i} \
 		--tokenizer-model t5-base \
		--partitions 1 \
		--workers 97 \
		--append-eod
done