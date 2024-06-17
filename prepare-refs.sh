#!/bin/sh

GENCODE_VERSION=${GENCODE_VERSION:-46}

mkdir -p refs
cd refs

wget -N http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
pyfastx index hg19.fa.gz
wget -N http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
pyfastx index hg38.fa.gz

GENCODE_FILENAME="gencode.v${GENCODE_VERSION}lift37.basic.annotation.gtf.gz"
wget -N "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_VERSION}/GRCh37_mapping/${GENCODE_FILENAME}"
python3 ../scripts/create_db.py "$GENCODE_FILENAME"

GENCODE_FILENAME="gencode.v${GENCODE_VERSION}.basic.annotation.gtf.gz"
wget -N "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${GENCODE_VERSION}/${GENCODE_FILENAME}"
python3 ../scripts/create_db.py "$GENCODE_FILENAME"
