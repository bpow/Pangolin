import argparse
import os
import csv
import sys
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
from pangolin.model import L, W, AR, Pangolin
import gffutils
import numpy as np
import torch
import pysam

FLOAT_FORMAT = "0.2f"

# use a 0.1 threshold for ALL_NON_ZERO_SCORES because Pangolin's baseline probability seems to be ~0.05 for most
# positions, so 0.1 separates the unusually large scores
MIN_SCORE_THRESHOLD = 0.1

IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def refseq2ens(chrom):
    if not chrom.startswith('NC_'):
        return chrom
    chrom = int(float(chrom[3:]))
    if chrom == 23:
        return 'X'
    elif chrom == 24:
        return 'Y'
    return str(chrom)

def one_hot_encode(seq, strand):
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    if strand == '+':
        seq = np.asarray(list(map(int, list(seq))))
    elif strand == '-':
        seq = np.asarray(list(map(int, list(seq[::-1]))))
        seq = (5 - seq) % 5  # Reverse complement
    return IN_MAP[seq.astype('int8')]


def compute_score(ref_seq, alt_seq, strand, d, models):
    ref_seq = one_hot_encode(ref_seq, strand).T
    ref_seq = torch.from_numpy(np.expand_dims(ref_seq, axis=0)).float()
    alt_seq = one_hot_encode(alt_seq, strand).T
    alt_seq = torch.from_numpy(np.expand_dims(alt_seq, axis=0)).float()

    if torch.cuda.is_available():
        ref_seq = ref_seq.to(torch.device("cuda"))
        alt_seq = alt_seq.to(torch.device("cuda"))

    pangolin = []
    pangolin_ref = []
    pangolin_alt = []
    for j in range(4):
        score = []
        score_ref = []
        score_alt = []
        for model in models[3*j: 3*j+3]:
            with torch.no_grad():
                ref = model(ref_seq)[0][[1, 4, 7, 10][j], :].cpu().numpy()
                alt = model(alt_seq)[0][[1, 4, 7, 10][j], :].cpu().numpy()
                if strand == '-':
                    ref = ref[::-1]
                    alt = alt[::-1]

                l = 2*d+1
                ndiff = np.abs(len(ref)-len(alt))
                if len(ref) > len(alt):
                    alt = np.concatenate(
                        [alt[0:l//2+1], np.zeros(ndiff), alt[l//2+1:]])
                elif len(ref) < len(alt):
                    alt = np.concatenate(
                        [alt[0:l//2], np.max(alt[l//2:l//2+ndiff+1], keepdims=True), alt[l//2+ndiff+1:]])

                score.append(alt - ref)
                score_ref.append(ref)
                score_alt.append(alt)

        pangolin.append(np.mean(score, axis=0))
        pangolin_ref.append(np.mean(score_ref, axis=0))
        pangolin_alt.append(np.mean(score_alt, axis=0))

    pangolin = np.array(pangolin)
    pangolin_ref = np.array(pangolin_ref)
    pangolin_alt = np.array(pangolin_alt)
    pangolin_argmin = np.argmin(pangolin, axis=0)
    pangolin_argmax = np.argmax(pangolin, axis=0)
    pangolin_idx = np.arange(pangolin.shape[1])

    # Since pangolin computes max gain scores across 4 tissues and, separately, max loss scores across the 4 tissues,
    # the ref and alt probabilities for the gain scores can be from a different tissue than the splice loss probablities.
    # Therefore, we need to keep the ref and alt probabilities that underlie the gain score, and, separately, also the
    # ref and alt probabilities that underlie the loss score.
    # this is a 1d array that should == the difference between loss_ref and loss_alt
    loss = pangolin[pangolin_argmin, pangolin_idx]
    # at each position, select the ref sequence splice probability from the tissue that had the maximum loss score
    loss_ref = pangolin_ref[pangolin_argmin, pangolin_idx]
    # at each position, select the alt sequence splice probability from the tissue that had the maximum loss score
    loss_alt = pangolin_alt[pangolin_argmin, pangolin_idx]

    # basic internal consistency checks
    if len(loss) != len(loss_alt) or len(loss) != len(loss_ref):
        raise ValueError(f"len(loss) != len(loss_alt) or len(loss) != len(loss_ref): {
                         len(loss)} != {len(loss_alt)} or {len(loss)} != {len(loss_ref)}")
    if any(abs(l - (a - r)) > 1e-5 for l, a, r in zip(loss, loss_alt, loss_ref)):
        raise ValueError("Internal error: loss != loss_alt - loss_ref")

    # this is 1d array that should == the difference between gain_ref and gain_alt
    gain = pangolin[pangolin_argmax, pangolin_idx]
    gain_ref = pangolin_ref[pangolin_argmax, pangolin_idx]
    gain_alt = pangolin_alt[pangolin_argmax, pangolin_idx]
    # basic internal consistency checks
    if len(gain) != len(gain_alt) or len(gain) != len(gain_ref):
        raise ValueError(f"len(gain) != len(gain_alt) or len(gain) != len(gain_ref): {len(gain)} != {len(gain_alt)} or {len(gain)} != {len(gain_ref)}")
    if any(abs(g - (a - r)) > 1e-5 for g, a, r in zip(gain, gain_alt, gain_ref)):
        raise ValueError("Internal error: gain != gain_alt - gain_ref")

    return loss, gain, loss_ref, loss_alt, gain_ref, gain_alt


def get_genes(chr, pos, gtf):
    genes = gtf.region((chr, pos-1, pos-1), featuretype="transcript")
    genes_pos, genes_neg = {}, {}

    for gene in genes:
        if gene[3] > pos or gene[4] < pos:
            continue
        transcript_id = gene["transcript_id"][0]
        exons = []
        for exon in gtf.children(gene, featuretype="exon"):
            exons.extend([exon[3], exon[4]])
        if gene[6] == '+':
            genes_pos[transcript_id] = exons
        elif gene[6] == '-':
            genes_neg[transcript_id] = exons

    return genes_pos, genes_neg


def process_variant(lnum, chr, pos, ref, alt, gtf, models, args):
    d = args.distance

    if len(set("ACGT").intersection(set(ref))) == 0 or len(set("ACGT").intersection(set(alt))) == 0 \
            or (len(ref) != 1 and len(alt) != 1 and len(ref) != len(alt)):
        print("[Line %s]" %
              lnum, "WARNING, skipping variant: Variant format not supported.")
        return None
    elif len(ref) > 2*d:
        print("[Line %s]" %
              lnum, "WARNING, skipping variant: Deletion too large")
        return None

    fasta = pysam.FastaFile(args.reference_file)
    # try to make vcf chromosomes compatible with reference chromosomes
    if chr.startswith('NC_'):
        chr = refseq2ens(chr)
    if chr not in fasta.references and "chr"+chr in fasta.references:
        chr = "chr"+chr
    elif chr not in fasta.references and chr[3:] in fasta.references:
        chr = chr[3:]

    try:
        seq = fasta.fetch(chr, pos-5001-d, pos+len(ref)+4999+d)
    except Exception as e:
        print(e)
        print("[Line %s]" % lnum, "WARNING, skipping variant: Could not get sequence, possibly because the variant is too close to chromosome ends. "
                                  "See error message above.")
        return None

    if seq[5000+d:5000+d+len(ref)] != ref:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Mismatch between FASTA (ref base: %s) and variant file (ref base: %s)."
              % (seq[5000+d:5000+d+len(ref)], ref))
        return None

    ref_seq = seq
    alt_seq = seq[:5000+d] + alt + seq[5000+d+len(ref):]

    # get genes that intersect variant
    genes_pos, genes_neg = get_genes(chr, pos, gtf)
    if len(genes_pos) + len(genes_neg) == 0:
        print("[Line %s]" % lnum, "WARNING, skipping variant: Variant not contained in a gene body. Do GTF/FASTA chromosome names match?")
        return None

    # get splice scores
    genomic_coords = np.arange(pos-d, pos+d+len(ref))

    results = []
    for genes, strand in [(genes_pos, "+"), (genes_neg, "-")]:
        if not genes:
            continue

        orig_loss, orig_gain, loss_ref, loss_alt, gain_ref, gain_alt = compute_score(
            ref_seq, alt_seq, strand, d, models)

        for transcript_id, positions in genes.items():
            positions = np.array(positions)
            positions = positions - (pos - d)

            if args.mask != "True":
                loss = orig_loss
                gain = orig_gain
            else:
                # Make copies of the loss/gain for each gene to avoid overwriting data between genes
                loss = np.copy(orig_loss)
                gain = np.copy(orig_gain)

                if len(positions) != 0:
                    positions_filt = positions[(
                        positions >= 0) & (positions < len(loss))]
                    # set splice gain at annotated sites to 0
                    gain[positions_filt] = np.minimum(gain[positions_filt], 0)
                    # set splice loss at unannotated sites to 0
                    not_positions = ~np.isin(
                        np.arange(len(loss)), positions_filt)
                    loss[not_positions] = np.maximum(loss[not_positions], 0)

                else:
                    loss[:] = np.maximum(loss[:], 0)

            if len(genomic_coords) != len(gain):
                raise ValueError(f"Internal error: len(genomic_coords) != len(gain): {
                                 len(genomic_coords)} != {len(gain)}")
            if len(genomic_coords) != len(loss):
                raise ValueError(f"Internal error: len(genomic_coords) != len(loss): {
                                 len(genomic_coords)} != {len(loss)}")

            l, g = np.argmin(loss), np.argmax(gain)
            results.append({
                "NAME": transcript_id,
                # splice gain delta score at the position where the splice gain delta score is maximum
                "DS_SG": f"{gain[g]:{FLOAT_FORMAT}}",
                # splice loss delta score at the position where the splice loss delta score is maximum
                "DS_SL": f"{loss[l]:{FLOAT_FORMAT}}",
                # relative position where the splice gain delta score is maximum
                "DP_SG": int(g-d),
                # relative position where the splice loss delta score is maximum
                "DP_SL": int(l-d),
                # reference sequence splice probability at position and tissue where splice gain is maximum
                "SG_REF": f"{gain_ref[g]:{FLOAT_FORMAT}}",
                # alt sequence splice probability at position and tissue where splice gain is maximum
                "SG_ALT": f"{gain_alt[g]:{FLOAT_FORMAT}}",
                # reference sequence splice probability at position and tissue where splice loss is maximum
                "SL_REF": f"{loss_ref[l]:{FLOAT_FORMAT}}",
                # alt sequence splice probability at position and tissue where splice loss is maximum
                "SL_ALT": f"{loss_alt[l]:{FLOAT_FORMAT}}",
                "ALL_NON_ZERO_SCORES": [
                    {
                        "pos": int(genomic_coord),
                        # reference sequence splice probability in the tissue where the splice loss delta score is largest at this position
                        "SL_REF": f"{loss_ref_score:{FLOAT_FORMAT}}",
                        # alt sequence splice probability in the tissue where the splice loss delta score is largest at this position
                        "SL_ALT": f"{loss_alt_score:{FLOAT_FORMAT}}",
                        # reference sequence splice probability in the tissue where the splice gain delta score is largest at this position
                        "SG_REF": f"{gain_ref_score:{FLOAT_FORMAT}}",
                        # alt sequence splice probability in the tissue where the splice gain delta score is largest at this position
                        "SG_ALT": f"{gain_alt_score:{FLOAT_FORMAT}}",
                    } for i, (genomic_coord, loss_ref_score, loss_alt_score, gain_ref_score, gain_alt_score) in enumerate(zip(
                        genomic_coords, loss_ref, loss_alt, gain_ref, gain_alt)
                    ) if any(score >= MIN_SCORE_THRESHOLD for score in (
                        loss_ref_score, loss_alt_score, gain_ref_score, gain_alt_score)) or i in (l, g)
                ],
                "STRAND": strand,
            })

    return results


def convert_scores_to_string(scores):
    return ",".join([
        "|".join([s["NAME"], f"{s['DP_SG']}:{s['DS_SG']}", f"{s['DP_SL']}:{s['DS_SL']}"]) for s in scores
    ])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant_file", help="VCF or CSV file with a header (see COLUMN_IDS option).")
    parser.add_argument(
        "reference_file", help="FASTA file containing a reference genome sequence.")
    parser.add_argument(
        "annotation_file", help="gffutils database file. Can be generated using create_db.py.")
    parser.add_argument(
        "output_file", help="Prefix for output file. Will be a VCF/CSV if variant_file is VCF/CSV.")
    parser.add_argument("-c", "--column_ids", default="CHROM,POS,REF,ALT", help="(If variant_file is a CSV) Column IDs for: chromosome, variant position, reference bases, and alternative bases. "
                                                                                "Separate IDs by commas. (Default: CHROM,POS,REF,ALT)")
    parser.add_argument("-m", "--mask", default="True", choices=[
                        "False", "True"], help="If True, splice gains (increases in score) at annotated splice sites and splice losses (decreases in score) at unannotated splice sites will be set to 0. (Default: True)")
    parser.add_argument("-s", "--score_cutoff", type=float,
                        help="Output all sites with absolute predicted change in score >= cutoff, instead of only the maximum loss/gain sites.")
    parser.add_argument("-d", "--distance", type=int, default=50,
                        help="Number of bases on either side of the variant for which splice scores should be calculated. (Default: 50)")
    parser.add_argument("--score_exons", default="False", choices=[
                        "False", "True"], help="Output changes in score for both splice sites of annotated exons, as long as one splice site is within the considered range (specified by -d). Output will be: gene|site1_pos:score|site2_pos:score|...")
    args = parser.parse_args()

    variants = args.variant_file
    gtf = args.annotation_file
    try:
        gtf = gffutils.FeatureDB(gtf)
    except:
        print("ERROR, annotation_file could not be opened. Is it a gffutils database file?")
        exit()

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    models = []
    for i in [0, 2, 4, 6]:
        for j in range(1, 4):
            model = Pangolin(L, W, AR)
            resource_ref = importlib_resources.files(__name__) / f"models/final.{j}.{i}.3.v2"
            with importlib_resources.as_file(resource_ref) as resource_path:
                if torch.cuda.is_available():
                    model.cuda()
                    weights = torch.load(resource_path)
                else:
                    weights = torch.load(
                        resource_path, map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)

    if variants.endswith(".vcf"):
        lnum = 0
        # count the number of header lines
        for line in open(variants, 'r'):
            lnum += 1
            if line[0] != '#':
                break

        with pysam.VariantFile(variants) as variant_file, pysam.VariantFile(
            args.output_file+".vcf", "w", header=variant_file.header
        ) as out_variant_file:
            out_variant_file.header.add_meta(
                key="INFO",
                items=[
                    ("ID", "Pangolin"),
                    ("Number", "."),
                    ("Type", "String"),
                    (
                        "Description",
                        "Pangolin splice scores. Format: gene|pos:score_change|pos:score_change|warnings,..."
                    ),
                ]
            )
            for i, variant_record in enumerate(variant_file):
                variant_record.translate(out_variant_file.header)
                assert variant_record.ref, f"Empty REF field in variant record {variant_record}"
                assert variant_record.alts, f"Empty ALT field in variant record {variant_record}"
                scores = process_variant(
                    lnum + i,
                    str(variant_record.contig),
                    int(variant_record.pos),
                    variant_record.ref,
                    str(variant_record.alts[0]),
                    gtf,
                    models,
                    args,
                )
                if scores is not None:
                    variant_record.info["Pangolin"] = convert_scores_to_string(scores)
                out_variant_file.write(variant_record)

    elif variants.endswith(".csv"):
        col_ids = args.column_ids.split(',')
        with open(variants, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            with open(args.output_file+".csv", 'w') as fout:
                writer = csv.DictWriter(fout, fieldnames=reader.fieldnames+["Pangolin"], lineterminator=os.linesep)

                writer.writeheader()
                for lnum, variant in enumerate(reader):
                    chr, pos, ref, alt = (variant[x] for x in col_ids)
                    ref, alt = ref.upper(), alt.upper()
                    scores = process_variant(
                        lnum+1, str(chr), int(pos), ref, alt, gtf, models, args)
                    if scores:
                        variant['Pangolin'] = convert_scores_to_string(scores)
                    writer.writerow(variant)

    else:
        print("ERROR, variant_file needs to be a CSV or VCF.")


if __name__ == '__main__':
    main()
