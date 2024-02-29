from collections import defaultdict
from datetime import datetime
import json
import os
import pandas as pd
import re
import traceback
import time

from pkg_resources import resource_filename
from pangolin.model import torch, Pangolin, L, W, AR
from pangolin.pangolin import process_variant as process_variant_using_pangolin
import gffutils

import psycopg2

# flask imports
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from flask_talisman import Talisman

app = Flask(__name__)

CORS(app)

DEBUG = False # if socket.gethostname() == "spliceai-lookup" else True
if not DEBUG:
    Talisman(app)


HG19_FASTA_PATH = "/hg19.fa.gz"
HG38_FASTA_PATH = "/hg38.fa.gz"

GENCODE_VERSION = "v44"
PANGOLIN_GRCH37_ANNOTATIONS = f"/gencode.{GENCODE_VERSION}lift37.basic.annotation.without_chr_prefix.db"
PANGOLIN_GRCH38_ANNOTATIONS = f"/gencode.{GENCODE_VERSION}.basic.annotation.db"
TRANSCRIPT_GRCH37_ANNOTATIONS = f"/gencode.{GENCODE_VERSION}lift37.basic.annotation.transcript_annotations.json"
TRANSCRIPT_GRCH38_ANNOTATIONS = f"/gencode.{GENCODE_VERSION}.basic.annotation.transcript_annotations.json"

PANGOLIN_DEFAULT_DISTANCE = 500  # maximum distance between the variant and gained/lost splice site, defaults to 500
PANGOLIN_MAX_DISTANCE_LIMIT = 10000
PANGOLIN_DEFAULT_MASK = 0        # mask scores representing annotated acceptor/donor gain and unannotated acceptor/donor loss, defaults to 0
PANGOLIN_EXAMPLE = f"/pangolin/?hg=38&distance=500&mask=0&variant=chr8-140300615-C-G"


PANGOLIN_MODELS = []
TRANSCRIPT_ANNOTATIONS = {}

VARIANT_RE = re.compile(
    "(chr)?(?P<chrom>[0-9XYMTt]{1,2})"
    "[-\s:]+"
    "(?P<pos>[0-9]{1,9})"
    "[-\s:]+"
    "(?P<ref>[ACGT]+)"
    "[-\s:>]+"
    "(?P<alt>[ACGT]+)"
)


def init_pangolin():
    if PANGOLIN_MODELS:
        return

    torch.set_num_threads(os.cpu_count()*2)


    for i in 0, 2, 4, 6:
        for j in 1, 2, 3:
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename("pangolin", "models/final.%s.%s.3.v2" % (j, i)))
            else:
                weights = torch.load(resource_filename("pangolin", "models/final.%s.%s.3.v2" % (j, i)), map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            PANGOLIN_MODELS.append(model)


def error_response(error_message, source="Pangolin"):
    response_json = {"error": str(error_message)}
    if source:
        response_json["source"] = source
    return Response(json.dumps(response_json), status=200, mimetype='application/json')


def parse_variant(variant_str):
    match = VARIANT_RE.match(variant_str)
    if not match:
        raise ValueError(f"Unable to parse variant: {variant_str}")

    return match['chrom'], int(match['pos']), match['ref'], match['alt']


DATABASE_CONNECTION = None
def init_database_connection():
    global DATABASE_CONNECTION
    try:
        DATABASE_CONNECTION = psycopg2.connect(
            host="/cloudsql/spliceai-lookup-412920:us-central1:spliceai-lookup-db",
            database="spliceai-lookup-db",
            user="postgres",
            password=os.environ.get("DB_PASSWORD"))
        DATABASE_CONNECTION.autocommit = True
    except Exception as e:
        print(f"ERROR: Unable to connect to the database: {e}")
        traceback.print_exc()
        return

    def does_table_exist(table_name):
        results = run_sql(f"SELECT EXISTS (SELECT 1 AS result FROM pg_tables WHERE tablename = '{table_name}');")
        does_table_already_exist = results[0][0]
        return does_table_already_exist

    if not does_table_exist("cache"):
        print("Creating cache table")
        run_sql("""CREATE TABLE cache (key TEXT UNIQUE, value TEXT, counter INT, accessed TIMESTAMP DEFAULT now())""")
        run_sql("""CREATE INDEX cache_index ON cache (key)""")

    if not does_table_exist("log"):
        print("Creating event_log table")
        run_sql("""CREATE TABLE log (event_name TEXT, ip TEXT, logtime TIMESTAMP DEFAULT now(), duration REAL, variant TEXT, genome VARCHAR(10), distance INT, mask INT4, details TEXT)""")
        run_sql("""CREATE INDEX log_index1 ON log (event_name)""")
        run_sql("""CREATE INDEX log_index2 ON log (ip)""")


def run_sql(sql_query, *args):
    with DATABASE_CONNECTION:
        c = DATABASE_CONNECTION.cursor()
        c.execute(sql_query, *args)
        try:
            results = c.fetchall()
        except:
            results = []
        c.close()
    return results


def get_splicing_scores_cache_key(variant, genome_version, distance, mask):
    return f"pangolin__{variant}__hg{genome_version}__d{distance}__m{mask}"


def get_splicing_scores_from_cache(variant, genome_version, distance, mask):
    if DATABASE_CONNECTION is None:
        return

    key = get_splicing_scores_cache_key(variant, genome_version, distance, mask)
    results = None
    try:
        rows = run_sql(f"SELECT value FROM cache WHERE key = '{key}'")
        if rows:
            results = json.loads(rows[0][0])
            results["source"] += ":cache"
    except Exception as e:
        print(f"Cache error: {e}", flush=True)

    return results


def add_splicing_scores_to_cache(variant, genome_version, distance, mask, results):
    if DATABASE_CONNECTION is None:
        return

    key = get_splicing_scores_cache_key(variant, genome_version, distance, mask)
    try:
        results_string = json.dumps(results)

        run_sql(r"""INSERT INTO cache (key, value, counter, accessed) VALUES (%s, %s, 1, now()) """ +
                r"""ON CONFLICT (key) DO """ +
                r"""UPDATE SET key=%s, value=%s, counter=cache.counter+1, accessed=now()""", (key, results_string, key, results_string))
    except Exception as e:
        print(f"Cache error: {e}", flush=True)


def get_pangolin_scores(variant, genome_version, distance_param, mask_param):
    if genome_version not in ("37", "38"):
        raise ValueError(f"Invalid genome_version: {mask_param}")

    if mask_param not in ("True", "False"):
        raise ValueError(f"Invalid mask_param: {mask_param}")

    try:
        chrom, pos, ref, alt = parse_variant(variant)
    except ValueError as e:
        print(f"ERROR while parsing variant {variant}: {e}")
        traceback.print_exc()

        return {
            "variant": variant,
            "source": "pangolin",
            "error": f"ERROR: {e}",
        }

    if len(ref) > 1 and len(alt) > 1:
        return {
            "variant": variant,
            "source": "pangolin",
            "error": f"ERROR: Pangolin does not currently support complex InDels like {chrom}-{pos}-{ref}-{alt}",
        }

    class PangolinArgs:
        reference_file = HG19_FASTA_PATH if genome_version == "37" else HG38_FASTA_PATH
        distance = distance_param
        mask = mask_param
        score_cutoff = None
        score_exons = "False"

    if genome_version == "37":
        pangolin_gene_db = gffutils.FeatureDB(PANGOLIN_GRCH37_ANNOTATIONS)
    else:
        pangolin_gene_db = gffutils.FeatureDB(PANGOLIN_GRCH38_ANNOTATIONS)

    if genome_version not in TRANSCRIPT_ANNOTATIONS:
        with open(TRANSCRIPT_GRCH37_ANNOTATIONS if genome_version == "37" else TRANSCRIPT_GRCH38_ANNOTATIONS, "rt") as ta_f:
            TRANSCRIPT_ANNOTATIONS[genome_version] = json.load(ta_f)

    scores = process_variant_using_pangolin(
        0, chrom, int(pos), ref, alt, pangolin_gene_db, PANGOLIN_MODELS, PangolinArgs)

    if not scores:
        return {
            "variant": variant,
            "source": "pangolin",
            "error": f"ERROR: Pangolin was unable to compute scores for this variant",
        }

    # to reduce the response size, return all non-zero scores only for the canonial transcript (or the 1st transcript)
    all_non_zero_scores = None
    all_non_zero_scores_strand = None
    all_non_zero_scores_transcript_id = None
    max_delta_score_sum = 0
    for i, transcript_scores in enumerate(scores):
        if "ALL_NON_ZERO_SCORES" not in transcript_scores:
            continue

        transcript_id_without_version = transcript_scores.get("NAME", "").split(".")[0]

        # get json annotations for this transcript
        transcript_annotations = TRANSCRIPT_ANNOTATIONS[genome_version].get(transcript_id_without_version)
        if transcript_annotations is None:
            raise ValueError(f"Missing annotations for {transcript_id_without_version} in {genome_version} annotations")

        # add the extra transcript annotations from the json file to the transcript scores dict
        transcript_scores.update(transcript_annotations)

        # decide whether to use ALL_NON_ZERO_SCORES from this gene
        delta_score_sum = sum(abs(float(s.get("SG_ALT", 0)) - float(s.get("SG_REF", 0)))
                              for s in transcript_scores["ALL_NON_ZERO_SCORES"])
        delta_score_sum += sum(abs(float(s.get("SL_ALT", 0)) - float(s.get("SL_REF", 0)))
                               for s in transcript_scores["ALL_NON_ZERO_SCORES"])

        # return all_non_zero_scores for the transcript or gene with the highest delta score sum
        if delta_score_sum > max_delta_score_sum:
            all_non_zero_scores = transcript_scores["ALL_NON_ZERO_SCORES"]
            all_non_zero_scores_strand = transcript_scores["STRAND"]
            all_non_zero_scores_transcript_id = transcript_scores["NAME"]
            max_delta_score_sum = delta_score_sum

        for redundant_key in "NAME", "STRAND", "ALL_NON_ZERO_SCORES":
            del transcript_scores[redundant_key]

    return {
        "variant": variant,
        "genomeVersion": genome_version,
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "distance": distance_param,
        "scores": scores,
        "source": "pangolin",
        "allNonZeroScores": all_non_zero_scores,
        "allNonZeroScoresStrand": all_non_zero_scores_strand,
        "allNonZeroScoresTranscriptId": all_non_zero_scores_transcript_id,
    }

@app.route("/pangolin/", methods=['POST', 'GET'])
def run_pangolin():
    """Handles Pangolin API request"""
    start_time = datetime.now()
    logging_prefix = start_time.strftime("%m/%d/%Y %H:%M:%S") + f" t{os.getpid()}"

    # check params
    params = {}
    if request.values:
        params.update(request.values)

    if 'variant' not in params:
        params.update(request.get_json(force=True, silent=True) or {})

    variant = params.get('variant', '')
    variant = variant.strip().strip("'").strip('"').strip(",")
    if not variant:
        return error_response(f'"variant" not specified.\n')

    if not isinstance(variant, str):
        return error_response(f'"variant" value must be a string rather than a {type(variant)}.\n')

    genome_version = params.get("hg")
    if not genome_version:
        return error_response(f'"hg" not specified. The URL must include an "hg" arg: hg=37 or hg=38. For example: {PANGOLIN_EXAMPLE}\n')

    if genome_version not in ("37", "38"):
        return error_response(f'Invalid "hg" value: "{genome_version}". The value must be either "37" or "38". For example: {PANGOLIN_EXAMPLE}\n')

    distance_param = params.get("distance", PANGOLIN_DEFAULT_DISTANCE)
    try:
        distance_param = int(distance_param)
    except Exception as e:
        return error_response(f'Invalid "distance": "{distance_param}". The value must be an integer.\n')

    if distance_param > PANGOLIN_MAX_DISTANCE_LIMIT:
        return error_response(f'Invalid "distance": "{distance_param}". The value must be < {PANGOLIN_MAX_DISTANCE_LIMIT}.\n')

    mask_param = params.get("mask", str(PANGOLIN_DEFAULT_MASK))
    if mask_param not in ("0", "1"):
        return error_response(f'Invalid "mask" value: "{mask_param}". The value must be either "0" or "1". For example: {PANGOLIN_EXAMPLE}\n')

    print(f"{logging_prefix}: {request.remote_addr}: ======================", flush=True)
    print(f"{logging_prefix}: {request.remote_addr}: {variant} processing with hg={genome_version}, "
          f"distance={distance_param}, mask={mask_param}", flush=True)

    init_database_connection()
    init_pangolin()

    # check cache before processing the variant
    results = get_splicing_scores_from_cache(variant, genome_version, distance_param, mask_param)
    duration = (datetime.now() - start_time).total_seconds()
    if results:
        log("pangolin:from-cache", ip=get_user_ip(request), variant=variant, genome=genome_version, distance=distance_param, mask=mask_param)
    else:
        pangolin_mask_param = "True" if mask_param == "1" else "False"
        results = get_pangolin_scores(variant, genome_version, distance_param, pangolin_mask_param)

        duration = (datetime.now() - start_time).total_seconds()
        log("pangolin:computed", ip=get_user_ip(request), duration=duration, variant=variant, genome=genome_version, distance=distance_param, mask=mask_param)

        if "error" not in results:
            add_splicing_scores_to_cache(variant, genome_version, distance_param, mask_param, results)

    if "error" in results:
        log("pangolin:error", ip=get_user_ip(request), variant=variant, genome=genome_version, distance=distance_param, mask=mask_param, details=results["error"])

    response_json = {}
    response_json.update(params)  # copy input params to output
    response_json.update(results)

    print(f"{logging_prefix}: {request.remote_addr}: {variant} took {str(datetime.now() - start_time)}", flush=True)

    return Response(json.dumps(response_json), status=200, mimetype='application/json')


def log(event_name, ip=None, duration=None, variant=None, genome=None, distance=None, mask=None, details=None):
    """Utility method for logging an event"""

    try:
        if duration is not None: duration = float(duration)
        if distance is not None: distance = int(distance)
        if mask is not None: mask = int(mask)
    except Exception as e:
        print(f"Error parsing log params: {e}", flush=True)
        return

    try:
        run_sql(r"INSERT INTO log (event_name, ip, duration, variant, genome, distance, mask, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (event_name, ip, duration, variant, genome, distance, mask, details))
    except Exception as e:
        print(f"Log error: {e}", flush=True)

def get_user_ip(request):
    return request.environ.get("HTTP_X_FORWARDED_FOR")

@app.route('/log/<string:name>/', strict_slashes=False)
def log_event(name):
    if DATABASE_CONNECTION is None:
        message = f"Log error: Database not available"
        print(message, flush=True)
        return message

    if name != "show_igv":
        message = f"Log error: invalid event name: {name}"
        print(message, flush=True)
        return message

    details = request.values.get("details")
    if details:
        details = str(details)
        details = details[:2000]

    log(name, ip=get_user_ip(request), details=details)

    return "Done"

@app.route('/', strict_slashes=False, defaults={'path': ''})
@app.route('/<path:path>/')
def catch_all(path):
    return "pangolin api"

app.run(debug=DEBUG, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
