import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="TDS GA8 Deterministic Evaluation Service")

# -----------------------------------------------------------------------------
# Common Utility Functions
# -----------------------------------------------------------------------------

def compact_json(obj: Any) -> str:
    """Return compact JSON without extra whitespace, preserving non-ASCII characters directly."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def sha256_hex(data: Union[str, bytes]) -> str:
    """Compute lowercase SHA-256 hex digest of UTF-8 string or bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest().lower()

def parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Validate and parse ISO timestamp:
    YYYY-MM-DDTHH:mm:ss[.sss](Z|±HH:mm)
    Fraction has 1-3 digits.
    Offset magnitude <= 14:00; hour 14 requires minutes 00.
    """
    if not isinstance(ts_str, str):
        return None
    
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$"
    if not re.match(pattern, ts_str):
        return None
    
    if ts_str[-1] != "Z":
        offset_part = ts_str[-6:]
        try:
            oh = int(offset_part[1:3])
            om = int(offset_part[4:6])
        except ValueError:
            return None
        if om < 0 or om > 59:
            return None
        if oh > 14 or (oh == 14 and om != 0):
            return None

    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def format_utc_iso_ms(dt: datetime) -> str:
    """Format UTC datetime as YYYY-MM-DDTHH:mm:ss.sssZ."""
    dt_utc = dt.astimezone(timezone.utc)
    ms = int(round(dt_utc.microsecond / 1000.0))
    if ms >= 1000:
        dt_utc = dt_utc + timedelta(seconds=1)
        ms = 0
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"


# =============================================================================
# Q1: POST /build-corpus
# =============================================================================

CRC32C_POLYNOMIAL = 0x82F63B78
_crc32c_table = []
for i in range(256):
    crc = i
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ CRC32C_POLYNOMIAL
        else:
            crc >>= 1
    _crc32c_table.append(crc)

def compute_crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _crc32c_table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF

def crc32c_hex(data: bytes) -> str:
    val = compute_crc32c(data)
    return f"{val:08x}".lower()

def canonicalize_text(text: str) -> str:
    # Unicode NFKC, lowercase, trim, and collapse Unicode whitespace to one ASCII space
    norm = unicodedata.normalize("NFKC", text).lower().strip()
    tokens = re.split(r"\s+", norm, flags=re.UNICODE)
    return " ".join([t for t in tokens if t])

def get_word_set(text: str) -> set:
    # lowercase Unicode letter/number word-set
    norm = unicodedata.normalize("NFKC", text).lower()
    words = re.findall(r"[^\W_]+", norm, flags=re.UNICODE)
    return set(words)

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return inter / union if union > 0 else 0.0

@app.post("/build-corpus")
async def build_corpus(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "policy" not in body or "objects" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    policy = body.get("policy")
    objects = body.get("objects")
    
    if not isinstance(policy, dict) or not isinstance(objects, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    min_time_raw = policy.get("minTime")
    max_time_raw = policy.get("maxTime")
    contam_thresh = policy.get("contaminationThreshold")
    
    dt_min = parse_iso_timestamp(min_time_raw)
    dt_max = parse_iso_timestamp(max_time_raw)
    
    policy_valid = True
    if dt_min is None or dt_max is None or dt_min > dt_max:
        policy_valid = False
    if not isinstance(contam_thresh, (int, float)) or not math.isfinite(contam_thresh) or not (0.0 <= contam_thresh <= 1.0):
        policy_valid = False
    
    rejected_objects = []
    rejected_rows_map = {}
    accepted_rows = []
    lineage = []
    
    for obj_idx, obj in enumerate(objects):
        obj_codes = set()
        
        if not isinstance(obj, dict):
            rejected_objects.append({"uri": None, "reasonCodes": ["SCHEMA_INVALID"]})
            continue

        uri_val = obj.get("uri")
        uri_str = uri_val if isinstance(uri_val, str) else None
        
        # 1. URI check: gs://bucket/object (bucket cannot have /, object cannot be empty)
        if not isinstance(uri_val, str) or not re.match(r"^gs://[^/]+/.+$", uri_val):
            obj_codes.add("URI_INVALID")

        # 2. Generation check
        gen = obj.get("generation")
        f_gen = obj.get("fetchedGeneration")
        gen_is_dec = isinstance(gen, str) and bool(re.match(r"^\d+$", gen))
        f_gen_is_dec = isinstance(f_gen, str) and bool(re.match(r"^\d+$", f_gen))

        if not gen_is_dec or not f_gen_is_dec:
            obj_codes.add("GENERATION_INVALID")
        if gen != f_gen:
            obj_codes.add("GENERATION_MISMATCH")

        # 3. CRC32C check
        crc = obj.get("crc32c")
        crc_syntax_valid = isinstance(crc, str) and bool(re.match(r"^[0-9a-f]{8}$", crc))
        if not crc_syntax_valid:
            obj_codes.add("CRC32C_INVALID")

        content = obj.get("content")
        if isinstance(content, str) and crc_syntax_valid:
            expected_crc = crc32c_hex(content.encode("utf-8"))
            if crc != expected_crc:
                obj_codes.add("CRC32C_MISMATCH")

        # 4. Schema ID check
        schema_id = obj.get("schemaId")
        if schema_id != "training-v1":
            obj_codes.add("SCHEMA_INVALID")

        # 5. Content check
        parsed_rows = []
        if not isinstance(content, str):
            obj_codes.add("SCHEMA_INVALID")
        else:
            lines = content.splitlines()
            non_blank_lines = [line for line in lines if line.strip() != ""]
            if len(non_blank_lines) == 0:
                obj_codes.add("SCHEMA_INVALID")
            else:
                has_json_err = False
                has_schema_err = False
                for line in non_blank_lines:
                    try:
                        row = json.loads(line)
                    except Exception:
                        has_json_err = True
                        break
                    if not isinstance(row, dict):
                        has_schema_err = True
                        break
                    keys = set(row.keys())
                    if keys != {"id", "entity", "eventTime", "revision", "text"}:
                        has_schema_err = True
                        break
                    if not (isinstance(row["id"], str) and isinstance(row["entity"], str) and isinstance(row["eventTime"], str) and isinstance(row["text"], str)):
                        has_schema_err = True
                        break
                    rev = row["revision"]
                    if not (isinstance(rev, int) and not isinstance(rev, bool) and rev >= 0):
                        has_schema_err = True
                        break
                    dt_event = parse_iso_timestamp(row["eventTime"])
                    if dt_event is None:
                        has_schema_err = True
                        break
                    parsed_rows.append((row, dt_event))
                
                if has_json_err:
                    obj_codes.add("JSONL_INVALID")
                if has_schema_err:
                    obj_codes.add("SCHEMA_INVALID")

        if obj_codes:
            sorted_codes = sorted(list(obj_codes), key=lambda x: x.encode("utf-8"))
            rejected_objects.append({"uri": uri_str, "reasonCodes": sorted_codes})
        else:
            lineage.append({
                "uri": uri_str,
                "generation": gen,
                "crc32c": crc,
                "schemaId": schema_id
            })
            for row, dt_event in parsed_rows:
                can_entity = canonicalize_text(row["entity"])
                can_text = canonicalize_text(row["text"])
                can_time = format_utc_iso_ms(dt_event)
                accepted_rows.append({
                    "id": row["id"],
                    "entity": can_entity,
                    "eventTime": can_time,
                    "revision": row["revision"],
                    "text": can_text,
                    "dt_event": dt_event,
                    "raw_id": row["id"]
                })

    rejected_objects.sort(key=lambda o: ((o["uri"] or "").encode("utf-8"), compact_json(o)))
    lineage.sort(key=lambda l: (l["uri"].encode("utf-8"), compact_json(l)))

    dedup_groups = {}
    for r in accepted_rows:
        key = (r["entity"], r["eventTime"], r["text"])
        dedup_groups.setdefault(key, []).append(r)
        
    retained_rows = []
    for key, group in dedup_groups.items():
        if len(group) == 1:
            retained_rows.append(group[0])
        else:
            group_sorted = sorted(group, key=lambda x: (-x["revision"], x["id"].encode("utf-8")))
            winner = group_sorted[0]
            retained_rows.append(winner)
            for loser in group_sorted[1:]:
                rejected_rows_map.setdefault(loser["id"], set()).add("DUPLICATE")

    valid_retained_rows = []
    for r in retained_rows:
        r_codes = set()
        if not policy_valid:
            r_codes.add("POLICY_INVALID")
        else:
            if not (dt_min <= r["dt_event"] <= dt_max):
                r_codes.add("OUT_OF_WINDOW")
        
        if r_codes:
            rejected_rows_map.setdefault(r["id"], set()).update(r_codes)
        else:
            valid_retained_rows.append(r)

    train_rows = []
    val_rows = []
    test_rows = []
    
    for r in valid_retained_rows:
        entity_bytes = r["entity"].encode("utf-8")
        hash_first_byte = hashlib.sha256(entity_bytes).digest()[0]
        bucket = hash_first_byte % 10
        r_clean = {
            "id": r["id"],
            "entity": r["entity"],
            "eventTime": r["eventTime"],
            "revision": r["revision"],
            "text": r["text"]
        }
        if 0 <= bucket <= 5:
            train_rows.append(r_clean)
        elif 6 <= bucket <= 7:
            val_rows.append(r_clean)
        else:
            test_rows.append(r_clean)

    train_word_sets = [get_word_set(tr["text"]) for tr in train_rows]
    
    final_val_rows = []
    for vr in val_rows:
        v_words = get_word_set(vr["text"])
        is_contam = False
        for tw in train_word_sets:
            if jaccard_similarity(v_words, tw) >= contam_thresh:
                is_contam = True
                break
        if is_contam:
            rejected_rows_map.setdefault(vr["id"], set()).add("TRAIN_CONTAMINATION")
        else:
            final_val_rows.append(vr)

    final_test_rows = []
    for tr in test_rows:
        t_words = get_word_set(tr["text"])
        is_contam = False
        for tw in train_word_sets:
            if jaccard_similarity(t_words, tw) >= contam_thresh:
                is_contam = True
                break
        if is_contam:
            rejected_rows_map.setdefault(tr["id"], set()).add("TRAIN_CONTAMINATION")
        else:
            final_test_rows.append(tr)

    def sort_rows(rows_list):
        return sorted(rows_list, key=lambda x: (x["id"].encode("utf-8"), compact_json(x)))

    final_train_rows = sort_rows(train_rows)
    final_val_rows = sort_rows(final_val_rows)
    final_test_rows = sort_rows(final_test_rows)

    def compute_split_digest(rows_list):
        serialized_bytes = b"".join(
            (compact_json(r) + "\n").encode("utf-8") for r in rows_list
        )
        return hashlib.sha256(serialized_bytes).hexdigest().lower()

    digests = {
        "train": compute_split_digest(final_train_rows),
        "validation": compute_split_digest(final_val_rows),
        "test": compute_split_digest(final_test_rows)
    }

    rejected_rows_list = []
    for rid, codes in rejected_rows_map.items():
        sorted_codes = sorted(list(codes), key=lambda x: x.encode("utf-8"))
        rejected_rows_list.append({"id": rid, "reasonCodes": sorted_codes})
    rejected_rows_list.sort(key=lambda r: (r["id"].encode("utf-8"), compact_json(r)))

    return {
        "splits": {
            "train": final_train_rows,
            "validation": final_val_rows,
            "test": final_test_rows
        },
        "rejectedObjects": rejected_objects,
        "rejectedRows": rejected_rows_list,
        "digests": digests,
        "lineage": lineage
    }


# =============================================================================
# Q2: POST /bqml
# =============================================================================

_bqml_store: Dict[str, Dict[str, Any]] = {}

@app.post("/bqml")
async def bqml_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "phase" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    phase = body.get("phase")
    if phase == "select":
        run_id = body.get("runId")
        if not isinstance(run_id, str) or len(run_id) == 0 or len(run_id) > 128:
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        
        if run_id in _bqml_store:
            stored = _bqml_store[run_id]
            if stored["input"] == body:
                return stored["response"]
            else:
                return JSONResponse(status_code=409, content={"error": "RUN_ID_CONFLICT"})

        reason_codes = set()
        forbidden_features = body.get("forbiddenFeatures", [])
        num_trials_limit = body.get("numTrialsLimit")
        rows = body.get("rows")
        trials = body.get("trials")
        
        if not (isinstance(forbidden_features, list) and isinstance(num_trials_limit, int) and num_trials_limit > 0 and isinstance(rows, list) and len(rows) > 0 and isinstance(trials, list)):
            reason_codes.add("INVALID_INPUT")
        
        if isinstance(trials, list) and isinstance(num_trials_limit, int) and len(trials) > num_trials_limit:
            reason_codes.add("TRIAL_LIMIT_EXCEEDED")
            
        row_ids_set = set()
        parsed_rows = []
        if isinstance(rows, list):
            for r in rows:
                if not isinstance(r, dict):
                    reason_codes.add("INVALID_INPUT")
                    break
                r_id = r.get("id")
                if not isinstance(r_id, str) or r_id in row_ids_set:
                    reason_codes.add("INVALID_INPUT")
                row_ids_set.add(r_id)
                v = r.get("version")
                if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
                    reason_codes.add("INVALID_INPUT")
                dt_event = parse_iso_timestamp(r.get("eventTime"))
                dt_pred = parse_iso_timestamp(r.get("predictionTime"))
                if dt_event is None or dt_pred is None:
                    reason_codes.add("INVALID_INPUT")
                split = r.get("split")
                if split not in ["TRAIN", "EVAL"]:
                    reason_codes.add("INVALID_INPUT")
                parsed_rows.append({
                    "id": r_id,
                    "entity": r.get("entity"),
                    "eventTimeUtc": format_utc_iso_ms(dt_event) if dt_event else "",
                    "predictionTimeDt": dt_pred,
                    "version": v,
                    "split": split,
                    "features": r.get("features", {})
                })

        dedup_map = {}
        for r in parsed_rows:
            key = (r["entity"], r["eventTimeUtc"])
            dedup_map.setdefault(key, []).append(r)
        
        retained_rows = []
        for key, group in dedup_map.items():
            winner = sorted(group, key=lambda x: (-x["version"], x["id"].encode("utf-8")))[0]
            retained_rows.append(winner)

        forbidden_set = set(forbidden_features) if isinstance(forbidden_features, list) else set()
        all_features = set()
        for r in retained_rows:
            all_features.update(r["features"].keys())
        
        eligible_features = []
        for f_name in all_features:
            if f_name in forbidden_set:
                continue
            is_valid_f = True
            for r in retained_rows:
                if f_name not in r["features"]:
                    is_valid_f = False
                    break
                f_obj = r["features"][f_name]
                if not isinstance(f_obj, dict) or "availableAt" not in f_obj:
                    is_valid_f = False
                    break
                dt_avail = parse_iso_timestamp(f_obj["availableAt"])
                if dt_avail is None or dt_avail > r["predictionTimeDt"]:
                    is_valid_f = False
                    break
            if is_valid_f:
                eligible_features.append(f_name)

        sorted_features = sorted(eligible_features, key=lambda x: x.encode("utf-8"))
        
        train_row_ids = sorted([r["id"] for r in retained_rows if r["split"] == "TRAIN"], key=lambda x: x.encode("utf-8"))
        eval_row_ids = sorted([r["id"] for r in retained_rows if r["split"] == "EVAL"], key=lambda x: x.encode("utf-8"))

        succeeded_trials = []
        if isinstance(trials, list):
            for t in trials:
                if isinstance(t, dict) and t.get("status") == "SUCCEEDED":
                    m = t.get("evalMetric")
                    tid = t.get("trialId")
                    if isinstance(m, (int, float)) and math.isfinite(m) and isinstance(tid, int) and tid >= 0:
                        succeeded_trials.append(t)
        
        selected_trial_id = None
        if not succeeded_trials:
            reason_codes.add("NO_SUCCESSFUL_TRIAL")
        else:
            best_trial = sorted(succeeded_trials, key=lambda t: (-t["evalMetric"], t["trialId"]))[0]
            selected_trial_id = best_trial["trialId"]

        digest_dict = {
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": sorted_features
        }
        dataset_digest = sha256_hex(compact_json(digest_dict))
        
        if reason_codes:
            selected_trial_id = None
            if "INVALID_INPUT" in reason_codes:
                dataset_digest = None

        sorted_codes = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))
        response_data = {
            "runId": run_id,
            "selectedTrialId": selected_trial_id,
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": sorted_features,
            "datasetDigest": dataset_digest,
            "reasonCodes": sorted_codes
        }

        _bqml_store[run_id] = {
            "input": body,
            "response": response_data,
            "success": len(reason_codes) == 0 and selected_trial_id is not None
        }
        return response_data

    elif phase == "evaluate":
        run_id = body.get("runId")
        sel_trial_id = body.get("selectedTrialId")
        dataset_digest = body.get("datasetDigest")
        metric_floor = body.get("metricFloor")
        required_slices = body.get("requiredSlices")
        rows = body.get("rows")
        bytes_proc = body.get("bytesProcessed")
        max_bytes = body.get("maxBytes")

        reason_codes = set()
        lineage_valid = True

        if run_id not in _bqml_store:
            lineage_valid = False
        else:
            stored = _bqml_store[run_id]
            if not stored.get("success"):
                lineage_valid = False
            else:
                resp = stored["response"]
                if resp["selectedTrialId"] != sel_trial_id or resp["datasetDigest"] != dataset_digest:
                    lineage_valid = False
        
        if not lineage_valid:
            reason_codes.add("INVALID_LINEAGE")

        if not (isinstance(metric_floor, (int, float)) and math.isfinite(metric_floor) and 0.0 <= metric_floor <= 1.0 and
                isinstance(required_slices, dict) and isinstance(bytes_proc, int) and bytes_proc >= 0 and
                isinstance(max_bytes, int) and max_bytes >= 0 and isinstance(rows, list)):
            reason_codes.add("INVALID_INPUT")

        if isinstance(bytes_proc, int) and isinstance(max_bytes, int) and bytes_proc > max_bytes:
            reason_codes.add("BYTE_LIMIT")

        rows_valid = True
        if not isinstance(rows, list) or len(rows) == 0:
            rows_valid = False
        else:
            for r in rows:
                if not isinstance(r, dict):
                    rows_valid = False
                    break
                lbl = r.get("label")
                pred = r.get("prediction")
                slice_val = r.get("slice")
                if lbl not in [0, 1] or pred not in [0, 1] or not isinstance(slice_val, str) or len(slice_val) == 0:
                    rows_valid = False
                    break
        
        if not rows_valid and isinstance(rows, list) and len(rows) > 0:
            reason_codes.add("INVALID_TEST_ROW")

        test_metric = None
        critical_slice_pass = True

        if rows_valid and len(rows) > 0:
            correct_total = sum(1 for r in rows if r["label"] == r["prediction"])
            test_metric = round(correct_total / len(rows), 12)

            if test_metric < metric_floor:
                reason_codes.add("AGGREGATE_FLOOR")

            slice_counts = {}
            slice_correct = {}
            for r in rows:
                s = r["slice"]
                slice_counts[s] = slice_counts.get(s, 0) + 1
                if r["label"] == r["prediction"]:
                    slice_correct[s] = slice_correct.get(s, 0) + 1
            
            for req_slice, req_floor in required_slices.items():
                if req_slice not in slice_counts:
                    reason_codes.add(f"MISSING_SLICE:{req_slice}")
                    critical_slice_pass = False
                else:
                    slice_acc = round(slice_correct[req_slice] / slice_counts[req_slice], 12)
                    if slice_acc < req_floor:
                        reason_codes.add(f"SLICE_FLOOR:{req_slice}")
                        critical_slice_pass = False
        else:
            critical_slice_pass = False

        if not lineage_valid or "INVALID_INPUT" in reason_codes or not rows_valid:
            critical_slice_pass = False

        decision = "admit" if len(reason_codes) == 0 else "reject"
        sorted_codes = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))

        return {
            "runId": run_id,
            "selectedTrialId": sel_trial_id,
            "datasetDigest": dataset_digest,
            "testMetric": test_metric,
            "criticalSlicePass": critical_slice_pass,
            "decision": decision,
            "bytesProcessed": bytes_proc,
            "reasonCodes": sorted_codes
        }

    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


# =============================================================================
# Q3: POST /promote
# =============================================================================

@app.post("/promote")
async def promote_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "policy" not in body or "versions" not in body or "championVersion" not in body or "asOf" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    policy = body.get("policy")
    versions = body.get("versions")
    champion_ver = body.get("championVersion")
    as_of_raw = body.get("asOf")

    if not isinstance(policy, dict) or not isinstance(versions, list) or not isinstance(champion_ver, str):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    dt_as_of = parse_iso_timestamp(as_of_raw)
    policy_dataset_digest = policy.get("datasetDigest")
    policy_schema_digest = policy.get("schemaDigest")
    max_age_sec = policy.get("maxAgeSeconds")
    acc_floor = policy.get("accuracyFloor")
    req_slices = policy.get("requiredSlices", {})
    max_lat_ms = policy.get("maxLatencyMs")
    max_size_bytes = policy.get("maxSizeBytes")
    min_improvement = policy.get("minImprovement")

    policy_valid = (
        dt_as_of is not None and
        isinstance(policy_dataset_digest, str) and len(policy_dataset_digest) > 0 and
        isinstance(policy_schema_digest, str) and len(policy_schema_digest) > 0 and
        isinstance(max_age_sec, int) and max_age_sec >= 0 and
        isinstance(acc_floor, (int, float)) and math.isfinite(acc_floor) and 0.0 <= acc_floor <= 1.0 and
        isinstance(req_slices, dict) and
        isinstance(max_lat_ms, (int, float)) and math.isfinite(max_lat_ms) and max_lat_ms >= 0 and
        isinstance(max_size_bytes, int) and max_size_bytes >= 0 and
        isinstance(min_improvement, (int, float)) and math.isfinite(min_improvement) and 0.0 <= min_improvement <= 1.0
    )

    failed_gates = {}
    seen_versions = set()
    version_objects = []

    for v in versions:
        if not isinstance(v, dict):
            continue
        vid = v.get("version")
        v_codes = set()

        if not isinstance(vid, str) or not re.match(r"^[1-9]\d*$", vid):
            v_codes.add("INVALID_VERSION")
        else:
            if vid in seen_versions:
                v_codes.add("DUPLICATE_VERSION")
            seen_versions.add(vid)

        if not policy_valid:
            v_codes.add("INVALID_POLICY")

        eval_obj = v.get("evaluation")
        if not isinstance(eval_obj, dict):
            v_codes.add("MISSING_EVALUATION")
        else:
            created_at_raw = eval_obj.get("createdAt")
            dt_created = parse_iso_timestamp(created_at_raw)
            if dt_created is None:
                v_codes.add("INVALID_TIMESTAMP")
            elif dt_as_of is not None:
                if dt_created > dt_as_of:
                    v_codes.add("FUTURE_EVALUATION")
                elif dt_created < (dt_as_of - timedelta(seconds=max_age_sec)):
                    v_codes.add("STALE_EVALUATION")

            acc = eval_obj.get("accuracy")
            lat = eval_obj.get("latencyMs")
            size = eval_obj.get("sizeBytes")

            if not (isinstance(acc, (int, float)) and math.isfinite(acc) and
                    isinstance(lat, (int, float)) and math.isfinite(lat) and
                    isinstance(size, int) and size >= 0):
                v_codes.add("NON_FINITE")
            else:
                if not (0.0 <= acc <= 1.0):
                    v_codes.add("METRIC_RANGE")
                elif acc < acc_floor:
                    v_codes.add("ACCURACY_FLOOR")

                if lat > max_lat_ms:
                    v_codes.add("LATENCY_LIMIT")
                if size > max_size_bytes:
                    v_codes.add("SIZE_LIMIT")

            if eval_obj.get("artifactDigest") != v.get("artifactDigest"):
                v_codes.add("ARTIFACT_MISMATCH")
            if eval_obj.get("datasetDigest") != policy_dataset_digest:
                v_codes.add("DATASET_MISMATCH")
            if eval_obj.get("schemaDigest") != policy_schema_digest:
                v_codes.add("SCHEMA_MISMATCH")

            eval_slices = eval_obj.get("slices")
            if not isinstance(eval_slices, dict):
                for s_name in req_slices.keys():
                    v_codes.add(f"MISSING_SLICE:{s_name}")
            else:
                for s_name, floor_val in req_slices.items():
                    if s_name not in eval_slices:
                        v_codes.add(f"MISSING_SLICE:{s_name}")
                    else:
                        s_val = eval_slices[s_name]
                        if not (isinstance(s_val, (int, float)) and math.isfinite(s_val) and 0.0 <= s_val <= 1.0):
                            v_codes.add(f"SLICE_RANGE:{s_name}")
                        elif s_val < floor_val:
                            v_codes.add(f"SLICE_FLOOR:{s_name}")

        failed_gates[str(vid)] = sorted(list(v_codes), key=lambda x: x.encode("utf-8"))
        if len(v_codes) == 0:
            version_objects.append({
                "version": vid,
                "int_version": int(vid),
                "accuracy": eval_obj["accuracy"],
                "latencyMs": eval_obj["latencyMs"],
                "sizeBytes": eval_obj["sizeBytes"],
                "evaluation": eval_obj
            })

    eligible_versions = [v["version"] for v in version_objects]

    champion_eligible = any(v["version"] == champion_ver for v in version_objects)
    if not champion_eligible:
        return {
            "action": "block",
            "championVersion": champion_ver,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None
        }

    ranked = sorted(version_objects, key=lambda x: (-x["accuracy"], x["latencyMs"], x["sizeBytes"], x["int_version"]))
    challenger = ranked[0]
    champion_obj = next(v for v in version_objects if v["version"] == champion_ver)

    acc_diff = round(challenger["accuracy"] - champion_obj["accuracy"], 12)
    if challenger["version"] != champion_ver and acc_diff >= min_improvement:
        action = "promote"
        selected = challenger["version"]
        alias_mut = {"alias": "champion", "version": selected}
        evidence = challenger["evaluation"]
    else:
        action = "retain"
        selected = champion_ver
        alias_mut = None
        evidence = champion_obj["evaluation"]

    return {
        "action": action,
        "championVersion": champion_ver,
        "selectedVersion": selected,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": alias_mut,
        "evidence": evidence
    }


# =============================================================================
# Q4: POST /adapt
# =============================================================================

INTERVENTION_PRIORITY = ["prompt_only", "retrieval", "lora", "qlora"]

@app.post("/adapt")
async def adapt_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "operation" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    op = body.get("operation")
    if op == "choose":
        policy = body.get("policy")
        candidates = body.get("candidates")
        if not isinstance(policy, dict) or not isinstance(candidates, list) or len(candidates) != 4:
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        
        min_q = policy.get("minQuality")
        fresh_req = policy.get("freshnessRequired")
        max_lat = policy.get("maxLatencyMs")
        max_mem = policy.get("maxMemoryMb")
        max_data = policy.get("maxLabeledExamples")
        max_cost = policy.get("maxTotalCost")
        horizon = policy.get("horizonRequests")

        reason_codes = {k: [] for k in INTERVENTION_PRIORITY}
        total_costs = {}
        eligible = []

        cand_map = {}
        for c in candidates:
            if isinstance(c, dict) and "name" in c:
                cand_map[c["name"]] = c

        for name in INTERVENTION_PRIORITY:
            c = cand_map.get(name)
            codes = set()
            if not c:
                codes.add("INVALID_INPUT")
                total_costs[name] = None
            else:
                avail = c.get("available")
                q = c.get("quality")
                fresh = c.get("freshness")
                lat = c.get("latencyMs")
                mem = c.get("memoryMb")
                data = c.get("labeledExamples")
                one_cost = c.get("oneTimeCost")
                rec_cost = c.get("recurringCost")

                if not avail:
                    codes.add("UNAVAILABLE")
                if q < min_q:
                    codes.add("QUALITY_FLOOR")
                if fresh_req and not fresh:
                    codes.add("FRESHNESS_REQUIRED")
                if lat > max_lat:
                    codes.add("LATENCY_LIMIT")
                if mem > max_mem:
                    codes.add("MEMORY_LIMIT")
                if data > max_data:
                    codes.add("DATA_LIMIT")

                t_cost = round(one_cost + (horizon * rec_cost), 12)
                total_costs[name] = t_cost
                if t_cost > max_cost:
                    codes.add("COST_LIMIT")

                if len(codes) == 0:
                    eligible.append(name)
            
            reason_codes[name] = sorted(list(codes), key=lambda x: x.encode("utf-8"))

        selected = eligible[0] if len(eligible) > 0 else None
        return {
            "selected": selected,
            "eligible": eligible,
            "totalCosts": total_costs,
            "reasonCodes": reason_codes
        }

    elif op == "repair":
        reason_codes = set()
        tokens = body.get("tokens")
        tpl_count = body.get("templateApplications")
        params = body.get("parameters")
        allowed_targets = body.get("allowedTargets")
        inf_mode = body.get("inferenceMode")
        train_ids = body.get("trainRowIds")
        eval_ids = body.get("evalRowIds")
        dropout_eval = body.get("dropoutActiveDuringEval")
        artifact_files = body.get("artifactFiles")
        base_rev = body.get("baseRevision")
        ds_digest = body.get("datasetDigest")
        code_digest = body.get("codeDigest")
        cfg_digest = body.get("configDigest")
        expected_digests = body.get("expectedDigests", {})
        mb = body.get("microBatch")
        ga = body.get("gradientAccumulation")
        rep = body.get("replicas")
        exp_batch = body.get("expectedEffectiveBatch")
        chk = body.get("checkpoint")
        unint_w = body.get("uninterruptedWeights")
        res_w = body.get("resumedWeights")
        res_tol = body.get("resumeTolerance")

        tokens_valid = True
        labels = []
        if not isinstance(tokens, list) or len(tokens) == 0:
            tokens_valid = False
            reason_codes.add("INVALID_TOKEN")
        else:
            for t in tokens:
                if not (isinstance(t, dict) and isinstance(t.get("id"), int) and t.get("id") >= 0 and
                        t.get("role") in ["system", "user", "assistant"] and
                        isinstance(t.get("padding"), bool) and isinstance(t.get("text"), str)):
                    tokens_valid = False
                    reason_codes.add("INVALID_TOKEN")
                    break
        
        if tokens_valid:
            for t in tokens:
                if t["role"] == "assistant" and not t["padding"]:
                    labels.append(t["id"])
                else:
                    labels.append(-100)
        else:
            labels = [-100] * (len(tokens) if isinstance(tokens, list) else 0)

        tpl_pass = (tpl_count == 1)
        if not tpl_pass:
            reason_codes.add("CHAT_TEMPLATE_COUNT")

        trainable_params = []
        trainable_count = 0
        peft_cfg_pass = True
        
        if not (isinstance(params, list) and isinstance(allowed_targets, list)):
            peft_cfg_pass = False
            reason_codes.add("INVALID_PARAMETER")
        else:
            allowed_set = set(allowed_targets)
            param_names = set()
            for p in params:
                pname = p.get("name")
                ptarget = p.get("target")
                pnumel = p.get("numel")
                if not (isinstance(pname, str) and isinstance(ptarget, str) and isinstance(pnumel, int) and pnumel > 0):
                    peft_cfg_pass = False
                    reason_codes.add("INVALID_PARAMETER")
                    break
                if pname in param_names:
                    peft_cfg_pass = False
                    reason_codes.add("INVALID_PARAMETER")
                    break
                param_names.add(pname)

                if ptarget in allowed_set and (pname.endswith(".lora_A.weight") or pname.endswith(".lora_B.weight")):
                    trainable_params.append(pname)
                    trainable_count += pnumel
            
            if len(trainable_params) == 0:
                peft_cfg_pass = False

        trainable_params.sort(key=lambda x: x.encode("utf-8"))

        if inf_mode is not False:
            peft_cfg_pass = False
            reason_codes.add("INFERENCE_MODE")

        expected_artifacts = {"adapter_config.json", "adapter_model.safetensors"}
        if isinstance(artifact_files, list) and set(artifact_files) == expected_artifacts and len(artifact_files) == 2:
            adapter_files = sorted(artifact_files, key=lambda x: x.encode("utf-8"))
        else:
            adapter_files = []
            reason_codes.add("ADAPTER_FILE_SET")

        chk_complete = True
        chk_keys = {"model", "optimizer", "scheduler", "step", "rng", "dataPosition"}
        if not isinstance(chk, dict) or not chk_keys.issubset(set(chk.keys())):
            chk_complete = False
            reason_codes.add("INCOMPLETE_CHECKPOINT")

        lineage_pass = True
        if not (isinstance(base_rev, str) and re.match(r"^[0-9a-f]{40}$", base_rev)):
            lineage_pass = False
            reason_codes.add("MUTABLE_BASE_REVISION")
        
        if not (isinstance(ds_digest, str) and len(ds_digest) == 64 and
                isinstance(code_digest, str) and len(code_digest) == 64 and
                isinstance(cfg_digest, str) and len(cfg_digest) == 64 and
                expected_digests.get("dataset") == ds_digest and
                expected_digests.get("code") == code_digest and
                expected_digests.get("config") == cfg_digest):
            lineage_pass = False
            reason_codes.add("LINEAGE_MISMATCH")

        if not (isinstance(mb, int) and mb > 0 and isinstance(ga, int) and ga > 0 and
                isinstance(rep, int) and rep > 0 and (mb * ga * rep == exp_batch)):
            lineage_pass = False
            reason_codes.add("EFFECTIVE_BATCH_MISMATCH")

        eval_isolated = True
        if not (isinstance(train_ids, list) and len(train_ids) > 0 and
                isinstance(eval_ids, list) and len(eval_ids) > 0 and
                set(train_ids).isdisjoint(set(eval_ids))):
            eval_isolated = False
            reason_codes.add("EVAL_LEAKAGE")

        eval_det = (dropout_eval is False)
        if not eval_det:
            reason_codes.add("EVAL_DROPOUT_ACTIVE")

        resume_pass = True
        if not (isinstance(unint_w, list) and isinstance(res_w, list) and len(unint_w) == len(res_w) and len(unint_w) > 0 and
                isinstance(res_tol, (int, float)) and res_tol >= 0):
            resume_pass = False
            reason_codes.add("RESUME_DIVERGENCE")
        else:
            for u, r in zip(unint_w, res_w):
                if abs(u - r) > res_tol:
                    resume_pass = False
                    reason_codes.add("RESUME_DIVERGENCE")
                    break

        sorted_codes = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))
        return {
            "labels": labels,
            "templatePass": tpl_pass,
            "trainableParams": trainable_params,
            "trainableCount": trainable_count,
            "peftConfigPass": peft_cfg_pass,
            "adapterFiles": adapter_files,
            "checkpointComplete": chk_complete,
            "lineagePass": lineage_pass,
            "evalIsolated": eval_isolated,
            "evaluationDeterministic": eval_det,
            "resumePass": resume_pass,
            "reasonCodes": sorted_codes
        }

    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


# =============================================================================
# Q5: POST /quantize
# =============================================================================

_freeze_store: Dict[str, Dict[str, Any]] = {}

@app.post("/quantize")
async def quantize_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "phase" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    phase = body.get("phase")
    if phase == "freeze":
        freeze_id = body.get("freezeId")
        if not isinstance(freeze_id, str) or len(freeze_id) == 0 or len(freeze_id) > 128:
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        
        candidates = body.get("candidates")
        if not isinstance(candidates, list) or len(candidates) == 0:
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        if freeze_id in _freeze_store:
            stored = _freeze_store[freeze_id]
            if stored["input"] == body:
                return stored["response"]
            else:
                return JSONResponse(status_code=409, content={"error": "FREEZE_ID_CONFLICT"})

        calib_digest = body.get("calibrationDigest")
        tok_digest = body.get("tokenizerDigest")
        allowed_reasons = set(body.get("allowedUnsupportedReasons", []))

        frozen_candidates = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cname = c.get("name")
            c_codes = set()
            files = c.get("files")
            loadable = c.get("loadable")
            c_calib = c.get("calibrationDigest")
            c_tok = c.get("tokenizerDigest")
            unsup_reason = c.get("unsupportedReason")

            inventory = []
            total_bytes = None
            package_digest = None

            if not isinstance(files, dict) or len(files) == 0:
                c_codes.add("INVALID_INPUT")
            else:
                file_inv = []
                for fname, fcontent in files.items():
                    if not isinstance(fname, str) or not isinstance(fcontent, str):
                        c_codes.add("INVALID_INPUT")
                        break
                    f_bytes = fcontent.encode("utf-8")
                    file_inv.append({
                        "name": fname,
                        "bytes": len(f_bytes),
                        "sha256": hashlib.sha256(f_bytes).hexdigest().lower()
                    })
                
                if "INVALID_INPUT" not in c_codes:
                    file_inv.sort(key=lambda x: x["name"].encode("utf-8"))
                    inventory = file_inv
                    total_bytes = sum(f["bytes"] for f in file_inv)
                    package_digest = sha256_hex(compact_json(inventory))

            if unsup_reason:
                if unsup_reason in allowed_reasons:
                    status = "unsupported"
                else:
                    status = "invalid"
                    c_codes.add("UNALLOWED_UNSUPPORTED_REASON")
            else:
                if not loadable:
                    c_codes.add("NOT_LOADABLE")
                if c_calib != calib_digest:
                    c_codes.add("CALIBRATION_MISMATCH")
                if c_tok != tok_digest:
                    c_codes.add("TOKENIZER_MISMATCH")

                if len(c_codes) == 0:
                    status = "frozen"
                else:
                    status = "invalid"

            if status == "invalid" and "INVALID_INPUT" in c_codes:
                inventory = []
                total_bytes = None
                package_digest = None

            frozen_candidates.append({
                "name": cname,
                "status": status,
                "inventory": inventory,
                "totalBytes": total_bytes,
                "packageDigest": package_digest,
                "reasonCodes": sorted(list(c_codes), key=lambda x: x.encode("utf-8"))
            })

        frozen_candidates.sort(key=lambda x: x["name"].encode("utf-8"))
        response_data = {
            "freezeId": freeze_id,
            "candidates": frozen_candidates
        }

        _freeze_store[freeze_id] = {
            "input": body,
            "response": response_data
        }
        return response_data

    elif phase == "select":
        freeze_id = body.get("freezeId")
        candidates = body.get("candidates")
        policy = body.get("policy")
        latencies = body.get("latencies", {})
        rows = body.get("rows")

        if not (isinstance(freeze_id, str) and isinstance(candidates, list) and isinstance(policy, dict) and isinstance(rows, list)):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        stored_freeze = _freeze_store.get(freeze_id)

        max_bytes = policy.get("maxBytes")
        agg_floor = policy.get("aggregateFloor")
        req_slices = policy.get("requiredSlices", {})
        max_lat_ms = policy.get("maxLatencyMs")
        cand_order = policy.get("candidateOrder", [])

        results = []
        admitted_cands = []

        for cand in candidates:
            cname = cand.get("name")
            c_codes = set()
            
            if not stored_freeze or cand not in stored_freeze["response"]["candidates"]:
                c_codes.add("NOT_FROZEN")
            
            if cand.get("status") != "frozen":
                c_codes.add("NOT_FROZEN")

            inv = cand.get("inventory", [])
            recomputed_bytes = sum(f.get("bytes", 0) for f in inv)
            recomputed_digest = sha256_hex(compact_json(inv))
            if recomputed_digest != cand.get("packageDigest"):
                c_codes.add("INVALID_MANIFEST")

            preds_valid = True
            cand_rows = []
            for r in rows:
                if not isinstance(r, dict):
                    preds_valid = False
                    break
                p_map = r.get("predictions", {})
                if not isinstance(p_map, dict) or cname not in p_map or p_map[cname] not in [0, 1]:
                    preds_valid = False
                    break
                cand_rows.append((r.get("label"), p_map[cname], r.get("slice")))
            
            aggregate_acc = None
            slice_acc_map = {}

            if not preds_valid:
                c_codes.add("INVALID_PREDICTIONS")
            else:
                correct_total = sum(1 for lbl, pred, sl in cand_rows if lbl == pred)
                aggregate_acc = round(correct_total / len(cand_rows), 12) if len(cand_rows) > 0 else 0.0

                if aggregate_acc < agg_floor:
                    c_codes.add("AGGREGATE_FLOOR")

                for sl_name, floor_val in req_slices.items():
                    sl_rows = [(lbl, pred) for lbl, pred, sl in cand_rows if sl == sl_name]
                    if len(sl_rows) == 0:
                        c_codes.add(f"MISSING_SLICE:{sl_name}")
                    else:
                        sl_corr = sum(1 for lbl, pred in sl_rows if lbl == pred)
                        s_acc = round(sl_corr / len(sl_rows), 12)
                        slice_acc_map[sl_name] = s_acc
                        if s_acc < floor_val:
                            c_codes.add(f"SLICE_FLOOR:{sl_name}")

            lat = latencies.get(cname)
            if not isinstance(lat, (int, float)) or lat > max_lat_ms:
                c_codes.add("LATENCY_LIMIT")

            if recomputed_bytes > max_bytes:
                c_codes.add("SIZE_LIMIT")

            admitted = (len(c_codes) == 0)
            res_entry = {
                "name": cname,
                "aggregate": aggregate_acc,
                "slices": slice_acc_map,
                "totalBytes": recomputed_bytes,
                "latencyMs": lat,
                "admitted": admitted,
                "reasonCodes": sorted(list(c_codes), key=lambda x: x.encode("utf-8"))
            }
            results.append(res_entry)
            if admitted:
                admitted_cands.append(res_entry)

        results.sort(key=lambda x: cand_order.index(x["name"]) if x["name"] in cand_order else 999)

        selected = None
        pkg_manifest = None
        if admitted_cands:
            winner = sorted(admitted_cands, key=lambda x: (
                x["totalBytes"],
                x["latencyMs"],
                cand_order.index(x["name"]) if x["name"] in cand_order else 999
            ))[0]
            selected = winner["name"]
            winner_cand_orig = next(c for c in candidates if c["name"] == selected)
            pkg_manifest = winner_cand_orig

        return {
            "freezeId": freeze_id,
            "selected": selected,
            "results": results,
            "packageManifest": pkg_manifest
        }

    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})


# =============================================================================
# Q6: POST /pipeline
# =============================================================================

_pipeline_sessions: Dict[str, Dict[str, Any]] = {}

DAG_NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]

@app.post("/pipeline")
async def pipeline_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST"})
    
    if not isinstance(body, dict) or "session" not in body or "revision" not in body or "inputs" not in body or "events" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST"})
    
    session_id = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events")

    if not isinstance(session_id, str) or not isinstance(revision, int) or revision <= 0 or not isinstance(inputs, dict) or not isinstance(events, list):
        return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST"})

    req_input_keys = [
        "generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
        "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
        "schemaDigest", "publishConfig"
    ]
    for k in req_input_keys:
        if k not in inputs or not isinstance(inputs[k], str) or len(inputs[k]) == 0:
            return JSONResponse(status_code=400, content={"error": "INVALID_REQUEST"})

    session = _pipeline_sessions.setdefault(session_id, {
        "revision": revision,
        "inputs": inputs,
        "cache": {},
        "attempts": {},
        "events_seen": {}
    })

    if session["revision"] == revision:
        if session["inputs"] != inputs:
            return JSONResponse(status_code=409, content={"error": "REVISION_CONFLICT"})
    elif revision > session["revision"]:
        session["revision"] = revision
        session["inputs"] = inputs
        session["attempts"] = {}

    def compute_dag_keys():
        keys = {}
        dep_digests = {}
        
        k_vd = sha256_hex(compact_json([inputs["generation"], inputs["checksum"]]))
        keys["verify_data"] = k_vd
        dep_digests["verify_data"] = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
            "cacheKey": k_vd
        }

        k_prep = sha256_hex(compact_json([inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]]))
        keys["prepare"] = k_prep
        dep_digests["prepare"] = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "cacheKey": k_prep
        }

        prep_cached = session["cache"].get(k_prep)
        if prep_cached:
            k_train = sha256_hex(compact_json([prep_cached["artifact"], inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]]))
            keys["train"] = k_train
            dep_digests["train"] = {
                "prepareArtifact": prep_cached["artifact"],
                "trainCode": inputs["trainCode"],
                "trainConfig": inputs["trainConfig"],
                "runtime": inputs["runtime"],
                "cacheKey": k_train
            }
        else:
            keys["train"] = None
            dep_digests["train"] = {}

        train_cached = session["cache"].get(keys.get("train")) if keys.get("train") else None
        if train_cached:
            k_eval = sha256_hex(compact_json([train_cached["artifact"], inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]]))
            keys["evaluate"] = k_eval
            dep_digests["evaluate"] = {
                "trainArtifact": train_cached["artifact"],
                "canonicalData": inputs["canonicalData"],
                "evaluateCode": inputs["evaluateCode"],
                "evaluateConfig": inputs["evaluateConfig"],
                "cacheKey": k_eval
            }
        else:
            keys["evaluate"] = None
            dep_digests["evaluate"] = {}

        eval_cached = session["cache"].get(keys.get("evaluate")) if keys.get("evaluate") else None
        if eval_cached:
            k_reg = sha256_hex(compact_json([eval_cached["artifact"], inputs["schemaDigest"]]))
            keys["register"] = k_reg
            dep_digests["register"] = {
                "evaluateArtifact": eval_cached["artifact"],
                "schemaDigest": inputs["schemaDigest"],
                "cacheKey": k_reg
            }
        else:
            keys["register"] = None
            dep_digests["register"] = {}

        reg_cached = session["cache"].get(keys.get("register")) if keys.get("register") else None
        if reg_cached:
            k_pub = sha256_hex(compact_json([reg_cached["artifact"], inputs["publishConfig"]]))
            keys["publish"] = k_pub
            dep_digests["publish"] = {
                "registerArtifact": reg_cached["artifact"],
                "publishConfig": inputs["publishConfig"],
                "cacheKey": k_pub
            }
        else:
            keys["publish"] = None
            dep_digests["publish"] = {}

        return keys, dep_digests

    accepted_event_ids = []
    ignored_event_ids = []

    for ev in events:
        if not isinstance(ev, dict):
            return JSONResponse(status_code=400, content={"error": "INVALID_EVENT"})
        
        ev_id = ev.get("eventId")
        ev_rev = ev.get("revision")
        ev_node = ev.get("node")
        ev_att = ev.get("attempt")
        ev_status = ev.get("status")
        ev_key = ev.get("key")
        ev_art = ev.get("artifactDigest")
        ev_rcpt = ev.get("receiptId")

        ev_compact = compact_json(ev)

        if ev_id in session["events_seen"]:
            if session["events_seen"][ev_id] == ev_compact:
                ignored_event_ids.append(ev_id)
                continue
            else:
                return JSONResponse(status_code=409, content={"error": "EVENT_ID_CONFLICT"})

        if ev_rev != session["revision"]:
            ignored_event_ids.append(ev_id)
            continue

        keys, _ = compute_dag_keys()
        current_node_key = keys.get(ev_node)

        if ev_node not in DAG_NODES or current_node_key is None or ev_key != current_node_key:
            ignored_event_ids.append(ev_id)
            continue

        if ev_status not in ["started", "succeeded", "retryable_failed", "terminal_failed"]:
            ignored_event_ids.append(ev_id)
            continue

        if ev_status == "succeeded":
            if not isinstance(ev_art, str) or len(ev_art) == 0:
                ignored_event_ids.append(ev_id)
                continue
            if ev_node in ["register", "publish"]:
                if ev_rcpt != f"receipt:{ev_node}:{ev_key}":
                    ignored_event_ids.append(ev_id)
                    continue
            else:
                if ev_rcpt is not None:
                    ignored_event_ids.append(ev_id)
                    continue
        else:
            if ev_art is not None or ev_rcpt is not None:
                ignored_event_ids.append(ev_id)
                continue

        cached_entry = session["cache"].get(current_node_key)
        if cached_entry:
            if ev_status == "succeeded":
                if ev_art != cached_entry["artifact"]:
                    return JSONResponse(status_code=409, content={"error": "EVIDENCE_CONFLICT"})
                else:
                    ignored_event_ids.append(ev_id)
                    continue
            else:
                return JSONResponse(status_code=409, content={"error": "STATUS_CONFLICT"})

        curr_state = session["attempts"].get(ev_node)
        if curr_state is None:
            if ev_status == "started" and ev_att == 1:
                session["attempts"][ev_node] = {"attempt": 1, "status": "started", "eventId": ev_id}
                session["events_seen"][ev_id] = ev_compact
                accepted_event_ids.append(ev_id)
            else:
                ignored_event_ids.append(ev_id)
                continue
        else:
            c_att = curr_state["attempt"]
            c_stat = curr_state["status"]

            if c_stat == "terminal_failed":
                return JSONResponse(status_code=409, content={"error": "STATUS_CONFLICT"})

            if ev_att < c_att:
                ignored_event_ids.append(ev_id)
                continue

            if c_stat == "started" and ev_att == c_att:
                if ev_status in ["succeeded", "retryable_failed", "terminal_failed"]:
                    session["attempts"][ev_node] = {"attempt": c_att, "status": ev_status, "eventId": ev_id}
                    if ev_status == "succeeded":
                        session["cache"][current_node_key] = {"artifact": ev_art, "eventId": ev_id, "receipt": ev_rcpt}
                    session["events_seen"][ev_id] = ev_compact
                    accepted_event_ids.append(ev_id)
                else:
                    return JSONResponse(status_code=409, content={"error": "STATUS_CONFLICT"})
            elif c_stat == "retryable_failed" and ev_att == c_att + 1 and ev_status == "started":
                session["attempts"][ev_node] = {"attempt": ev_att, "status": "started", "eventId": ev_id}
                session["events_seen"][ev_id] = ev_compact
                accepted_event_ids.append(ev_id)
            else:
                return JSONResponse(status_code=409, content={"error": "STATUS_CONFLICT"})

    keys, dep_digests = compute_dag_keys()
    node_responses = []
    upstream_terminal_encountered = False
    upstream_pending_encountered = False

    for node in DAG_NODES:
        k = keys.get(node)
        c_entry = session["cache"].get(k) if k else None
        att_state = session["attempts"].get(node)

        if upstream_terminal_encountered:
            node_responses.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_TERMINAL"],
                "dependencyDigests": dep_digests.get(node, {}),
                "triggeringEventIds": []
            })
            continue

        if upstream_pending_encountered:
            node_responses.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_PENDING"],
                "dependencyDigests": dep_digests.get(node, {}),
                "triggeringEventIds": []
            })
            continue

        if c_entry:
            node_responses.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": ["CACHE_HIT"],
                "dependencyDigests": dep_digests.get(node, {}),
                "triggeringEventIds": [c_entry["eventId"]]
            })
        elif att_state:
            if att_state["status"] == "started":
                node_responses.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": ["RUNNING"],
                    "dependencyDigests": dep_digests.get(node, {}),
                    "triggeringEventIds": [att_state["eventId"]]
                })
                upstream_pending_encountered = True
            elif att_state["status"] == "retryable_failed":
                node_responses.append({
                    "node": node,
                    "action": "rerun",
                    "reasonCodes": ["RETRYABLE_FAILURE"],
                    "dependencyDigests": dep_digests.get(node, {}),
                    "triggeringEventIds": [att_state["eventId"]]
                })
                upstream_pending_encountered = True
            elif att_state["status"] == "terminal_failed":
                node_responses.append({
                    "node": node,
                    "action": "block",
                    "reasonCodes": ["TERMINAL_FAILURE"],
                    "dependencyDigests": dep_digests.get(node, {}),
                    "triggeringEventIds": [att_state["eventId"]]
                })
                upstream_terminal_encountered = True
        else:
            node_responses.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": ["CACHE_MISS"],
                "dependencyDigests": dep_digests.get(node, {}),
                "triggeringEventIds": []
            })
            upstream_pending_encountered = True

    return {
        "revision": session["revision"],
        "acceptedEventIds": accepted_event_ids,
        "ignoredEventIds": ignored_event_ids,
        "nodes": node_responses
    }


# =============================================================================
# Q7: POST /verify-bundle
# =============================================================================

REQUIRED_BUNDLE_FILES = {
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json"
}

@app.post("/verify-bundle")
async def verify_bundle(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    if not isinstance(body, dict) or "policy" not in body or "files" not in body:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    
    policy = body.get("policy")
    files = body.get("files")
    if not isinstance(policy, dict) or not isinstance(files, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    violations = set()

    req_slices = policy.get("requiredSlices")
    lic = policy.get("license")
    use = policy.get("intendedUse")
    lim = policy.get("limitations")

    if not (isinstance(req_slices, list) and len(req_slices) > 0 and
            all(isinstance(s, str) and len(s) > 0 for s in req_slices) and
            len(req_slices) == len(set(req_slices)) and
            isinstance(lic, str) and len(lic) > 0 and
            isinstance(use, str) and len(use) > 0 and
            isinstance(lim, str) and len(lim) > 0):
        violations.add("INVALID_POLICY")

    for req_f in REQUIRED_BUNDLE_FILES:
        if req_f not in files:
            violations.add(f"MISSING_FILE:{req_f}")
        else:
            if not isinstance(files[req_f], str):
                violations.add(f"INVALID_FILE:{req_f}")

    unsafe_exts = (".bin", ".pt", ".pth", ".pkl", ".pickle")
    for f_name in files.keys():
        if f_name not in REQUIRED_BUNDLE_FILES:
            violations.add("UNTRACKED_FILE")
        if any(f_name.endswith(ext) for ext in unsafe_exts):
            violations.add("UNSAFE_WEIGHTS")

    recomputed_inv = []
    for f_name, f_content in files.items():
        if f_name != "inventory.json" and isinstance(f_content, str):
            f_bytes = f_content.encode("utf-8")
            recomputed_inv.append({
                "name": f_name,
                "bytes": len(f_bytes),
                "sha256": hashlib.sha256(f_bytes).hexdigest().lower()
            })
    recomputed_inv.sort(key=lambda x: x["name"].encode("utf-8"))
    recomputed_inv_json = compact_json(recomputed_inv)
    inventory_digest = sha256_hex(recomputed_inv_json)

    if "inventory.json" in files and isinstance(files["inventory.json"], str):
        try:
            supplied_inv = json.loads(files["inventory.json"])
            if supplied_inv != recomputed_inv:
                violations.add("INVENTORY_MISMATCH")
        except Exception:
            violations.add("INVALID_JSON:inventory.json")

    if "adapter_config.json" in files and isinstance(files["adapter_config.json"], str):
        try:
            cfg = json.loads(files["adapter_config.json"])
            if not (isinstance(cfg, dict) and isinstance(cfg.get("r"), int) and cfg.get("r") > 0 and
                    isinstance(cfg.get("target_modules"), list) and len(cfg.get("target_modules")) > 0 and
                    len(cfg.get("target_modules")) == len(set(cfg.get("target_modules"))) and
                    all(isinstance(m, str) and len(m) > 0 for m in cfg.get("target_modules"))):
                violations.add("INVALID_ADAPTER_CONFIG")
        except Exception:
            violations.add("INVALID_JSON:adapter_config.json")

    model_digest = None
    if "adapter_model.safetensors" in files and isinstance(files["adapter_model.safetensors"], str):
        model_digest = sha256_hex(files["adapter_model.safetensors"].encode("utf-8"))

    eval_digest = None
    if "evaluation.json" in files and isinstance(files["evaluation.json"], str):
        eval_digest = sha256_hex(files["evaluation.json"].encode("utf-8"))

    manifest = None
    if "training_manifest.json" in files and isinstance(files["training_manifest.json"], str):
        try:
            manifest = json.loads(files["training_manifest.json"])
            if not isinstance(manifest, dict):
                violations.add("INVALID_TRAINING_MANIFEST")
            else:
                base_rev = manifest.get("baseRevision")
                if not (isinstance(base_rev, str) and re.match(r"^[0-9a-f]{40}$", base_rev)):
                    violations.add("MUTABLE_BASE_REVISION")

                req_man_fields = ["task", "datasetDigest", "codeDigest", "trainingConfigDigest", "modelArtifactDigest", "evaluationArtifactDigest"]
                for mf in req_man_fields:
                    val = manifest.get(mf)
                    if not isinstance(val, str) or len(val) == 0:
                        violations.add(f"MISSING_MANIFEST_FIELD:{mf}")

                if manifest.get("modelArtifactDigest") != model_digest:
                    violations.add("MODEL_ARTIFACT_MISMATCH")
                if manifest.get("evaluationArtifactDigest") != eval_digest:
                    violations.add("EVALUATION_ARTIFACT_MISMATCH")
        except Exception:
            violations.add("INVALID_JSON:training_manifest.json")

    if "evaluation.json" in files and isinstance(files["evaluation.json"], str):
        try:
            eval_data = json.loads(files["evaluation.json"])
            if not isinstance(eval_data, dict):
                violations.add("INVALID_EVALUATION")
            else:
                if eval_data.get("modelArtifactDigest") != model_digest:
                    violations.add("EVALUATION_DIGEST_MISMATCH")

                agg = eval_data.get("aggregate")
                if not (isinstance(agg, (int, float)) and math.isfinite(agg) and 0.0 <= agg <= 1.0):
                    violations.add("INVALID_AGGREGATE")

                slices = eval_data.get("slices", {})
                if not isinstance(slices, dict):
                    for s in req_slices or []:
                        violations.add(f"MISSING_SLICE:{s}")
                else:
                    for s in req_slices or []:
                        if s not in slices:
                            violations.add(f"MISSING_SLICE:{s}")
                        else:
                            sval = slices[s]
                            if not (isinstance(sval, (int, float)) and math.isfinite(sval) and 0.0 <= sval <= 1.0):
                                violations.add(f"SLICE_RANGE:{s}")
        except Exception:
            violations.add("INVALID_JSON:evaluation.json")

    if "README.md" in files and isinstance(files["README.md"], str):
        readme_str = files["README.md"]
        pattern = r"<!-- tds-model-card (.*?) -->"
        matches = re.findall(pattern, readme_str, re.DOTALL)
        if len(matches) == 0:
            violations.add("MODEL_CARD_COUNT")
            violations.add("MISSING_MODEL_CARD")
        elif len(matches) > 1:
            violations.add("MODEL_CARD_COUNT")
        else:
            payload_str = matches[0].strip()
            try:
                card_json = json.loads(payload_str)
                if not isinstance(card_json, dict):
                    violations.add("INVALID_MODEL_CARD")
                else:
                    if (manifest and
                        (card_json.get("task") != manifest.get("task") or
                         card_json.get("baseRevision") != manifest.get("baseRevision") or
                         card_json.get("datasetDigest") != manifest.get("datasetDigest") or
                         card_json.get("modelArtifactDigest") != manifest.get("modelArtifactDigest") or
                         card_json.get("license") != lic or
                         card_json.get("intendedUse") != use or
                         card_json.get("limitations") != lim)):
                        violations.add("MODEL_CARD_MISMATCH")
            except Exception:
                violations.add("INVALID_MODEL_CARD")

    decision = "admit" if len(violations) == 0 else "reject"
    sorted_violations = sorted(list(violations), key=lambda x: x.encode("utf-8"))

    return {
        "decision": decision,
        "violations": sorted_violations,
        "inventoryDigest": inventory_digest
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
