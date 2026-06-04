from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import random
import heapq
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
import warnings
from datetime import datetime
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_ROOT = Path(__file__).parent / "data" / "semantic_boundary"

DATASET_ALIASES = {
    "gum": "gum",
    "codesearchnet": "codesearchnet",
    "juice": "juice",
    "lean_workbook": "lean_workbook",
    "proofnet": "proofnet",
}

CODESEARCHNET_LANGUAGES = ["python", "java", "javascript", "ruby", "go", "php"]
SPLIT_NAMES = ("train", "valid", "test")

LEAN_WORKBOOK_JSON_URL = "https://huggingface.co/datasets/internlm/Lean-Workbook/resolve/main/lean_workbook.json"
PROOFNET_VALID_URL = "https://huggingface.co/datasets/hoskinson-center/proofnet/resolve/main/valid.jsonl"
PROOFNET_TEST_URL = "https://huggingface.co/datasets/hoskinson-center/proofnet/resolve/main/test.jsonl"
GUM_REPO_URL = "https://github.com/amir-zeldes/gum.git"
JUICE_REPO_URL = "https://github.com/microsoft/DataScienceProblems.git"
JUICE_ARCHIVE_URL = "https://media.githubusercontent.com/media/microsoft/DataScienceProblems/main/juice-github-repos.tar.gz"
CODESEARCHNET_ZIP_TEMPLATE = "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{language}.zip"
CODESEARCHNET_ZENODO_TEMPLATE = "https://zenodo.org/records/7857872/files/{language}.zip?download=1"
GUM_DISRPT_BASE_URL = "https://raw.githubusercontent.com/amir-zeldes/gum/master/rst/disrpt"

PUNCT_NO_SPACE_BEFORE = {
    ".",
    ",",
    ";",
    ":",
    "?",
    "!",
    "%",
    ")",
    "]",
    "}",
    "''",
    "'s",
    "'re",
    "'ve",
    "'m",
    "'d",
    "'ll",
    "n't",
}
PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "``", "$", "#"}
MATH_CONNECTOR_SEGMENTS = {
    "then",
    "thus",
    "hence",
    "therefore",
    "so",
    "so that",
    "finally",
    "next",
    "now",
}
TASK_FAMILY_BY_SOURCE = {
    "gum": "discourse",
    "codesearchnet": "code",
    "juice": "notebook",
    "lean_workbook": "formal_math",
    "proofnet": "formal_math",
}
TASK_USABLE_LABEL_SCHEMA_VERSION = "task_usable_v1"
FINAL_ANSWER_PATTERNS = (
    re.compile(r"\boxed\s*\{"),
    re.compile(r"^\s*####\s*.+$"),
    re.compile(r"^\s*(?:final\s+answer|answer)\s*[:=]", re.IGNORECASE),
)
CODE_SEGMENT_PREFIXES = (
    "def ",
    "async def ",
    "class ",
    "return ",
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "try:",
    "except ",
    "with ",
    "import ",
    "from ",
    "public ",
    "private ",
    "protected ",
    "function ",
    "const ",
    "let ",
    "var ",
    "package ",
    "func ",
    "#include",
)
CODESEARCHNET_MIN_BLOCK_LINES = 2
CODESEARCHNET_MIN_BLOCK_CHARS = 40
LEAN_TACTIC_PREFIXES = (
    "simp",
    "simpa",
    "rw",
    "rwa",
    "nlinarith",
    "linarith",
    "ring",
    "ring_nf",
    "norm_num",
    "omega",
    "positivity",
    "aesop",
    "tauto",
    "finish",
    "exact",
    "apply",
    "refine",
    "constructor",
    "cases",
    "rcases",
    "rintro",
    "intro",
    "intros",
    "have",
    "let",
    "specialize",
    "obtain",
    "use",
    "exists",
    "choose",
    "field_simp",
    "norm_cast",
    "push_neg",
    "by_cases",
    "by_contra",
    "contrapose",
    "conv",
    "calc",
    "all_goals",
    "first",
    "assumption",
    "trivial",
    "decide",
    "left",
    "right",
    "split",
    "subst",
    "induction",
    "funext",
    "ext",
)


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        log(f"[skip] {destination} already exists")
        return

    log(f"[download] {url} -> {destination}")
    request = urllib.request.Request(url, headers={"User-Agent": "Codex boundary data downloader"})
    with urllib.request.urlopen(request) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def download_with_fallbacks(urls: Sequence[str], destination: Path, overwrite: bool = False) -> None:
    ensure_dir(destination.parent)
    if destination.exists() and not overwrite:
        log(f"[skip] {destination} already exists")
        return

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            download_file(url, destination, overwrite=overwrite)
            return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            log(f"[warn] download failed for {url}: {exc}")

    raise RuntimeError(f"All download sources failed for {destination}") from last_error


def clone_or_update_repo(repo_url: str, destination: Path, skip_lfs_smudge: bool = True) -> None:
    if destination.exists():
        log(f"[update] {destination}")
        run_cmd(["git", "fetch", "--all", "--tags"], cwd=destination)
        run_cmd(["git", "pull", "--ff-only"], cwd=destination)
        return

    ensure_dir(destination.parent)
    env = {"GIT_LFS_SKIP_SMUDGE": "1"} if skip_lfs_smudge else None
    log(f"[clone] {repo_url} -> {destination}")
    run_cmd(["git", "clone", "--depth", "1", repo_url, str(destination)], env_overrides=env)


def extract_tar_if_needed(archive_path: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        log(f"[skip] extracted directory already exists: {destination}")
        return
    ensure_dir(destination)
    log(f"[extract] {archive_path} -> {destination}")
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)


def extract_zip_if_needed(archive_path: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        log(f"[skip] extracted directory already exists: {destination}")
        return
    ensure_dir(destination)
    log(f"[extract] {archive_path} -> {destination}")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)


def dataset_root(root: Path, dataset_name: str) -> Path:
    return root / "raw" / dataset_name


def processed_root(root: Path) -> Path:
    return root / "processed"


def processed_root_for_subdir(root: Path, output_subdir: str) -> Path:
    return processed_root(root) / output_subdir


def download_gum(root: Path) -> None:
    base = dataset_root(root, "gum")
    download_file(f"{GUM_DISRPT_BASE_URL}/eng.erst.gum_train.tok", base / "disrpt" / "eng.erst.gum_train.tok")
    download_file(f"{GUM_DISRPT_BASE_URL}/eng.erst.gum_dev.tok", base / "disrpt" / "eng.erst.gum_dev.tok")
    download_file(f"{GUM_DISRPT_BASE_URL}/eng.erst.gum_test.tok", base / "disrpt" / "eng.erst.gum_test.tok")
    download_file("https://raw.githubusercontent.com/amir-zeldes/gum/master/rst/README.md", base / "disrpt" / "README.md")


def download_lean_workbook(root: Path) -> None:
    target = dataset_root(root, "lean_workbook") / "lean_workbook.json"
    download_file(LEAN_WORKBOOK_JSON_URL, target)


def download_proofnet(root: Path) -> None:
    base = dataset_root(root, "proofnet")
    download_file(PROOFNET_VALID_URL, base / "valid.jsonl")
    download_file(PROOFNET_TEST_URL, base / "test.jsonl")


def download_juice(root: Path, extract_archive: bool = False) -> None:
    base = dataset_root(root, "juice")
    archive_path = base / "juice-github-repos.tar.gz"
    download_file(JUICE_ARCHIVE_URL, archive_path)
    if extract_archive:
        extract_tar_if_needed(archive_path, base / "juice-github-repos")


def download_codesearchnet(root: Path, languages: Sequence[str], extract_archives: bool = True) -> None:
    base = dataset_root(root, "codesearchnet")
    download_dir = base / "downloads"
    extracted_dir = base / "extracted"
    for language in languages:
        archive_path = download_dir / f"{language}.zip"
        urls = [
            CODESEARCHNET_ZIP_TEMPLATE.format(language=language),
            CODESEARCHNET_ZENODO_TEMPLATE.format(language=language),
        ]
        download_with_fallbacks(urls, archive_path)
        if extract_archives:
            extract_zip_if_needed(archive_path, extracted_dir / language)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_sentences_in_paragraph(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?!;:])\s+(?=(?:[A-Z0-9$\\(\[]|\"|'))", paragraph)
    return [part.strip() for part in parts if part.strip()]


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
    segments: List[str] = []
    for paragraph in paragraphs:
        cleaned = split_sentences_in_paragraph(paragraph)
        if cleaned:
            segments.extend(cleaned)
        else:
            segments.append(paragraph)
    return segments


def normalize_math_paragraph(text: str) -> str:
    text = normalize_newlines(text)
    text = re.sub(r"([.?!])(?=(?:[A-Z$\\]))", r"\1 ", text)
    lines = []
    for line in text.splitlines():
        stripped = re.sub(r"[ \t]+", " ", line).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def join_nonempty_lines(parts: Sequence[str]) -> str:
    return "\n".join(part for part in parts if part and part.strip())


def is_standalone_latex_env_marker(text: str) -> bool:
    return bool(re.fullmatch(r"\\(?:begin|end)\{[^{}]+\}", text.strip()))


def is_latex_begin_marker(text: str) -> bool:
    return bool(re.fullmatch(r"\\begin\{[^{}]+\}", text.strip()))


def is_latex_end_marker(text: str) -> bool:
    return bool(re.fullmatch(r"\\end\{[^{}]+\}", text.strip()))


def is_math_connector_segment(text: str) -> bool:
    stripped = text.strip().lower()
    stripped = stripped.rstrip(",;:")
    return stripped in MATH_CONNECTOR_SEGMENTS

def merge_math_segments(segments: Sequence[str]) -> List[str]:
    collapsed: List[str] = []
    idx = 0
    raw_segments = [segment.strip() for segment in segments if segment and segment.strip()]
    while idx < len(raw_segments):
        segment = raw_segments[idx]
        if segment == "$$":
            block_parts = [segment]
            idx += 1
            while idx < len(raw_segments):
                block_parts.append(raw_segments[idx])
                if raw_segments[idx] == "$$":
                    break
                idx += 1
            collapsed.append(join_nonempty_lines(block_parts))
            idx += 1
            continue

        collapsed.append(segment)
        idx += 1

    merged: List[str] = []
    pending_prefix: List[str] = []
    for segment in collapsed:
        if segment == "$":
            continue

        if segment in {"\\[", "\\]"}:
            pending_prefix.append(segment)
            continue

        if is_latex_begin_marker(segment) or is_math_connector_segment(segment):
            pending_prefix.append(segment)
            continue

        if pending_prefix:
            segment = join_nonempty_lines([*pending_prefix, segment])
            pending_prefix = []

        if is_latex_end_marker(segment):
            if merged:
                merged[-1] = join_nonempty_lines([merged[-1], segment])
            else:
                merged.append(segment)
            continue

        merged.append(segment)

    if pending_prefix:
        if merged:
            merged[-1] = join_nonempty_lines([merged[-1], *pending_prefix])
        else:
            merged.append(join_nonempty_lines(pending_prefix))

    return [segment for segment in merged if segment and segment.strip() and not is_standalone_latex_env_marker(segment.strip())]


def is_math_structure_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in {"$$", "$", "\\[", "\\]"}:
        return True
    if is_standalone_latex_env_marker(stripped):
        return True
    if any(pattern.search(stripped) for pattern in FINAL_ANSWER_PATTERNS):
        return True
    if any(marker in stripped for marker in ("=", "\\boxed{", "\\frac", "\\sum", "\\int", "\\begin{", "\\end{")):
        return True
    return bool(re.search(r"\d", stripped) and re.search(r"[=+\-*/^<>]", stripped))

def split_math_reasoning_clauses(text: str) -> List[str]:
    clause_patterns = (
        re.compile(r",\s+(?=(?:which|so|thus|therefore|hence)\b)", re.IGNORECASE),
        re.compile(r"\s+(?=(?:Then|Therefore|Thus|Hence|So|Next|Finally)\b)"),
    )
    clauses = [text.strip()]
    for pattern in clause_patterns:
        next_clauses: List[str] = []
        for clause in clauses:
            parts = [part.strip() for part in pattern.split(clause) if part.strip()]
            if len(parts) > 1 and any(looks_like_math_reasoning_step(part) for part in parts[1:]):
                next_clauses.extend(parts)
            else:
                next_clauses.append(clause)
        clauses = next_clauses
    return clauses



def split_math_text(text: str) -> List[str]:
    text = normalize_newlines(text).strip()
    if not text:
        return []

    paragraphs = [normalize_math_paragraph(paragraph) for paragraph in re.split(r"\n\s*\n+", text)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]

    segments: List[str] = []
    for paragraph in paragraphs:
        if paragraph == "$$" or "$$\n" in paragraph or "\n$$" in paragraph or is_standalone_latex_env_marker(paragraph):
            segments.append(paragraph)
            continue

        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        for line in lines:
            if line.startswith("\\begin{") or line.startswith("\\end{"):
                segments.append(line)
                continue

            sentence_segments = split_sentences_in_paragraph(line)
            if len(sentence_segments) <= 1:
                clause_segments = split_math_reasoning_clauses(line)
                if len(clause_segments) > 1:
                    sentence_segments = clause_segments
            if is_math_structure_line(line) and len(sentence_segments) <= 1:
                segments.append(line)
                continue

            if sentence_segments:
                for sentence_segment in sentence_segments:
                    if sentence_segment.startswith("$$"):
                        trailing = sentence_segment[2:].lstrip()
                        if trailing and re.match(r"^[A-Z0-9\(\[]", trailing):
                            if segments:
                                segments[-1] = segments[-1].rstrip()
                                if not segments[-1].endswith("$$"):
                                    segments[-1] = f"{segments[-1]} $$"
                            sentence_segment = trailing
                    if sentence_segment:
                        segments.append(sentence_segment)
            else:
                segments.append(line)

    return merge_math_segments(segments)

def normalize_whitespace(text: str) -> str:
    text = normalize_newlines(text)
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_code_cell_text(text: str) -> str:
    return normalize_newlines(text).strip("\n")


def infer_task_family(source: str) -> str:
    return TASK_FAMILY_BY_SOURCE.get(str(source or "").strip().lower(), "generic")


def looks_like_explicit_final_answer(text: str) -> bool:
    stripped = normalize_newlines(text).strip()
    if not stripped:
        return False
    if any(pattern.search(stripped) for pattern in FINAL_ANSWER_PATTERNS):
        return True
    return bool(re.fullmatch(r"[-+]?\$?\d+(?:\.\d+)?(?:/\d+)?\$?", stripped))


def looks_like_math_reasoning_step(text: str) -> bool:
    stripped = normalize_newlines(text).strip()
    if not stripped:
        return False
    if looks_like_explicit_final_answer(stripped):
        return True
    if is_math_structure_line(stripped):
        return True
    return bool(re.search(r"\d", stripped) and re.search(r"[=+\-*/^<>]", stripped))


def looks_like_code_segment(text: str) -> bool:
    stripped = normalize_newlines(text).strip()
    if not stripped:
        return False
    if stripped.startswith(CODE_SEGMENT_PREFIXES):
        return True
    if re.search(r"[{}();]", stripped) and not stripped.endswith("."):
        return True
    if "\n" in stripped and any(token in stripped for token in ("def ", "class ", "return ", "import ", "from ", "{")):
        return True
    return False


def dedupe_preserve_order(labels: Sequence[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def infer_segment_boundary_types(record: Dict) -> List[List[str]]:
    segments = [str(segment) for segment in record.get("segments", []) if str(segment).strip()]
    task_family = infer_task_family(record.get("source", ""))
    cell_types = [str(cell_type).strip().lower() or "unknown" for cell_type in record.get("cell_types", [])]

    boundary_types: List[List[str]] = []
    for idx, segment in enumerate(segments):
        labels: List[str] = []
        is_last = idx == len(segments) - 1
        cell_type = cell_types[idx] if idx < len(cell_types) else ""

        if task_family == "code":
            if idx == 0 and record.get("has_docstring") and not looks_like_code_segment(segment):
                labels.append("internal_step_boundary")
            if idx > 0 or looks_like_code_segment(segment):
                labels.append("safe_commit_boundary")
            if not labels:
                labels.append("internal_step_boundary")
        elif task_family == "notebook":
            if looks_like_explicit_final_answer(segment):
                labels.append("final_answer_anchor")
            elif cell_type == "code" or looks_like_code_segment(segment):
                labels.append("safe_commit_boundary")
            else:
                labels.append("internal_step_boundary")
        elif task_family == "formal_math":
            if looks_like_explicit_final_answer(segment) or (is_last and looks_like_math_reasoning_step(segment)):
                labels.append("final_answer_anchor")
            else:
                labels.append("internal_step_boundary")
        else:
            labels.append("internal_step_boundary")
            if looks_like_explicit_final_answer(segment):
                labels.append("final_answer_anchor")

        boundary_types.append(dedupe_preserve_order(labels))

    return boundary_types


def enrich_task_usable_record(record: Dict, training_mode: str = "separate") -> Dict:
    enriched = dict(record)
    source = str(record.get("source", "")).strip().lower()
    if source:
        enriched["source"] = source
    source_dataset = str(record.get("source_dataset", source)).strip().lower()
    if source_dataset:
        enriched["source_dataset"] = source_dataset
    segments = [str(segment) for segment in record.get("segments", []) if str(segment).strip()]
    if segments:
        enriched["segments"] = segments
    enriched["task_family"] = infer_task_family(source)
    enriched["training_mode"] = training_mode or str(record.get("training_mode", "separate"))
    enriched["label_schema_version"] = TASK_USABLE_LABEL_SCHEMA_VERSION
    if segments:
        boundary_types = infer_segment_boundary_types({**record, **enriched})
        enriched["segment_boundary_types"] = boundary_types
        enriched["has_final_answer_anchor"] = any("final_answer_anchor" in labels for labels in boundary_types)
    else:
        enriched["has_final_answer_anchor"] = bool(record.get("has_final_answer_anchor", False))
    return enriched



def stable_split(key: str, train_cutoff: int = 980) -> str:
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
    return "train" if bucket < train_cutoff else "valid"


def detokenize_tokens(tokens: Sequence[str]) -> str:
    pieces: List[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if not pieces:
            pieces.append(token)
            continue

        prev = pieces[-1]
        if token in PUNCT_NO_SPACE_BEFORE or token.startswith("'"):
            pieces[-1] = prev + token
        elif prev in PUNCT_NO_SPACE_AFTER or prev.endswith(("/", "(", "[", "{", "$", "#")):
            pieces[-1] = prev + token
        elif token in {"-", "–", "—"}:
            pieces[-1] = prev + token
        else:
            pieces.append(token)
    return " ".join(pieces).replace(" - ", "-").strip()


def write_jsonl(path: Path, records: Iterable[Dict], training_mode: str = "separate") -> int:
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            row = dict(record)
            if row.get("segments"):
                row = enrich_task_usable_record(
                    row,
                    training_mode=training_mode or str(row.get("training_mode", "separate")),
                )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count



def parse_gum_tok(path: Path, split_name: str) -> Iterator[Dict]:
    current_doc_id: Optional[str] = None
    current_segments: List[str] = []
    current_tokens: List[str] = []

    def flush_segment() -> None:
        nonlocal current_tokens
        if current_tokens:
            segment = detokenize_tokens(current_tokens)
            if segment:
                current_segments.append(segment)
            current_tokens = []

    def flush_doc() -> Optional[Dict]:
        nonlocal current_doc_id, current_segments
        flush_segment()
        if current_doc_id and current_segments:
            record = {
                "source": "gum",
                "split": split_name,
                "doc_id": current_doc_id,
                "segments": current_segments,
            }
            current_doc_id = None
            current_segments = []
            return record
        current_doc_id = None
        current_segments = []
        return None

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("# newdoc id = "):
                record = flush_doc()
                if record is not None:
                    yield record
                current_doc_id = line.split("=", 1)[1].strip()
                continue
            if not line:
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                continue
            token = parts[1]
            segmentation = parts[-1].strip()
            if segmentation == "Seg=B-seg":
                flush_segment()
            current_tokens.append(token)

    record = flush_doc()
    if record is not None:
        yield record


def preprocess_gum(root: Path) -> Dict[str, Path]:
    gum_repo = dataset_root(root, "gum") / "disrpt"
    split_paths = {
        "train": gum_repo / "eng.erst.gum_train.tok",
        "valid": gum_repo / "eng.erst.gum_dev.tok",
        "test": gum_repo / "eng.erst.gum_test.tok",
    }
    outputs: Dict[str, Path] = {}
    for split_name, source_path in split_paths.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing GUM segmentation file: {source_path}")
        out_path = processed_root(root) / "gum" / f"{split_name}.jsonl"
        count = write_jsonl(out_path, parse_gum_tok(source_path, split_name))
        log(f"[preprocess] GUM {split_name}: {count} documents -> {out_path}")
        outputs[split_name] = out_path
    return outputs


def preprocess_lean_workbook(
    root: Path,
    include_formal_statements: bool = False,
    include_proof_steps: bool = True,
) -> Dict[str, Path]:
    source_path = dataset_root(root, "lean_workbook") / "lean_workbook.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing Lean Workbook file: {source_path}")

    with open(source_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    buckets: DefaultDict[str, List[Dict]] = defaultdict(list)
    for idx, row in enumerate(records):
        key = row.get("formal_statement") or row.get("natural_language_statement") or f"lean_workbook_{idx}"
        split_name = stable_split(key)
        segments: List[str] = []

        nl_statement = row.get("natural_language_statement", "")
        segments.extend(split_math_text(nl_statement))

        if include_formal_statements:
            formal_statement = normalize_whitespace(row.get("formal_statement", ""))
            if formal_statement:
                segments.append(formal_statement)

        if include_proof_steps:
            for step in row.get("proof", []):
                normalized_step = normalize_whitespace(step)
                if normalized_step:
                    segments.append(normalized_step)

        segments = [segment for segment in segments if segment]
        if not segments:
            continue

        buckets[split_name].append(
            {
                "source": "lean_workbook",
                "split": split_name,
                "row_id": idx,
                "lean_split": row.get("split", ""),
                "tags": row.get("tags", []),
                "segments": segments,
            }
        )

    outputs: Dict[str, Path] = {}
    for split_name, rows in buckets.items():
        out_path = processed_root(root) / "lean_workbook" / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows)
        log(f"[preprocess] Lean Workbook {split_name}: {count} records -> {out_path}")
        outputs[split_name] = out_path
    return outputs


def preprocess_proofnet(root: Path, include_formal_statements: bool = False) -> Dict[str, Path]:
    base = dataset_root(root, "proofnet")
    split_paths = {
        "valid": base / "valid.jsonl",
        "test": base / "test.jsonl",
    }
    outputs: Dict[str, Path] = {}
    for split_name, source_path in split_paths.items():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing ProofNet file: {source_path}")

        def iter_records() -> Iterator[Dict]:
            with open(source_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    segments: List[str] = []
                    segments.extend(split_math_text(row.get("nl_statement", "")))
                    segments.extend(split_math_text(row.get("nl_proof", "")))
                    if include_formal_statements:
                        formal_statement = normalize_whitespace(row.get("formal_statement", ""))
                        if formal_statement:
                            segments.append(formal_statement)
                    segments = [segment for segment in segments if segment]
                    if not segments:
                        continue
                    yield {
                        "source": "proofnet",
                        "split": split_name,
                        "id": row.get("id", ""),
                        "segments": segments,
                    }

        out_path = processed_root(root) / "proofnet" / f"{split_name}.jsonl"
        count = write_jsonl(out_path, iter_records())
        log(f"[preprocess] ProofNet {split_name}: {count} records -> {out_path}")
        outputs[split_name] = out_path
    return outputs




def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def looks_like_lean_tactic_line(line: str) -> bool:
    stripped = normalize_whitespace(line)
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(LEAN_TACTIC_PREFIXES):
        return True
    if lowered.startswith("by "):
        return True
    if ":=" in stripped and any(token in lowered for token in ("have ", "let ", "show ", "calc", " by")):
        return True
    if any(token in lowered for token in ("nlinarith", "linarith", "ring_nf", "ring", "norm_num", "field_simp", "simp ", "rw ", "aesop")):
        return True
    words = re.findall(r"[A-Za-z_']+", stripped)
    return len(words) <= 4 and not stripped.endswith(".")


def extract_lean_explanatory_segments(proof_steps: Sequence[str]) -> List[str]:
    explanatory: List[str] = []
    for step in proof_steps:
        normalized = normalize_newlines(str(step or "")).strip()
        if not normalized:
            continue
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        non_tactic_lines = [line for line in lines if not looks_like_lean_tactic_line(line)]
        if non_tactic_lines:
            explanatory.extend(split_math_text("\n".join(non_tactic_lines)))
    return [segment for segment in explanatory if segment and segment.strip()]


def should_keep_lean_workbook_math_v2_row(
    row: Dict,
    min_statement_segments: int = 3,
) -> Tuple[bool, Dict[str, object]]:
    proof_steps = [str(step).strip() for step in row.get("proof", []) if str(step).strip()]
    statement_segments = split_math_text(row.get("natural_language_statement", ""))
    has_final_anchor = any(looks_like_explicit_final_answer(segment) for segment in statement_segments)
    explanatory_segments = extract_lean_explanatory_segments(proof_steps)
    stats: Dict[str, object] = {
        "proof_step_count": len(proof_steps),
        "statement_segment_count": len(statement_segments),
        "has_final_answer_anchor": has_final_anchor,
        "explanatory_segment_count": len(explanatory_segments),
        "kept_reason": "",
    }
    if not proof_steps:
        stats["kept_reason"] = "missing_proof"
        return False, stats
    if len(statement_segments) >= min_statement_segments or has_final_anchor:
        stats["kept_reason"] = "statement_reasoning"
        return True, stats
    if explanatory_segments:
        stats["kept_reason"] = "explanatory_proof"
        return True, stats
    stats["kept_reason"] = "statement_too_shallow"
    return False, stats


def sample_rows_by_limit(rows_by_split: Dict[str, List[Dict]], limit: int, seed: int) -> Dict[str, List[Dict]]:
    if limit <= 0:
        return {split_name: [] for split_name in rows_by_split}
    flattened: List[Tuple[str, Dict]] = []
    for split_name, rows in rows_by_split.items():
        for row in rows:
            flattened.append((split_name, row))
    if len(flattened) <= limit:
        return {split_name: list(rows) for split_name, rows in rows_by_split.items()}
    rng = random.Random(seed)
    selected = rng.sample(flattened, limit)
    sampled: DefaultDict[str, List[Dict]] = defaultdict(list)
    for split_name, row in selected:
        sampled[split_name].append(dict(row))
    return dict(sampled)


def write_rows_by_split(root: Path, output_subdir: str, dataset_name: str, rows_by_split: Dict[str, List[Dict]]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for split_name, rows in rows_by_split.items():
        if not rows:
            continue
        out_path = processed_root_for_subdir(root, output_subdir) / dataset_name / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows)
        log(f"[preprocess][math_v2] {dataset_name} {split_name}: {count} records -> {out_path}")
        outputs[split_name] = str(out_path)
    return outputs


def prepare_math_v2_corpora(
    root: Path,
    output_subdir: str = "math_v2_full",
    pilot_output_subdir: Optional[str] = None,
    pilot_limits: Optional[Dict[str, int]] = None,
    seed: int = 0,
    include_formal_statements: bool = False,
) -> Dict[str, object]:
    root = Path(root)
    pilot_limits = pilot_limits or {}
    summary: Dict[str, object] = {
        "full_outputs": {},
        "pilot_outputs": {},
        "stats": {},
    }

    proofnet_rows: DefaultDict[str, List[Dict]] = defaultdict(list)
    for split_name in ("valid", "test"):
        source_path = dataset_root(root, "proofnet") / f"{split_name}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing ProofNet file: {source_path}")
        with open(source_path, "r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                segments = split_math_text(row.get("nl_statement", "")) + split_math_text(row.get("nl_proof", ""))
                if include_formal_statements:
                    formal_statement = normalize_whitespace(row.get("formal_statement", ""))
                    if formal_statement:
                        segments.append(formal_statement)
                segments = [segment for segment in segments if segment and segment.strip()]
                if not segments:
                    continue
                proofnet_rows[split_name].append(
                    {
                        "source": "proofnet",
                        "source_dataset": "proofnet",
                        "split": split_name,
                        "id": row.get("id", ""),
                        "segments": segments,
                        "proof_style": "natural_language_proof",
                        "is_filtered_from_lean": False,
                        "dataset_version": "math_v2",
                    }
                )

    lean_source = dataset_root(root, "lean_workbook") / "lean_workbook.json"
    if not lean_source.exists():
        raise FileNotFoundError(f"Missing Lean Workbook file: {lean_source}")
    with open(lean_source, "r", encoding="utf-8") as handle:
        lean_records = json.load(handle)

    lean_rows: DefaultDict[str, List[Dict]] = defaultdict(list)
    lean_kept = 0
    lean_dropped = 0
    for idx, row in enumerate(lean_records):
        keep, keep_stats = should_keep_lean_workbook_math_v2_row(row)
        if not keep:
            lean_dropped += 1
            continue
        key = row.get("formal_statement") or row.get("natural_language_statement") or f"lean_workbook_{idx}"
        split_name = stable_split(key)
        statement_segments = split_math_text(row.get("natural_language_statement", ""))
        explanatory_segments = extract_lean_explanatory_segments(row.get("proof", []))
        segments = [segment for segment in statement_segments if segment and segment.strip()]
        if explanatory_segments:
            segments.extend(explanatory_segments)
            proof_style = "filtered_lean_statement_plus_explanatory_proof"
        else:
            proof_style = "filtered_lean_statement"
        if include_formal_statements:
            formal_statement = normalize_whitespace(row.get("formal_statement", ""))
            if formal_statement:
                segments.append(formal_statement)
        segments = [segment for segment in segments if segment and segment.strip()]
        if not segments:
            lean_dropped += 1
            continue
        lean_rows[split_name].append(
            {
                "source": "lean_workbook",
                "source_dataset": "lean_workbook",
                "split": split_name,
                "row_id": idx,
                "lean_split": row.get("split", ""),
                "tags": row.get("tags", []),
                "segments": segments,
                "proof_style": proof_style,
                "is_filtered_from_lean": True,
                "dataset_version": "math_v2",
                "selection_reason": keep_stats.get("kept_reason", ""),
            }
        )
        lean_kept += 1

    summary["full_outputs"]["proofnet"] = write_rows_by_split(root, output_subdir, "proofnet", dict(proofnet_rows))
    summary["full_outputs"]["lean_workbook"] = write_rows_by_split(root, output_subdir, "lean_workbook", dict(lean_rows))

    if pilot_output_subdir:
        summary["pilot_outputs"]["proofnet"] = write_rows_by_split(
            root,
            pilot_output_subdir,
            "proofnet",
            sample_rows_by_limit(dict(proofnet_rows), int(pilot_limits.get("proofnet", 0) or 0), seed + 11),
        )
        summary["pilot_outputs"]["lean_workbook"] = write_rows_by_split(
            root,
            pilot_output_subdir,
            "lean_workbook",
            sample_rows_by_limit(dict(lean_rows), int(pilot_limits.get("lean_workbook", 0) or 0), seed + 29),
        )

    audit_lines = [
        "# \u6570\u5b66\u8bed\u4e49\u8fb9\u754c V2 \u6570\u636e\u5ba1\u8ba1",
        "",
        f"- \u65f6\u95f4\u6233\uff1a{timestamp_slug()}",
        f"- ProofNet \u6837\u672c\u6570\uff1a{sum(len(rows) for rows in proofnet_rows.values())}",
        f"- Lean Workbook \u4fdd\u7559\u6837\u672c\uff1a{lean_kept}",
        f"- Lean Workbook \u5254\u9664\u6837\u672c\uff1a{lean_dropped}",
        f"- \u8f93\u51fa\u76ee\u5f55\uff1a{processed_root_for_subdir(root, output_subdir)}",
    ]
    audit_report_path = processed_root_for_subdir(root, output_subdir) / f"math_v2_audit_{timestamp_slug()}.md"
    ensure_dir(audit_report_path.parent)
    audit_report_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    summary["audit_report_path"] = str(audit_report_path)
    summary["stats"] = {
        "proofnet_records": sum(len(rows) for rows in proofnet_rows.values()),
        "lean_kept": lean_kept,
        "lean_dropped": lean_dropped,
    }
    return summary

def iter_notebook_paths(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.ipynb"):
        if path.is_file():
            yield path


def notebook_cell_text(cell: Dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def preprocess_juice(root: Path, prefix_cell_type: bool = False) -> Dict[str, Path]:
    notebook_root = dataset_root(root, "juice") / "juice-github-repos"
    if not notebook_root.exists():
        raise FileNotFoundError(
            f"Missing JuICe notebook directory: {notebook_root}. "
            "Download with --juice-extract first."
        )

    buckets: DefaultDict[str, List[Dict]] = defaultdict(list)
    stats: Counter = Counter()
    for notebook_path in iter_notebook_paths(notebook_root):
        try:
            with open(notebook_path, "r", encoding="utf-8") as handle:
                notebook = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            stats["skipped_invalid_notebook"] += 1
            log(f"[preprocess][warn] JuICe skip invalid notebook: {notebook_path} ({exc})")
            continue

        cells = notebook.get("cells", [])
        segments: List[str] = []
        cell_types: List[str] = []
        for cell in cells:
            cell_type = str(cell.get("cell_type", "")).strip() or "unknown"
            raw_text = notebook_cell_text(cell)
            if cell_type == "code":
                text = normalize_code_cell_text(raw_text)
            else:
                text = normalize_whitespace(raw_text)
            if not text:
                continue
            if prefix_cell_type:
                text = f"[{cell_type.upper()}] {text}"
            segments.append(text)
            cell_types.append(cell_type)

        if len(segments) < 2:
            stats["dropped_too_short"] += 1
            continue

        key = notebook.get("metadata", {}).get("kernelspec", {}).get("name", "") + str(notebook_path)
        split_name = stable_split(key, train_cutoff=990)
        buckets[split_name].append(
            {
                "source": "juice",
                "split": split_name,
                "notebook_path": str(notebook_path.relative_to(notebook_root)),
                "cell_types": cell_types,
                "segments": segments,
            }
        )
        stats["kept"] += 1

    outputs: Dict[str, Path] = {}
    for split_name, rows in buckets.items():
        out_path = processed_root(root) / "juice" / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows)
        log(f"[preprocess] JuICe {split_name}: {count} notebooks -> {out_path}")
        outputs[split_name] = out_path

    if stats:
        summary = ", ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        log(f"[preprocess] JuICe stats: {summary}")
    return outputs


def iter_jsonl_rows(path: Path) -> Iterator[Dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def strip_leading_docstring_stmt(node: ast.AST) -> List[ast.stmt]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        value = body[0].value
        if isinstance(value.value, str):
            body = body[1:]
    return body



def leading_space_count(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def clean_segments(segments: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for segment in segments:
        normalized = segment.strip("\n")
        if not normalized.strip():
            continue

        # Merge brace-only / end-only fragments back into the previous block.
        # They are structural closers rather than standalone semantic steps.
        if normalized.strip() in {"}", "};", "end"} and cleaned:
            cleaned[-1] = f"{cleaned[-1]}\n{normalized}"
            continue

        cleaned.append(normalized)
    return cleaned


def looks_like_function_signature(signature: str) -> bool:
    for line in signature.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            return True
    return False


def looks_like_ruby_signature(signature: str) -> bool:
    for line in signature.splitlines():
        if line.strip().startswith("def "):
            return True
    return False


def first_substantive_code_line(segment: str) -> str:
    for raw_line in segment.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in {"{", "}", "};", "end"}:
            continue
        if stripped.startswith(("#", "//", "/*", "*")):
            continue
        return stripped
    return ""


def substantive_code_line_count(segment: str) -> int:
    count = 0
    for raw_line in segment.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in {"{", "}", "};", "end"}:
            continue
        if stripped.startswith(("#", "//", "/*", "*")):
            continue
        count += 1
    return count


def looks_like_codesearchnet_definition(language: str, segment: str) -> bool:
    first = first_substantive_code_line(segment)
    if not first:
        return False

    if language == "python":
        return first.startswith(("def ", "async def ", "class "))
    if language == "ruby":
        return bool(re.match(r"^(def|class|module)\b", first))
    if language == "javascript":
        return bool(
            re.match(r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b", first)
            or re.match(r"^(?:export\s+)?(?:default\s+)?class\b", first)
            or re.match(r"^(?:async\s+)?[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{$", first)
            or re.match(r"^(?:get|set)\s+[A-Za-z_$][\w$]*\s*\([^)]*\)\s*\{$", first)
        )
    if language == "java":
        return bool(
            re.search(r"\b(class|interface|enum)\b", first)
            or ("(" in first and first.endswith("{") and not re.match(r"^(if|for|while|switch|catch)\b", first))
        )
    if language == "php":
        return bool(
            re.match(r"^(?:public|private|protected|static|final|abstract|\s)*function\b", first)
            or re.search(r"\bclass\b", first)
        )
    if language == "go":
        return first.startswith(("func ", "type "))
    return False


def starts_with_allowed_internal_boundary(language: str, segment: str) -> bool:
    first = first_substantive_code_line(segment)
    if not first:
        return False

    if language == "python":
        return bool(re.match(r"^(if|for|while|try|with|match)\b", first))
    if language == "ruby":
        return bool(re.match(r"^(if|for|while|begin|case)\b", first))
    if language == "go":
        return bool(re.match(r"^(if|for|switch|select)\b", first))
    return bool(re.match(r"^(if|for|while|try|with|switch)\b", first))


def segment_meets_codesearchnet_minimum(segment: str) -> bool:
    return (
        substantive_code_line_count(segment) >= CODESEARCHNET_MIN_BLOCK_LINES
        or len(segment.strip()) >= CODESEARCHNET_MIN_BLOCK_CHARS
    )


def is_simple_codesearchnet_segment(language: str, segment: str) -> bool:
    if looks_like_codesearchnet_definition(language, segment) or starts_with_allowed_internal_boundary(language, segment):
        return False

    first = first_substantive_code_line(segment)
    if not first:
        return True

    if substantive_code_line_count(segment) <= 1:
        return True

    return bool(
        re.match(r"^(return|yield|raise|throw|break|continue|pass)\b", first)
        or re.match(r"^(import|from|using|package|require)\b", first)
        or re.match(r"^(const|let|var)\b", first)
        or re.match(r"^(?:self\.)?[A-Za-z_][\w\.]*\s*[+\-*/%]?=", first)
        or re.match(r"^[A-Za-z_][\w\.]*\s*\(", first)
    )


def merge_code_segments(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left.rstrip()}\n{right.lstrip()}"


def coarsen_codesearchnet_segments(language: str, segments: Sequence[str]) -> List[str]:
    cleaned = list(clean_segments(segments))
    if len(cleaned) <= 1:
        return cleaned

    coarse: List[str] = [cleaned[0]]
    buffered: List[str] = []

    for segment in cleaned[1:]:
        if looks_like_codesearchnet_definition(language, segment) or starts_with_allowed_internal_boundary(language, segment):
            if buffered:
                buffered_block = "\n".join(buffered).strip()
                if buffered_block:
                    if segment_meets_codesearchnet_minimum(buffered_block):
                        coarse.append(buffered_block)
                    else:
                        segment = merge_code_segments(buffered_block, segment)
                buffered = []
            coarse.append(segment)
            continue

        if is_simple_codesearchnet_segment(language, segment) or not segment_meets_codesearchnet_minimum(segment):
            if len(coarse) > 1:
                coarse[-1] = merge_code_segments(coarse[-1], segment)
            else:
                buffered.append(segment)
            continue

        buffered.append(segment)

    if buffered:
        buffered_block = "\n".join(buffered).strip()
        if buffered_block:
            if len(coarse) == 1 or segment_meets_codesearchnet_minimum(buffered_block):
                coarse.append(buffered_block)
            else:
                coarse[-1] = merge_code_segments(coarse[-1], buffered_block)

    return list(clean_segments(coarse))


def strip_internal_python_docstring_lines(segment: str) -> str:
    lines = normalize_newlines(segment).splitlines()
    if not lines:
        return ""

    kept: List[str] = []
    active_delimiter: Optional[str] = None

    for line in lines:
        stripped = line.lstrip()
        if active_delimiter is None:
            match = re.match(r"(?i)(?:[rubf]{0,2})?(\"\"\"|''')", stripped)
            if match:
                active_delimiter = match.group(1)
                suffix = stripped[match.end() :]
                if active_delimiter in suffix:
                    active_delimiter = None
                continue
            kept.append(line)
            continue

        if active_delimiter in line:
            active_delimiter = None

    return "\n".join(kept).strip()


def is_internal_docstring_like_segment(language: str, segment: str) -> bool:
    if language != "python":
        return False

    cleaned = normalize_newlines(segment).strip()
    if not cleaned:
        return False

    return not strip_internal_python_docstring_lines(cleaned)


def filter_codesearchnet_segments(language: str, segments: Sequence[str]) -> List[str]:
    cleaned = list(clean_segments(segments))
    if not cleaned:
        return []

    filtered: List[str] = [cleaned[0]]
    for segment in cleaned[1:]:
        normalized = segment
        if language == "python":
            normalized = strip_internal_python_docstring_lines(segment)
        else:
            normalized = normalize_newlines(segment).strip()
        if not normalized:
            continue
        filtered.append(normalized)

    return list(clean_segments(filtered))


def segments_are_reasonable(language: str, segments: Sequence[str], method: str) -> bool:
    cleaned = [segment.strip() for segment in segments if segment and segment.strip()]
    if len(cleaned) < 2 or len(cleaned) > 64:
        return False

    short_segments = 0
    for segment in cleaned:
        if len(segment) < 3 and segment not in {"...", "pass", "[]", "{}", "()"}:
            short_segments += 1
    if short_segments > max(1, len(cleaned) // 4):
        return False

    if method == "ast":
        if not looks_like_function_signature(cleaned[0]):
            return False
        if len(cleaned[0].splitlines()) > 12:
            return False
        if max(len(segment.splitlines()) for segment in cleaned[1:]) > 200:
            return False

    if language == "python" and not looks_like_function_signature(cleaned[0]):
        return False
    if language == "ruby" and method == "ruby_indent" and not looks_like_ruby_signature(cleaned[0]):
        return False

    return True


def split_python_code_ast(code: str) -> Optional[List[str]]:
    lines = code.splitlines()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(code)
    except SyntaxError:
        return None

    if len(module.body) != 1:
        return None

    node = module.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None

    docstring_stmt = None
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
        value = node.body[0].value
        if isinstance(value.value, str):
            docstring_stmt = node.body[0]

    body_stmts = strip_leading_docstring_stmt(node)
    if not body_stmts:
        return None

    signature_end_lineno = (docstring_stmt.lineno - 1) if docstring_stmt is not None else (body_stmts[0].lineno - 1)
    signature = "\n".join(lines[:signature_end_lineno]).strip()
    if not signature or not looks_like_function_signature(signature):
        return None

    segments: List[str] = [signature]
    for stmt in body_stmts:
        end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
        segment = "\n".join(lines[stmt.lineno - 1 : end_lineno]).strip()
        if segment:
            segments.append(segment)

    segments = clean_segments(segments)
    if not segments_are_reasonable("python", segments, method="ast"):
        return None
    return segments


def split_python_code_heuristic(code: str) -> List[str]:
    lines = code.splitlines()
    if not lines:
        return []

    signature_end = None
    saw_def = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("def ") or stripped.startswith("async def "):
            saw_def = True
        signature_candidate = stripped.split("#", 1)[0].rstrip()
        if saw_def and signature_candidate.endswith(":"):
            signature_end = idx
            break

    if signature_end is None:
        return []

    signature = "\n".join(lines[: signature_end + 1]).strip()
    if not looks_like_function_signature(signature):
        return []

    body_lines = lines[signature_end + 1 :]
    if not body_lines:
        return []

    # If we fell back from AST because a function is too large, we still do not
    # want the leading Python docstring to become dozens of fake boundary labels.
    first_nonempty = 0
    while first_nonempty < len(body_lines) and not body_lines[first_nonempty].strip():
        first_nonempty += 1
    if first_nonempty < len(body_lines):
        stripped = body_lines[first_nonempty].lstrip()
        match = re.match(r"(?i)(?:[rubf]{0,2})?(\"\"\"|''')", stripped)
        if match:
            delimiter = match.group(1)
            same_line = stripped[len(match.group(0)) :]
            docstring_end = None
            if delimiter in same_line:
                docstring_end = first_nonempty
            else:
                for idx in range(first_nonempty + 1, len(body_lines)):
                    if delimiter in body_lines[idx]:
                        docstring_end = idx
                        break
            if docstring_end is not None:
                body_lines = body_lines[docstring_end + 1 :]

    nonempty_body = [line for line in body_lines if line.strip()]
    if not nonempty_body:
        return []

    base_indent = min(leading_space_count(line) for line in nonempty_body)
    if base_indent <= 0:
        return []

    segments: List[str] = [signature]
    current: List[str] = []
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                current.append(line)
            continue

        indent = leading_space_count(line)
        if indent == base_indent and current:
            segment = "\n".join(current).strip()
            if segment:
                segments.append(segment)
            current = [line]
        else:
            current.append(line)

    if current:
        segment = "\n".join(current).strip()
        if segment:
            segments.append(segment)

    return clean_segments(segments)


def strip_strings_for_counting(line: str) -> str:
    line = re.sub(r"//.*$", "", line)
    line = re.sub(r"#.*$", "", line)
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    return line


def segment_line_count(segment: str) -> int:
    return max(1, len(segment.splitlines()))


def segment_starts_with_control_keyword(segment: str) -> bool:
    for line in segment.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#")):
            continue
        return bool(
            re.match(
                r"^(if|else|elif|elsif|when|for|while|switch|case|try|catch|finally|with|return|throw|break|continue|rescue|ensure)\b",
                stripped,
            )
        )
    return False


def choose_adjacent_merge_index(segments: Sequence[str]) -> int:
    best_idx = 0
    best_score: Optional[Tuple[int, int, int, int]] = None
    for idx in range(len(segments) - 1):
        left = segments[idx]
        right = segments[idx + 1]
        signature_penalty = 1000 if idx == 0 else 0
        control_penalty = 100 if segment_starts_with_control_keyword(left) or segment_starts_with_control_keyword(right) else 0
        long_penalty = 50 if segment_line_count(left) > 12 or segment_line_count(right) > 12 else 0
        score = (
            signature_penalty + control_penalty + long_penalty,
            segment_line_count(left) + segment_line_count(right),
            len(left) + len(right),
            idx,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx
    return best_idx


def coarsen_segments(segments: Sequence[str], max_segments: int = 48) -> List[str]:
    merged = list(clean_segments(segments))
    while len(merged) > max_segments:
        idx = choose_adjacent_merge_index(merged)
        merged[idx : idx + 2] = [f"{merged[idx]}\n{merged[idx + 1]}"]
        merged = list(clean_segments(merged))
    return merged


def split_ruby_code_heuristic(code: str) -> List[str]:
    lines = code.splitlines()
    if not lines:
        return []

    signature_start = None
    signature_end = None
    paren_depth = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if signature_start is None:
            if not re.match(r"^\s*def\b", line):
                continue
            signature_start = idx
            signature_end = idx
        clean = strip_strings_for_counting(line)
        paren_depth += clean.count("(") - clean.count(")")
        signature_end = idx
        if paren_depth <= 0 and not stripped.endswith(",") and not stripped.endswith("\\"):
            break

    if signature_start is None or signature_end is None:
        return []

    signature = "\n".join(lines[: signature_end + 1]).strip()
    if not looks_like_ruby_signature(signature):
        return []

    body_lines = lines[signature_end + 1 :]
    nonempty_body = [
        line
        for line in body_lines
        if line.strip() and not re.match(r"^\s*end\b", line.strip())
    ]
    if not nonempty_body:
        return []

    base_indent = min(leading_space_count(line) for line in nonempty_body)
    segments: List[str] = [signature]
    current: List[str] = []
    pending_prefix: List[str] = []

    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                current.append(line)
            elif pending_prefix:
                pending_prefix.append(line)
            continue

        if stripped.startswith("#"):
            if current:
                current.append(line)
            else:
                pending_prefix.append(line)
            continue

        indent = leading_space_count(line)
        if indent < base_indent or re.match(r"^(end|else|elsif|when|rescue|ensure)\b", stripped):
            if not current and pending_prefix:
                current = pending_prefix[:]
                pending_prefix = []
            if current:
                current.append(line)
            else:
                current = [line]
            continue

        if indent == base_indent and current:
            segment = "\n".join(current).strip()
            if segment:
                segments.append(segment)
            current = pending_prefix[:] + [line] if pending_prefix else [line]
            pending_prefix = []
        else:
            if not current and pending_prefix:
                current = pending_prefix[:]
                pending_prefix = []
            if current:
                current.append(line)
            else:
                current = [line]

    if current:
        segment = "\n".join(current).strip()
        if segment:
            segments.append(segment)

    return clean_segments(segments)


def split_code_fallback(code: str, language: str) -> List[str]:
    lines = code.splitlines()
    if not lines:
        return []

    segments: List[str] = []
    current: List[str] = []
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    body_started = False

    for line in lines:
        current.append(line)
        stripped = line.strip()
        clean = strip_strings_for_counting(line)

        brace_depth += clean.count("{") - clean.count("}")
        paren_depth += clean.count("(") - clean.count(")")
        bracket_depth += clean.count("[") - clean.count("]")

        if not body_started:
            if language in {"java", "javascript", "php", "go"} and "{" in clean:
                body_started = True
                segment = "\n".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                continue
            if language == "ruby" and re.match(r"^\s*def\b", stripped):
                body_started = True
                continue

        boundary = False
        if not stripped:
            boundary = True
        elif language in {"java", "javascript", "php", "go"}:
            if brace_depth <= 1 and paren_depth == 0 and bracket_depth == 0:
                if stripped.endswith(";") or stripped == "}" or stripped.endswith("}") or stripped.endswith("{"):
                    boundary = True
        elif language == "ruby":
            if re.match(r"^\s*(end|else|elsif|when|rescue|ensure)\b", stripped):
                boundary = True
            elif stripped.endswith("do") or stripped.endswith("{") or stripped.endswith("}"):
                boundary = True
        else:
            if stripped.endswith(";"):
                boundary = True

        if boundary:
            segment = "\n".join(current).strip()
            if segment:
                segments.append(segment)
            current = []

    if current:
        segment = "\n".join(current).strip()
        if segment:
            segments.append(segment)

    return [segment for segment in segments if segment]


def split_codesearchnet_code(language: str, code: str) -> Optional[Tuple[str, str, List[str]]]:
    language = language.lower()
    if language == "python":
        ast_segments = split_python_code_ast(code)
        if ast_segments is not None:
            ast_segments = coarsen_codesearchnet_segments(language, ast_segments)
            ast_segments = filter_codesearchnet_segments(language, ast_segments)
            if segments_are_reasonable(language, ast_segments, method="ast"):
                return ("ast", "high", ast_segments)

        heuristic_segments = coarsen_codesearchnet_segments(language, split_python_code_heuristic(code))
        heuristic_segments = filter_codesearchnet_segments(language, heuristic_segments)
        if segments_are_reasonable(language, heuristic_segments, method="heuristic"):
            return ("heuristic", "fallback", heuristic_segments)
        return None

    if language == "ruby":
        ruby_segments = coarsen_codesearchnet_segments(language, split_ruby_code_heuristic(code))
        if segments_are_reasonable(language, ruby_segments, method="ruby_indent"):
            return ("ruby_indent", "fallback", ruby_segments)

        fallback_segments = coarsen_codesearchnet_segments(language, split_code_fallback(code, language=language))
        if segments_are_reasonable(language, fallback_segments, method="heuristic"):
            return ("heuristic", "fallback", fallback_segments)
        return None

    fallback_segments = coarsen_codesearchnet_segments(language, split_code_fallback(code, language=language))
    if len(fallback_segments) > 64:
        coarsened_segments = coarsen_segments(fallback_segments, max_segments=48)
        coarsened_segments = coarsen_codesearchnet_segments(language, coarsened_segments)
        if segments_are_reasonable(language, coarsened_segments, method="coarsened"):
            return ("heuristic_coarsened", "coarsened", coarsened_segments)
        return None

    if not segments_are_reasonable(language, fallback_segments, method="heuristic"):
        return None
    return ("heuristic", "fallback", fallback_segments)


def find_codesearchnet_files(root: Path, languages: Sequence[str], splits: Sequence[str]) -> Iterator[Tuple[str, str, Path]]:
    base = dataset_root(root, "codesearchnet") / "extracted"
    for language in languages:
        language_root = base / language
        if not language_root.exists():
            continue
        candidates = list(language_root.rglob("*.jsonl")) + list(language_root.rglob("*.jsonl.gz"))
        for path in candidates:
            split_name = None
            for split in splits:
                if f"{os.sep}{split}{os.sep}" in str(path):
                    split_name = split
                    break
            if split_name is None:
                continue
            yield language, split_name, path



def preprocess_codesearchnet(
    root: Path,
    languages: Sequence[str],
    output_subdir: str = "codesearchnet",
    dataset_version: str = "codesearchnet_default",
) -> Dict[str, Path]:
    buckets: DefaultDict[str, List[Dict]] = defaultdict(list)
    language_stats: DefaultDict[str, Counter] = defaultdict(Counter)

    for language, split_name, path in find_codesearchnet_files(root, languages, ["train", "valid", "test"]):
        normalized_split = "valid" if split_name == "valid" else split_name
        for row in iter_jsonl_rows(path):
            code = row.get("code") or row.get("original_string") or ""
            code = code.strip()
            if not code:
                language_stats[language]["dropped_empty_code"] += 1
                continue

            segmentation = split_codesearchnet_code(language, code)
            if segmentation is None:
                language_stats[language]["dropped_lowconf"] += 1
                continue

            segmentation_method, boundary_quality, code_segments = segmentation
            docstring = normalize_whitespace(row.get("docstring", ""))
            segments = list(code_segments)

            if len(code_segments) < 2:
                language_stats[language]["dropped_too_few_code_segments"] += 1
                continue
            if not segments:
                language_stats[language]["dropped_empty_segments"] += 1
                continue

            language_stats[language]["kept"] += 1
            language_stats[language][f"method_{segmentation_method}"] += 1
            language_stats[language][f"quality_{boundary_quality}"] += 1

            buckets[normalized_split].append(
                {
                    "source": "codesearchnet",
                    "split": normalized_split,
                    "language": language,
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "func_name": row.get("func_name", ""),
                    "segmentation_method": segmentation_method,
                    "boundary_quality": boundary_quality,
                    "has_docstring": bool(docstring),
                    "code_segment_count": len(code_segments),
                    "dataset_version": dataset_version,
                    "segments": segments,
                }
            )

    outputs: Dict[str, Path] = {}
    for split_name, rows in buckets.items():
        out_path = processed_root(root) / output_subdir / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows)
        log(f"[preprocess] CodeSearchNet {split_name}: {count} functions -> {out_path}")
        outputs[split_name] = out_path

    for language in languages:
        stats = language_stats.get(language)
        if not stats:
            continue
        summary = ", ".join(f"{key}={value}" for key, value in sorted(stats.items()))
        log(f"[preprocess] CodeSearchNet stats {language}: {summary}")
    return outputs


def merge_processed_outputs(root: Path, per_dataset_outputs: Dict[str, Dict[str, Path]]) -> Dict[str, Path]:
    split_to_records: DefaultDict[str, List[Dict]] = defaultdict(list)
    for dataset_name, split_map in per_dataset_outputs.items():
        for split_name, path in split_map.items():
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        split_to_records[split_name].append(json.loads(line))

    outputs: Dict[str, Path] = {}
    for split_name, rows in split_to_records.items():
        out_path = processed_root(root) / "combined" / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows, training_mode="mixed")
        log(f"[merge] combined {split_name}: {count} records -> {out_path}")
        outputs[split_name] = out_path
    return outputs



def compute_split_counts(
    total: int,
    split_ratios: Sequence[int],
    split_names: Sequence[str] = SPLIT_NAMES,
) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if len(split_ratios) != len(split_names):
        raise ValueError("split_ratios and split_names must have the same length")
    if any(int(ratio) < 0 for ratio in split_ratios):
        raise ValueError("split ratios must be non-negative")

    denominator = sum(int(ratio) for ratio in split_ratios)
    if denominator <= 0:
        raise ValueError("split ratios must sum to a positive value")

    numerators = [total * int(ratio) for ratio in split_ratios]
    counts = [numerator // denominator for numerator in numerators]
    remainder = total - sum(counts)
    order = sorted(
        range(len(split_ratios)),
        key=lambda idx: (numerators[idx] % denominator, -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return {split_names[idx]: counts[idx] for idx in range(len(split_names))}


def iter_processed_dataset_paths(root: Path, dataset_name: str) -> Iterator[Path]:
    dataset_dir = processed_root(root) / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing processed dataset directory: {dataset_dir}")
    for path in sorted(dataset_dir.glob("*.jsonl")):
        yield path


def iter_dataset_pool_records(root: Path, dataset_name: str) -> Iterator[Dict]:
    for path in iter_processed_dataset_paths(root, dataset_name):
        yield from iter_jsonl_rows(path)


def dataset_pool_size(root: Path, dataset_name: str) -> int:
    return sum(1 for _ in iter_dataset_pool_records(root, dataset_name))


def stable_record_digest(record: Dict) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def select_balanced_dataset_subset(root: Path, dataset_name: str, sample_size: int) -> List[Dict]:
    if sample_size <= 0:
        return []

    heap: List[Tuple[int, str, int, Dict]] = []
    for index, record in enumerate(iter_dataset_pool_records(root, dataset_name)):
        digest = stable_record_digest(record)
        priority = int(digest, 16)
        entry = (-priority, digest, index, record)
        if len(heap) < sample_size:
            heapq.heappush(heap, entry)
            continue
        if priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    selected = [entry[3] for entry in sorted(heap, key=lambda item: (item[1], item[2]))]
    if len(selected) != sample_size:
        raise ValueError(
            f"Requested {sample_size} balanced samples from {dataset_name}, got {len(selected)}"
        )
    return selected


def build_balanced_combined_outputs(
    root: Path,
    datasets: Sequence[str],
    output_subdir: str = "combined_balanced_3_1_1",
    split_ratios: Sequence[int] = (3, 1, 1),
    equal_count: Optional[int] = None,
) -> Dict[str, Path]:
    datasets = [dataset for dataset in datasets]
    if not datasets:
        raise ValueError("At least one dataset is required for balanced combination")

    split_names = SPLIT_NAMES[: len(split_ratios)]
    if len(split_names) != len(split_ratios):
        raise ValueError("Balanced mode currently supports train/valid/test ratios only")

    pool_sizes = {dataset: dataset_pool_size(root, dataset) for dataset in datasets}
    if any(size <= 0 for size in pool_sizes.values()):
        raise ValueError(f"Balanced mode found an empty dataset pool: {pool_sizes}")

    target_count = min(pool_sizes.values()) if equal_count is None else int(equal_count)
    if target_count <= 0:
        raise ValueError("Balanced sample count must be positive")
    for dataset, size in pool_sizes.items():
        if target_count > size:
            raise ValueError(
                f"Requested {target_count} samples from {dataset}, but only {size} are available"
            )

    split_counts = compute_split_counts(target_count, split_ratios, split_names=split_names)
    split_to_records: DefaultDict[str, List[Dict]] = defaultdict(list)

    for dataset in datasets:
        selected_records = select_balanced_dataset_subset(root, dataset, target_count)
        cursor = 0
        for split_name in split_names:
            take = split_counts[split_name]
            chunk = selected_records[cursor : cursor + take]
            cursor += take
            for record in chunk:
                updated = dict(record)
                updated["original_split"] = record.get("split", "")
                updated["original_training_mode"] = record.get("training_mode", "")
                updated["split"] = split_name
                split_to_records[split_name].append(updated)
        if cursor != target_count:
            raise ValueError(f"Balanced split cursor mismatch for {dataset}: {cursor} != {target_count}")

    output_dir = processed_root(root) / output_subdir
    ensure_dir(output_dir)
    outputs: Dict[str, Path] = {}
    for split_name in split_names:
        rows = sorted(
            split_to_records[split_name],
            key=lambda row: (row.get("source", ""), stable_record_digest(row)),
        )
        out_path = output_dir / f"{split_name}.jsonl"
        count = write_jsonl(out_path, rows, training_mode="balanced_mixed")
        log(f"[balance] {split_name}: {count} records -> {out_path}")
        outputs[split_name] = out_path

    metadata = {
        "datasets": list(datasets),
        "pool_sizes": pool_sizes,
        "equal_count": target_count,
        "split_ratios": list(split_ratios),
        "split_counts": split_counts,
        "output_subdir": output_subdir,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    log(f"[balance] metadata -> {output_dir / 'metadata.json'}")
    return outputs


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def audit_downloads(root: Path) -> None:
    raw_root = root / "raw"
    ensure_dir(raw_root)
    for dataset_name in DATASET_ALIASES.values():
        path = raw_root / dataset_name
        exists = path.exists()
        size = directory_size(path) if exists else 0
        log(f"{dataset_name:15s} exists={str(exists):5s} size={human_size(size)} path={path}")

    gum_tok = dataset_root(root, "gum") / "disrpt" / "eng.erst.gum_train.tok"
    if gum_tok.exists():
        log(f"GUM EDU file present: {gum_tok}")

    juice_archive = dataset_root(root, "juice") / "juice-github-repos.tar.gz"
    if juice_archive.exists():
        log(f"JuICe archive present: {juice_archive} ({human_size(juice_archive.stat().st_size)})")

    combined_root = processed_root(root) / "combined"
    if combined_root.exists():
        for path in sorted(combined_root.glob("*.jsonl")):
            log(f"processed split: {path.name} ({human_size(path.stat().st_size)})")


def parse_dataset_list(items: Sequence[str]) -> List[str]:
    if not items or items == ["all"]:
        return list(DATASET_ALIASES.values())
    datasets = []
    for item in items:
        normalized = item.strip().lower()
        if normalized == "all":
            return list(DATASET_ALIASES.values())
        if normalized not in DATASET_ALIASES:
            raise ValueError(f"Unknown dataset: {item}")
        datasets.append(DATASET_ALIASES[normalized])
    return datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and preprocess semantic-boundary supervision corpora for LLaDA AdaBlock."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)

    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download raw datasets into the local boundary-data folder.")
    download_parser.add_argument("--datasets", nargs="+", default=["all"])
    download_parser.add_argument("--codesearchnet-languages", nargs="+", default=CODESEARCHNET_LANGUAGES)
    download_parser.add_argument("--skip-codesearchnet-extract", action="store_true")
    download_parser.add_argument("--juice-extract", action="store_true")

    preprocess_parser = subparsers.add_parser("preprocess", help="Convert raw datasets into JSONL files with semantic segments.")
    preprocess_parser.add_argument("--datasets", nargs="+", default=["all"])
    preprocess_parser.add_argument("--codesearchnet-languages", nargs="+", default=CODESEARCHNET_LANGUAGES)
    preprocess_parser.add_argument("--lean-include-formal-statements", action="store_true")
    preprocess_parser.add_argument("--proofnet-include-formal-statements", action="store_true")
    preprocess_parser.add_argument("--no-lean-proof-steps", action="store_true")
    preprocess_parser.add_argument("--juice-prefix-cell-type", action="store_true")
    preprocess_parser.add_argument("--no-combined", action="store_true")

    balance_parser = subparsers.add_parser(
        "balance",
        help="Rebuild a balanced train/valid/test combined set from complete per-dataset processed pools.",
    )
    balance_parser.add_argument("--datasets", nargs="+", default=["all"])
    balance_parser.add_argument("--output-subdir", type=str, default="combined_balanced_3_1_1")
    balance_parser.add_argument("--split-ratios", nargs=3, type=int, default=[3, 1, 1], metavar=("TRAIN", "VALID", "TEST"))
    balance_parser.add_argument("--equal-count", type=int, default=None)

    audit_parser = subparsers.add_parser("audit", help="Show which raw and processed datasets are available locally.")
    audit_parser.add_argument("--datasets", nargs="*", default=["all"])

    math_v2_parser = subparsers.add_parser("math-v2", help="Generate ProofNet + filtered Lean Workbook math_v2 corpora.")
    math_v2_parser.add_argument("--output-subdir", type=str, default="math_v2_full")
    math_v2_parser.add_argument("--pilot-output-subdir", type=str, default="math_v2_pilot")
    math_v2_parser.add_argument("--pilot-proofnet-limit", type=int, default=128)
    math_v2_parser.add_argument("--pilot-lean-limit", type=int, default=128)
    math_v2_parser.add_argument("--seed", type=int, default=0)
    math_v2_parser.add_argument("--include-formal-statements", action="store_true")
    math_v2_parser.add_argument("--no-pilot", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root)
    ensure_dir(root)

    if args.command == "download":
        datasets = parse_dataset_list(args.datasets)
        if "gum" in datasets:
            download_gum(root)
        if "codesearchnet" in datasets:
            download_codesearchnet(
                root,
                languages=args.codesearchnet_languages,
                extract_archives=not args.skip_codesearchnet_extract,
            )
        if "juice" in datasets:
            download_juice(root, extract_archive=args.juice_extract)
        if "lean_workbook" in datasets:
            download_lean_workbook(root)
        if "proofnet" in datasets:
            download_proofnet(root)
        audit_downloads(root)
        return

    if args.command == "preprocess":
        datasets = parse_dataset_list(args.datasets)
        per_dataset_outputs: Dict[str, Dict[str, Path]] = {}

        if "gum" in datasets:
            per_dataset_outputs["gum"] = preprocess_gum(root)
        if "codesearchnet" in datasets:
            per_dataset_outputs["codesearchnet"] = preprocess_codesearchnet(root, languages=args.codesearchnet_languages)
        if "juice" in datasets:
            per_dataset_outputs["juice"] = preprocess_juice(root, prefix_cell_type=args.juice_prefix_cell_type)
        if "lean_workbook" in datasets:
            per_dataset_outputs["lean_workbook"] = preprocess_lean_workbook(
                root,
                include_formal_statements=args.lean_include_formal_statements,
                include_proof_steps=not args.no_lean_proof_steps,
            )
        if "proofnet" in datasets:
            per_dataset_outputs["proofnet"] = preprocess_proofnet(
                root,
                include_formal_statements=args.proofnet_include_formal_statements,
            )

        if not args.no_combined and per_dataset_outputs:
            merge_processed_outputs(root, per_dataset_outputs)
        audit_downloads(root)
        return

    if args.command == "balance":
        datasets = parse_dataset_list(args.datasets)
        build_balanced_combined_outputs(
            root=root,
            datasets=datasets,
            output_subdir=args.output_subdir,
            split_ratios=tuple(args.split_ratios),
            equal_count=args.equal_count,
        )
        audit_downloads(root)
        return

    if args.command == "math-v2":
        pilot_output_subdir = None if args.no_pilot else args.pilot_output_subdir
        pilot_limits = None if args.no_pilot else {
            "proofnet": args.pilot_proofnet_limit,
            "lean_workbook": args.pilot_lean_limit,
        }
        summary = prepare_math_v2_corpora(
            root=root,
            output_subdir=args.output_subdir,
            pilot_output_subdir=pilot_output_subdir,
            pilot_limits=pilot_limits,
            seed=args.seed,
            include_formal_statements=args.include_formal_statements,
        )
        summary_path = processed_root_for_subdir(root, args.output_subdir) / f"math_v2_summary_{timestamp_slug()}.json"
        ensure_dir(summary_path.parent)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[preprocess][math_v2] summary -> {summary_path}")
        audit_downloads(root)
        return

    if args.command == "audit":
        audit_downloads(root)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
