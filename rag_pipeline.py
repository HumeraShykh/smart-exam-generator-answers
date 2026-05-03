"""
RAG Pipeline — Multi-Agent Exam Paper Generator
CSC505, Sukkur IBA University

Handles:
  1. Syllabus / lecture notes / textbook chapters
  2. CLOs + PLOs (YAML or plain text)
  3. Past exam papers + assignments

Embedding backend : OpenAI-compatible API (LM Studio locally, or cloud)
LLM backend       : OpenAI-compatible API (LM Studio, OpenAI, Groq, OpenRouter, …)
Vector store      : ChromaDB (local, persistent)

Local (LM Studio):
  1. LM Studio → load chat + embedding models → Local Server → Start (http://localhost:1234)
Cloud: .env — OpenAI, Groq + Google embeddings, or Groq + Hugging Face Inference API (free HF token).
"""

import hashlib
import os
import re
import urllib.error
import urllib.request
import yaml
from pathlib import Path

# Same folder as app.py, and one level up (some users put .env in the outer "NLP Project" folder)
_PKG_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _PKG_ROOT / ".env"
_ENV_PATH_ALT = _PKG_ROOT.parent / ".env"


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Later files override earlier (inner project .env wins over parent folder)
    for path in (_ENV_PATH_ALT, _ENV_PATH):
        if path.is_file():
            load_dotenv(path, override=True)


_load_env_files()

# --- LangChain 0.3 correct import paths ---
# moved out of langchain in 0.3
from langchain_text_splitters import RecursiveCharacterTextSplitter
# moved to langchain-core in 0.3
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ---------------------------------------------------------------------------
# Configuration — env vars override defaults (good for Mac / cloud without LM Studio)
# ---------------------------------------------------------------------------
#
# .env example (OpenAI — paid):
#   EXAMGEN_OPENAI_API_KEY=sk-...
#   EXAMGEN_LLM_MODEL=gpt-4o-mini
#   EXAMGEN_EMBEDDING_MODEL=text-embedding-3-small
#   (base URL defaults to https://api.openai.com/v1 when a real key is set)
#
# Groq LLM (free): https://console.groq.com/keys — pair with embeddings via:
#   • GOOGLE_API_KEY (Gemini embeddings), or
#   • HF_TOKEN + Hugging Face Inference API (default model BAAI/bge-small-en-v1.5), or
#   • EXAMGEN_EMBEDDINGS_MODE=local (CPU; needs working PyTorch)
#


def _normalize_openai_base(url: str) -> str:
    u = url.strip().rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _load_runtime_config() -> None:
    """Sets module-level LLM / embedding config from environment."""
    global USE_LOCAL_EMBEDDINGS, USE_GOOGLE_EMBEDDINGS, USE_HF_INFERENCE_EMBEDDINGS
    global HF_EMBEDDING_MODEL, HF_INFERENCE_MODEL
    global LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    global EMBED_OPENAI_BASE_URL, EMBED_OPENAI_API_KEY, EMBEDDING_MODEL

    shared_key = (
        os.getenv("EXAMGEN_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    ).strip()
    shared_base = (
        os.getenv("EXAMGEN_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    ).strip()

    llm_base_raw = (os.getenv("EXAMGEN_LLM_BASE_URL") or shared_base or "").strip()
    # Groq must use gsk_ key — do not fall back to an OpenAI sk- key from the same .env
    if "groq.com" in llm_base_raw.lower():
        gs = (os.getenv("EXAMGEN_LLM_API_KEY") or "").strip()
        if not gs and shared_key.startswith("gsk_"):
            gs = shared_key
        llm_key = gs
    else:
        llm_key = (os.getenv("EXAMGEN_LLM_API_KEY") or shared_key or "").strip()

    embed_key_only = (os.getenv("EXAMGEN_EMBEDDING_API_KEY") or "").strip()
    embed_base_only = (os.getenv("EXAMGEN_EMBEDDING_BASE_URL") or "").strip()

    emb_mode = (os.getenv("EXAMGEN_EMBEDDINGS_MODE") or "").strip().lower()
    google_embed_key = (
        os.getenv("EXAMGEN_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ).strip()
    hf_api_token = (
        os.getenv("EXAMGEN_HF_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or ""
    ).strip()

    if llm_base_raw:
        LLM_BASE_URL = _normalize_openai_base(llm_base_raw)
    elif llm_key:
        LLM_BASE_URL = "https://api.openai.com/v1"
    else:
        LLM_BASE_URL = "http://localhost:1234/v1"
    LLM_API_KEY = llm_key or "lm-studio"

    llm_m = (os.getenv("EXAMGEN_LLM_MODEL") or "").strip()
    if not llm_m:
        llm_m = (
            "gpt-4o-mini"
            if LLM_BASE_URL.rstrip("/") == "https://api.openai.com/v1"
            else "llama-3.2-3b-instruct"
        )
    LLM_MODEL = llm_m

    is_groq = "groq.com" in LLM_BASE_URL.lower()
    USE_GOOGLE_EMBEDDINGS = False
    USE_LOCAL_EMBEDDINGS = False
    USE_HF_INFERENCE_EMBEDDINGS = False
    if emb_mode in ("google", "gemini"):
        USE_GOOGLE_EMBEDDINGS = True
    elif emb_mode in (
        "huggingface_api",
        "hf_api",
        "hf_cloud",
        "huggingface_inference",
    ):
        USE_HF_INFERENCE_EMBEDDINGS = True
    elif emb_mode in ("local", "cpu"):
        USE_LOCAL_EMBEDDINGS = True
    elif emb_mode in ("hf", "huggingface"):
        if hf_api_token:
            USE_HF_INFERENCE_EMBEDDINGS = True
        else:
            USE_LOCAL_EMBEDDINGS = True
    elif emb_mode in ("1", "true", "yes"):
        USE_LOCAL_EMBEDDINGS = True
    elif emb_mode in ("openai", "api", "remote"):
        pass
    else:
        # Auto: Groq — Google key > Hugging Face Inference token > local PyTorch
        if is_groq and google_embed_key:
            USE_GOOGLE_EMBEDDINGS = True
        elif is_groq and hf_api_token:
            USE_HF_INFERENCE_EMBEDDINGS = True
        elif is_groq:
            USE_LOCAL_EMBEDDINGS = True

    HF_EMBEDDING_MODEL = (
        os.getenv("EXAMGEN_HF_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    # Hugging Face Inference API (https://huggingface.co/settings/tokens) — free tier
    HF_INFERENCE_MODEL = (
        os.getenv("EXAMGEN_HF_INFERENCE_MODEL") or "BAAI/bge-small-en-v1.5"
    ).strip()

    embed_key = embed_key_only or shared_key or LLM_API_KEY
    embed_base_raw = embed_base_only or shared_base
    if embed_base_raw:
        EMBED_OPENAI_BASE_URL = _normalize_openai_base(embed_base_raw)
    elif embed_key_only or (shared_key and not is_groq):
        EMBED_OPENAI_BASE_URL = "https://api.openai.com/v1"
    else:
        EMBED_OPENAI_BASE_URL = LLM_BASE_URL
    EMBED_OPENAI_API_KEY = embed_key_only or shared_key or LLM_API_KEY

    emb_m = (os.getenv("EXAMGEN_EMBEDDING_MODEL") or "").strip()
    if not emb_m:
        emb_m = (
            "text-embedding-3-small"
            if EMBED_OPENAI_BASE_URL.rstrip("/") == "https://api.openai.com/v1"
            else "text-embedding-nomic-embed-text-v1.5"
        )
    EMBEDDING_MODEL = emb_m


_load_runtime_config()

CHROMA_DB_PATH = "./chroma_db"              # persistent local vector store

# Collection names — kept separate so agents can query them independently
COLLECTION_SYLLABUS = "syllabus_notes"
COLLECTION_CLO_PLO = "clo_plo_outcomes"
COLLECTION_PAST_PAPERS = "past_papers_assignments"

# Chunking parameters
CHUNK_SIZE = 200    # small = fast embedding
CHUNK_OVERLAP = 30


def _hf_inference_token() -> str:
    return (
        os.getenv("EXAMGEN_HF_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or ""
    ).strip()


def _validate_hf_inference_token(tok: str) -> None:
    """Fail fast with a clear message when .env still has a placeholder or invalid HF token."""
    if not tok:
        raise RuntimeError(
            "HF_TOKEN is empty. Add it to .env next to app.py — "
            "https://huggingface.co/settings/tokens"
        )
    low = tok.lower()
    if any(
        x in low
        for x in ("replace", "your-token", "paste_here", "yahan", "example", "dummy")
    ):
        raise RuntimeError(
            "HF_TOKEN is still a placeholder in .env. Copy a real token from "
            "https://huggingface.co/settings/tokens (full string; usually starts with hf_)."
        )
    if not tok.startswith("hf_"):
        raise RuntimeError(
            "HF_TOKEN must be a Hugging Face access token starting with hf_. "
            "Create: huggingface.co → Settings → Access Tokens → New token (Read is enough)."
        )
    if len(tok) < 30:
        raise RuntimeError(
            "HF_TOKEN looks truncated or wrong (too short). Paste the full token from HF settings."
        )


class HuggingFaceInferenceEmbeddingsBatched:
    """
    Hugging Face embeddings via huggingface_hub.InferenceClient (Router API).
    Avoids deprecated api-inference URLs that return empty/non-JSON bodies.
    """

    def __init__(self, api_key: str, model_name: str, batch_size: int = 12):
        from huggingface_hub import InferenceClient

        self._model = model_name
        self._batch_size = max(1, batch_size)
        self._client = InferenceClient(model=model_name, token=api_key)

    def _vectors_from_array(self, arr: object, batch_len: int) -> list[list[float]]:
        import numpy as np

        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            if batch_len != 1:
                raise RuntimeError(
                    "Hugging Face returned one flat vector for a multi-text batch."
                )
            return [a.tolist()]
        if a.ndim == 2:
            if a.shape[0] == batch_len:
                return a.tolist()
            # Single text but token-level matrix → mean-pool to one vector
            if batch_len == 1:
                return [a.mean(axis=0).tolist()]
            raise RuntimeError(
                f"Hugging Face embedding rows ({a.shape[0]}) != batch size ({batch_len})."
            )
        raise RuntimeError(f"Unexpected embedding shape: {getattr(a, 'shape', None)}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import time

        from huggingface_hub.errors import HfHubHTTPError

        texts = [t.replace("\n", " ") for t in texts]
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            last_err: Exception | None = None
            for attempt in range(5):
                try:
                    arr = self._client.feature_extraction(batch)
                    all_vecs.extend(self._vectors_from_array(arr, len(batch)))
                    break
                except HfHubHTTPError as e:
                    sc = getattr(getattr(e, "response", None), "status_code", None)
                    if sc == 401:
                        last_err = RuntimeError(
                            "Hugging Face returned 401 — invalid or expired HF_TOKEN. "
                            "Fix: huggingface.co/settings/tokens → create New token → paste "
                            "into .env as HF_TOKEN=hf_... (no quotes). "
                            "Fine-grained tokens must allow Inference / Make calls to Inference Providers."
                        )
                    else:
                        detail = str(e)
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            body = getattr(resp, "text", "") or ""
                            detail = f"{detail} | body[:300]={body[:300]!r}"
                        last_err = RuntimeError(f"Hugging Face HTTP error: {detail}")
                    time.sleep(min(2 ** attempt, 12))
                except Exception as e:
                    last_err = e
                    time.sleep(min(2 ** attempt, 12))
            else:
                raise RuntimeError(
                    f"Hugging Face embeddings failed after retries: {last_err}"
                ) from last_err
        return all_vecs

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _ensure_llm_server_reachable() -> None:
    """
    Fail fast if nothing is listening at LLM_BASE_URL.
    OpenAI's client only reports a vague "Connection error." when the TCP
    connection fails; this gives a direct fix (start LM Studio or change URL).
    """
    local = "localhost" in LLM_BASE_URL or "127.0.0.1" in LLM_BASE_URL
    has_cloud_key = bool(
        (
            os.getenv("EXAMGEN_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("EXAMGEN_LLM_API_KEY")
            or ""
        ).strip()
    )
    if local and not has_cloud_key and LLM_API_KEY == "lm-studio":
        exists_here = _ENV_PATH.is_file()
        exists_parent = _ENV_PATH_ALT.is_file()
        raise RuntimeError(
            "No API key loaded — still using LM Studio (localhost:1234). "
            "Create a file named .env in this folder:\n"
            f"  {_PKG_ROOT}\n"
            "with at least one line (no quotes):\n"
            "  EXAMGEN_OPENAI_API_KEY=sk-...your-key...\n"
            "Or for free LLM (Groq) + local embeddings: EXAMGEN_LLM_API_KEY=gsk_... "
            "and EXAMGEN_LLM_BASE_URL=https://api.groq.com/openai/v1\n"
            "You can also use OPENAI_API_KEY=sk-... (same effect).\n"
            f"Checked: {_ENV_PATH} exists={exists_here}, {_ENV_PATH_ALT} exists={exists_parent}. "
            "After saving .env, restart Streamlit (stop terminal, run streamlit again)."
        )

    url = f"{LLM_BASE_URL.rstrip('/')}/models"
    origin = LLM_BASE_URL.rsplit("/v1", 1)[0] or LLM_BASE_URL
    headers = {"User-Agent": "ExamGen-RAG/1.0"}
    if LLM_API_KEY and LLM_API_KEY != "lm-studio":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status >= 500:
                raise OSError(f"HTTP {resp.status} from {url!r}")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return
        raise RuntimeError(
            f"LLM server at {origin!r} returned HTTP {e.code} for {url!r}. "
            "Check the server logs or API key if using a cloud endpoint."
        ) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"Cannot connect to the LLM/embedding API at {origin!r} (requested {url!r}). "
            "If you meant to use OpenAI cloud, put EXAMGEN_OPENAI_API_KEY or OPENAI_API_KEY in "
            f".env next to app.py ({_PKG_ROOT}) and restart Streamlit. "
            f"Underlying error: {e!s}"
        ) from e


# ---------------------------------------------------------------------------
# 1. Embeddings + LLM setup
# ---------------------------------------------------------------------------


def get_embeddings():
    """
    Embeddings: OpenAI-compatible API, Google AI Studio, Hugging Face Inference API
    (free token tier), or local HuggingFace (CPU).
    """
    if USE_HF_INFERENCE_EMBEDDINGS:
        tok = _hf_inference_token()
        if not tok:
            raise RuntimeError(
                "Hugging Face Inference embeddings need a token. Create one (free): "
                "https://huggingface.co/settings/tokens — then set HF_TOKEN or "
                "HUGGINGFACEHUB_API_TOKEN or EXAMGEN_HF_API_TOKEN in .env"
            )
        _validate_hf_inference_token(tok)
        return HuggingFaceInferenceEmbeddingsBatched(
            api_key=tok,
            model_name=HF_INFERENCE_MODEL,
            batch_size=int(os.getenv("EXAMGEN_HF_INFERENCE_BATCH", "12")),
        )
    if USE_GOOGLE_EMBEDDINGS:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "Google embeddings need: pip install langchain-google-genai"
            ) from e
        gkey = (
            os.getenv("EXAMGEN_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
        if not gkey:
            raise RuntimeError(
                "Set EXAMGEN_GOOGLE_API_KEY or GOOGLE_API_KEY "
                "(free: https://aistudio.google.com/app/apikey )"
            )
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gkey,
        )
    if USE_LOCAL_EMBEDDINGS:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "Local embeddings need: pip install sentence-transformers "
                "(see requirements.txt)"
            ) from e
        return HuggingFaceEmbeddings(
            model_name=HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=EMBED_OPENAI_BASE_URL,
        openai_api_key=EMBED_OPENAI_API_KEY,
        check_embedding_ctx_length=False,
    )


def get_llm() -> ChatOpenAI:
    """
    Chat model via any OpenAI-compatible API (OpenAI, Groq, LM Studio, OpenRouter, …).
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# 2. Document loaders — handles PDF, TXT, PPTX
# ---------------------------------------------------------------------------

def load_documents_from_path(path: str) -> list[Document]:
    """
    Loads all supported files from a file path or directory.
    Supported: .pdf, .txt, .md, .pptx
    """
    p = Path(path)
    docs = []

    if p.is_dir():
        for file in p.rglob("*"):
            docs.extend(_load_single_file(file))
    elif p.is_file():
        docs.extend(_load_single_file(p))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    print(f"  Loaded {len(docs)} raw document(s) from: {path}")
    return docs


def _load_single_file(file: Path) -> list[Document]:
    """Dispatches to the right loader based on file extension."""
    ext = file.suffix.lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(str(file)).load()
        elif ext in (".txt", ".md"):
            return TextLoader(str(file), encoding="utf-8").load()
        elif ext in (".pptx", ".ppt"):
            return UnstructuredPowerPointLoader(str(file)).load()
        else:
            return []   # skip unsupported files silently
    except Exception as e:
        print(f"  [WARNING] Could not load {file.name}: {e}")
        return []


# ---------------------------------------------------------------------------
# 3. Text cleaning + chunking
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Fixes PDF extraction artifacts where every character or word is
    separated by extra spaces/newlines (common in scanned or slide PDFs).
    Steps:
      1. Replace runs of whitespace (spaces, newlines, tabs) with a single space
      2. Fix common ligature replacements (fi, fl etc.)
      3. Strip leading/trailing whitespace
    """
    # Fix common PDF ligature artifacts first
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
    # These PDFs have a newline after every single word — collapse them into spaces
    # Strategy: a single newline between two non-empty lines = join with space
    # A blank line (double newline) = real paragraph break, keep it
    # protect real paragraph breaks
    text = re.sub(r'\n{2,}', '<<PARA>>', text)
    # collapse all remaining newlines to space
    text = re.sub(r'\n', ' ', text)
    text = text.replace('<<PARA>>', '\n\n')       # restore paragraph breaks
    text = re.sub(r' {2,}', ' ', text)             # collapse multiple spaces
    return text.strip()


def clean_documents(docs: list[Document]) -> list[Document]:
    """Applies clean_text() to every document's page_content in place."""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    # Remove documents that became empty after cleaning
    cleaned = [d for d in docs if len(d.page_content.strip()) > 30]
    removed = len(docs) - len(cleaned)
    if removed:
        print(f"  Removed {removed} empty/short doc(s) after cleaning.")
    return cleaned


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Splits documents into overlapping chunks using recursive character splitter.
    Preserves metadata (source, page) from the original documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunk(s).")
    return chunks


# ---------------------------------------------------------------------------
# 4. CLO / PLO ingestion
# ---------------------------------------------------------------------------

def _yaml_list_by_key(data: dict, *key_names: str) -> list:
    """Return the first list value whose key matches one of key_names (case-insensitive)."""
    if not isinstance(data, dict):
        return []
    lowered = {str(k).lower(): k for k in data}
    for want in key_names:
        w = want.lower()
        if w in lowered:
            val = data[lowered[w]]
            return val if isinstance(val, list) else []
    return []


def _outcome_fields(item: dict) -> tuple[str, str] | None:
    """Resolve code + description from common YAML shapes."""
    if not isinstance(item, dict):
        return None
    code = item.get("code") or item.get("id") or item.get("name")
    desc = (
        item.get("description")
        or item.get("desc")
        or item.get("text")
        or item.get("statement")
        or item.get("title")
    )
    if code is None or desc is None:
        return None
    return str(code).strip(), str(desc).strip()


def load_clo_plo(path: str) -> list[Document]:
    """
    Loads CLOs and PLOs from a YAML file or plain text file.

    YAML format expected (yaml):
        CLOs:
          - code: CLO-1
            description: "Apply transformer architectures to NLP tasks"
          - code: CLO-2
            ...
        PLOs:
          - code: PLO-3
            description: "Apply engineering knowledge to solve problems"

    Also accepts case variants (e.g. clos/plos) and item aliases (id, text, …).

    Each CLO/PLO becomes its own Document so Agent 4 can embed + query them
    independently from the main knowledge base.
    """
    p = Path(path)
    docs = []

    if p.suffix in (".yaml", ".yml"):
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(
                f"CLO/PLO YAML is empty or invalid: {p}"
            )

        if not isinstance(data, dict):
            raise ValueError(
                f"CLO/PLO YAML must be a mapping (top-level keys CLOs and PLOs). Got: {type(data).__name__}"
            )

        clo_items = _yaml_list_by_key(data, "clos", "clo", "course_learning_outcomes", "learning_outcomes")
        plo_items = _yaml_list_by_key(data, "plos", "plo", "program_learning_outcomes")

        # Backward-compatible exact keys if not matched above
        if not clo_items:
            clo_items = data.get("CLOs") or []
            if not isinstance(clo_items, list):
                clo_items = []
        if not plo_items:
            plo_items = data.get("PLOs") or []
            if not isinstance(plo_items, list):
                plo_items = []

        for clo in clo_items:
            pair = _outcome_fields(clo) if isinstance(clo, dict) else None
            if not pair:
                continue
            code, description = pair
            docs.append(Document(
                page_content=f"CLO {code}: {description}",
                metadata={"type": "CLO", "code": code, "source": str(p)},
            ))
        for plo in plo_items:
            pair = _outcome_fields(plo) if isinstance(plo, dict) else None
            if not pair:
                continue
            code, description = pair
            docs.append(Document(
                page_content=f"PLO {code}: {description}",
                metadata={"type": "PLO", "code": code, "source": str(p)},
            ))

        n_clo = sum(1 for d in docs if d.metadata.get("type") == "CLO")
        n_plo = sum(1 for d in docs if d.metadata.get("type") == "PLO")
        if n_clo == 0 or n_plo == 0:
            keys_preview = list(data.keys())[:25]
            raise ValueError(
                "CLO/PLO YAML must define at least one CLO and one PLO. "
                "Use top-level lists under keys like 'CLOs' and 'PLOs', each item with "
                "'code' and 'description' (see data/clo_plo.yaml in this project). "
                f"Parsed CLO count={n_clo}, PLO count={n_plo}. "
                f"Top-level YAML keys found: {keys_preview}. "
                "Agent config files (e.g. role/goal/backstory per agent) are not valid here."
            )

    elif p.suffix in (".txt", ".md"):
        # Plain text: each non-empty line treated as one CLO/PLO statement
        with open(p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, line in enumerate(lines):
            docs.append(Document(
                page_content=line,
                metadata={"type": "CLO_PLO_raw", "index": i, "source": str(p)},
            ))
    else:
        raise ValueError(f"Unsupported CLO/PLO file format: {p.suffix}")

    print(f"  Loaded {len(docs)} CLO/PLO entries from: {path}")
    return docs


# ---------------------------------------------------------------------------
# 5. Vector store helpers
# ---------------------------------------------------------------------------

def _get_or_create_collection(collection_name: str, embeddings) -> Chroma:
    """Returns a persistent ChromaDB collection (creates it if it doesn't exist)."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


def _doc_id(doc: Document, index: int) -> str:
    """
    Unique ID combining content hash + index position.
    Pure content-hash fails when multiple chunks have identical text
    (e.g. repeated slide headers) — adding the index makes every ID unique.
    """
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
    source = doc.metadata.get("source", "unknown")
    return hashlib.md5(f"{source}::{index}::{content_hash}".encode()).hexdigest()


def ingest_to_collection(
    docs: list[Document],
    collection_name: str,
    embeddings,
    deduplicate: bool = True,
) -> Chroma:
    """
    Embeds documents and upserts them into a named ChromaDB collection.
    If deduplicate=True, skips chunks already present (based on source+index hash).
    """
    store = _get_or_create_collection(collection_name, embeddings)

    # Generate unique IDs first
    all_ids = [_doc_id(d, i) for i, d in enumerate(docs)]

    if deduplicate:
        existing_ids = set(store._collection.get()["ids"])
        filtered = [(doc, id_) for doc, id_ in zip(
            docs, all_ids) if id_ not in existing_ids]
        skipped = len(docs) - len(filtered)
        if skipped:
            print(f"  Skipped {skipped} already-ingested chunk(s).")
        if not filtered:
            print(f"  Nothing new to ingest into '{collection_name}'.")
            return store
        docs, all_ids = zip(*filtered)
        docs, all_ids = list(docs), list(all_ids)

    if docs:
        store.add_documents(docs, ids=all_ids)
        print(
            f"  Ingested {len(docs)} chunk(s) into collection '{collection_name}'.")

    return store


# ---------------------------------------------------------------------------
# 6. Retrieval helpers (used by agents at query time)
# ---------------------------------------------------------------------------

def get_retriever(collection_name: str, embeddings, k: int = 3):
    """
    Returns a LangChain retriever for the given collection.
    Agents call retriever.invoke("your query") to get relevant chunks.
    """
    store = _get_or_create_collection(collection_name, embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, collection_name: str, embeddings, k: int = 5) -> str:
    """
    Convenience function: retrieves top-k chunks and returns them as a
    single formatted string ready to inject into an LLM prompt.
    """
    retriever = get_retriever(collection_name, embeddings, k=k)
    results = retriever.invoke(query)
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in results
    )


def get_all_clo_plo_docs(embeddings) -> list[Document]:
    """
    Returns ALL CLO/PLO documents from the vector store as a flat list.
    Agent 4 uses this to compute cosine similarity against every question.
    """
    store = _get_or_create_collection(COLLECTION_CLO_PLO, embeddings)
    result = store._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]


# ---------------------------------------------------------------------------
# 7. Top-level pipeline function
# ---------------------------------------------------------------------------

def build_rag_pipeline(
    syllabus_path: str,
    clo_plo_path: str,
    past_papers_path: str,
) -> dict:
    """
    Master function that runs the full ingestion pipeline.

    Parameters
    ----------
    syllabus_path     : path to folder or file containing lecture notes / slides / textbook chapters
    clo_plo_path      : path to YAML or TXT file with CLOs and PLOs
    past_papers_path  : path to folder or file containing past exam papers / assignments

    Returns
    -------
    dict with keys: embeddings, llm, retrievers (one per collection)
    """
    print("\n=== RAG Pipeline: Starting Ingestion ===\n")

    _ensure_llm_server_reachable()
    embeddings = get_embeddings()
    llm = get_llm()

    # --- Input 1: Syllabus / lecture notes ---
    print("[1/3] Processing syllabus and lecture notes...")
    syllabus_docs = load_documents_from_path(syllabus_path)
    syllabus_docs = clean_documents(syllabus_docs)
    syllabus_chunks = chunk_documents(syllabus_docs)
    ingest_to_collection(syllabus_chunks, COLLECTION_SYLLABUS, embeddings)

    # --- Input 2: CLOs + PLOs ---
    print("\n[2/3] Processing CLOs and PLOs...")
    clo_plo_docs = load_clo_plo(clo_plo_path)
    # CLOs/PLOs are short statements — no chunking needed
    ingest_to_collection(clo_plo_docs, COLLECTION_CLO_PLO,
                         embeddings, deduplicate=True)

    # --- Input 3: Past papers + assignments ---
    print("\n[3/3] Processing past exam papers and assignments...")
    past_docs = load_documents_from_path(past_papers_path)
    past_docs = clean_documents(past_docs)
    past_chunks = chunk_documents(past_docs)
    ingest_to_collection(past_chunks, COLLECTION_PAST_PAPERS, embeddings)

    print("\n=== RAG Pipeline: Ingestion Complete ===\n")

    # Return retrievers so agents can use them immediately
    return {
        "embeddings": embeddings,
        "llm": llm,
        "retrievers": {
            "syllabus":    get_retriever(COLLECTION_SYLLABUS,    embeddings),
            "clo_plo":     get_retriever(COLLECTION_CLO_PLO,     embeddings),
            "past_papers": get_retriever(COLLECTION_PAST_PAPERS, embeddings),
        },
    }


# ---------------------------------------------------------------------------
# 8. Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick test — replace the paths below with your actual CSC505 material paths.

    Folder structure example:
        data/
          syllabus/       <- lecture slides (PDF/PPTX), notes (TXT/MD)
          clo_plo.yaml    <- CLOs and PLOs
          past_papers/    <- past exam PDFs and assignment PDFs
    """

    pipeline = build_rag_pipeline(
        syllabus_path="data/Lectures",
        clo_plo_path="data/clo_plo.yaml",
        past_papers_path="data/Assessment",
    )

    # --- Test retrieval from each collection ---
    embeddings = pipeline["embeddings"]

    print("\n--- Test: Syllabus retrieval ---")
    print(retrieve_context(
        "What are the main topics covered in transformer architectures?",
        COLLECTION_SYLLABUS, embeddings, k=3
    ))

    print("\n--- Test: CLO/PLO retrieval ---")
    print(retrieve_context(
        "Apply deep learning models to solve NLP problems",
        COLLECTION_CLO_PLO, embeddings, k=3
    ))

    print("\n--- Test: Past papers retrieval ---")
    print(retrieve_context(
        "Short answer question about BERT fine-tuning",
        COLLECTION_PAST_PAPERS, embeddings, k=3
    ))

    print("\n--- Test: All CLO/PLO docs (for Agent 4) ---")
    all_outcomes = get_all_clo_plo_docs(embeddings)
    for doc in all_outcomes:
        print(
            f"  [{doc.metadata.get('type')} {doc.metadata.get('code', '')}] {doc.page_content}")
