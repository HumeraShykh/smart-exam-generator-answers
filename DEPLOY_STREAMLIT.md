# Deploy ExamGen on Streamlit Community Cloud (free)

This project runs at repo root with main file `app.py`. API keys must **never** be committed; use Streamlit **Secrets**.

## 1. Repository hygiene

- `.gitignore` excludes `.env`, `app_data/`, virtualenvs, and Chroma data.
- Push only code plus `.env.example` (no real tokens).

## 2. Create the GitHub repo and push

From this folder (`NLP Project/` inside your workspace — the one that contains `app.py`):

```bash
cd "/path/to/NLP Project"
git init
git add .
git status   # confirm .env is NOT listed
git commit -m "Initial commit: ExamGen Streamlit app"
```

Create an empty repository on GitHub (no README if you want a clean first push), then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 3. Connect Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
2. **New app** → select the repository and branch.
3. **Main file path:** `app.py` (if the GitHub repo root is this project folder). If you put the app in a subfolder, set path to `subfolder/app.py`.
4. **Deploy.**

## 4. Secrets (mirror your local `.env`)

In the deployed app: **Settings → Secrets**, paste **TOML** with the same variable names you use locally. Example (replace placeholders):

```toml
EXAMGEN_LLM_BASE_URL = "https://api.groq.com/openai/v1"
EXAMGEN_LLM_API_KEY = "gsk_your_groq_key"
EXAMGEN_LLM_MODEL = "llama-3.3-70b-versatile"
EXAMGEN_EMBEDDINGS_MODE = "huggingface_api"
HF_TOKEN = "hf_your_huggingface_token"
```

Optional (Google embeddings instead of HF):

```toml
GOOGLE_API_KEY = "your_google_ai_studio_key"
```

The app maps `st.secrets` into `os.environ` before the RAG stack loads (`app.py`), so these names must match what [`rag_pipeline.py`](rag_pipeline.py) expects.

## 5. Public deploy settings

[`.streamlit/config.toml`](.streamlit/config.toml) uses `enableXsrfProtection = true` and `maxUploadSize = 100` (MB) for safer defaults on a shared URL.

If **local** uploads return **403** when testing with Cursor preview or odd hosts, run Streamlit with `--server.address 127.0.0.1` and a normal browser, or temporarily add in **local only** a dev `config.toml` (do not commit relaxed XSRF for production).

## 6. Share the link

After deploy, Cloud shows a URL like `https://<app-name>.streamlit.app`. Share that link.

**Limits (free tier):** apps may sleep when idle; cold starts; heavy RAG runs may hit memory/time limits. API quotas still apply to Groq/Hugging Face/Google keys.

## Alternative: Hugging Face Spaces

Create a **Space** with SDK **Streamlit**, sync this repo, and add the same keys under Space **Secrets**. Point main file to `app.py`.
