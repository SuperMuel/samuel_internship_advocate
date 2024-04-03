# Samuel Internship Advocate

### Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Secrets

Create a `secrets.toml` file in [.streamlit](.streamlit) directory with the following content:

```toml
ANTHROPIC_API_KEY= "<Get you Anthropic API key from https://www.anthropic.com/api>"
SAMUEL_DETAILS_URL = "<Url where a text file containing the details of the intern is stored>"
```

### Usage

```bash
streamlit run app.py
```
