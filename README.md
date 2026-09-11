# Online Voting System — Streamlit

Converted from the original Flask application to Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

Deploy `streamlit_app.py` from GitHub.

Optional admin secrets:

```toml
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "change-this-password"
```

Add these under the app's Streamlit Secrets settings.

## Important deployment note

The original Windows `venv311` folder and the original SQLite database containing voter records were intentionally not included in this deployment package.

The app creates a fresh `database/voting.db` on first run and seeds the four candidates.

SQLite storage on Streamlit Community Cloud is suitable for a demo/prototype, not a production election system. Cloud app storage can be reset when the app is redeployed/restarted.

The face camera flow was changed from OpenCV desktop camera windows (`VideoCapture`) to Streamlit's browser camera widget.
