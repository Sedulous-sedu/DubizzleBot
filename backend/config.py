"""Application configuration settings loader."""

import os
from dotenv import load_dotenv

# Load .env with override=True so local configuration takes precedence over stale shell variables
load_dotenv(override=True)

class Settings:
    """Central configuration parameters for backend service."""
    PROJECT_NAME: str = "DubizzleBot API"
    VERSION: str = "0.1.0"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini/gemini-3.6-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dubizzle_bot.db")
    DATASET_PATH: str = os.getenv("DATASET_PATH", "Copy_of_sample_cars_dataset.xlsx")
    LEADS_CSV_PATH: str = os.getenv("LEADS_CSV_PATH", "leads.csv")

settings = Settings()
