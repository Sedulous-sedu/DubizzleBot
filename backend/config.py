"""Application configuration settings loader."""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Central configuration parameters for backend service."""
    PROJECT_NAME: str = "DubizzleBot API"
    VERSION: str = "0.1.0"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./dubizzle_bot.db")
    DATASET_PATH: str = os.getenv("DATASET_PATH", "Copy_of_sample_cars_dataset.xlsx")
    LEADS_CSV_PATH: str = os.getenv("LEADS_CSV_PATH", "leads.csv")

settings = Settings()
