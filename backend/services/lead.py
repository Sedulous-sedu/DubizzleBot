"""Service for validating qualified leads and managing local CSV persistence."""

import csv
import logging
import os
import re
import threading
from typing import Optional, List, Set

from backend.config import settings
from backend.models.lead import QualifiedLead, LeadDraft

logger = logging.getLogger(__name__)

class LeadService:
    """Manages thread-safe, idempotent lead qualification and CSV appending."""

    CSV_HEADERS = [
        "lead_id",
        "created_at",
        "user_id",
        "session_id",
        "name",
        "phone",
        "email",
        "min_budget_aed",
        "max_budget_aed",
        "interested_make",
        "interested_model",
        "interested_listing_id",
        "requirements",
        "booking_reference",
    ]

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or settings.LEADS_CSV_PATH
        self._lock = threading.Lock()
        self._ensure_csv_file()

    def _ensure_csv_file(self) -> None:
        """Creates the CSV file with headers if it does not exist."""
        with self._lock:
            if not os.path.exists(self.csv_path):
                parent_dir = os.path.dirname(self.csv_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                with open(self.csv_path, mode="w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(self.CSV_HEADERS)

    @staticmethod
    def validate_email(email: Optional[str]) -> bool:
        """Validates standard email format."""
        if not email or not isinstance(email, str):
            return False
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return bool(re.match(pattern, email.strip()))

    @staticmethod
    def validate_phone(phone: Optional[str]) -> bool:
        """
        Validates phone number format:
        Accepts international and local formats with optional +, digits, spaces, and hyphens.
        Requires at least 7 digits.
        """
        if not phone or not isinstance(phone, str):
            return False
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 7 or len(digits) > 15:
            return False
        pattern = r"^(\+?\d{1,4}[\s\-]?)?(\(?\d{2,4}\)?[\s\-]?)?[\d\s\-]{6,15}$"
        return bool(re.match(pattern, phone.strip()))

    def save_lead(self, lead: QualifiedLead) -> bool:
        """
        Appends a qualified lead to the CSV file.
        Enforces thread safety and idempotency by checking existing lead_id.
        """
        self._ensure_csv_file()
        with self._lock:
            existing_ids = self._read_existing_lead_ids_unlocked()
            if lead.lead_id in existing_ids:
                logger.info("Lead %s already exists in CSV, skipping duplicate write", lead.lead_id)
                return True

            with open(self.csv_path, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(lead.to_csv_dict())

            return True

    def get_leads(self) -> List[QualifiedLead]:
        """Reads all leads from CSV."""
        self._ensure_csv_file()
        leads: List[QualifiedLead] = []
        with self._lock:
            with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    min_budget = float(row["min_budget_aed"].replace(",", "")) if row["min_budget_aed"] else None
                    max_budget = float(row["max_budget_aed"].replace(",", "")) if row["max_budget_aed"] else None
                    listing_id = int(row["interested_listing_id"]) if row["interested_listing_id"] else None
                    leads.append(
                        QualifiedLead(
                            lead_id=row["lead_id"],
                            created_at=row["created_at"],
                            user_id=row["user_id"],
                            session_id=row["session_id"],
                            name=row["name"] or None,
                            phone=row["phone"] or None,
                            email=row["email"] or None,
                            min_budget_aed=min_budget,
                            max_budget_aed=max_budget,
                            interested_make=row["interested_make"] or None,
                            interested_model=row["interested_model"] or None,
                            interested_listing_id=listing_id,
                            requirements=row["requirements"] or None,
                            booking_reference=row["booking_reference"] or None,
                        )
                    )
        return leads

    def _read_existing_lead_ids_unlocked(self) -> Set[str]:
        if not os.path.exists(self.csv_path):
            return set()
        ids: Set[str] = set()
        try:
            with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lid = row.get("lead_id")
                    if lid:
                        ids.add(lid.strip())
        except Exception as e:
            logger.warning("Error reading lead IDs: %s", e)
        return ids
