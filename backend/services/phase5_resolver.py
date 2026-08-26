"""Resolver for Phase 5 test-drive booking and lead qualification workflows."""

import calendar
import logging
import re
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
from zoneinfo import ZoneInfo

from backend.config import settings
from backend.models.booking import BookingDraft, WorkflowStatus
from backend.models.lead import LeadDraft
from backend.models.memory import SessionState
from backend.services.context_resolver import ContextResolver

logger = logging.getLogger(__name__)

class Phase5Action(str, Enum):
    """Actions identified by Phase5Resolver."""
    NOT_PHASE5 = "not_phase5"
    START_BOOKING = "start_booking"
    CONTINUE_BOOKING = "continue_booking"
    CONFIRM_BOOKING = "confirm_booking"
    CANCEL_BOOKING = "cancel_booking"
    START_LEAD = "start_lead"
    CONTINUE_LEAD = "continue_lead"
    CONFIRM_LEAD = "confirm_lead"
    CANCEL_LEAD = "cancel_lead"

class Phase5Resolution:
    """Structured result from Phase5Resolver."""
    def __init__(
        self,
        action: Phase5Action,
        date_val: Optional[date] = None,
        time_val: Optional[time] = None,
        is_ambiguous_time: bool = False,
        raw_date_str: Optional[str] = None,
        raw_time_str: Optional[str] = None,
        clarification_prompt: Optional[str] = None,
        extracted_name: Optional[str] = None,
        extracted_phone: Optional[str] = None,
        extracted_email: Optional[str] = None,
        extracted_min_budget: Optional[float] = None,
        extracted_max_budget: Optional[float] = None,
        extracted_requirements: Optional[str] = None,
    ):
        self.action = action
        self.date_val = date_val
        self.time_val = time_val
        self.is_ambiguous_time = is_ambiguous_time
        self.raw_date_str = raw_date_str
        self.raw_time_str = raw_time_str
        self.clarification_prompt = clarification_prompt
        self.extracted_name = extracted_name
        self.extracted_phone = extracted_phone
        self.extracted_email = extracted_email
        self.extracted_min_budget = extracted_min_budget
        self.extracted_max_budget = extracted_max_budget
        self.extracted_requirements = extracted_requirements

class Phase5Resolver:
    """Deterministic parser and intent resolver for Phase 5 workflows."""

    MONTH_MAP = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    WEEKDAY_MAP = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }

    def __init__(
        self,
        context_resolver: Optional[ContextResolver] = None,
        timezone_name: Optional[str] = None,
    ):
        self.context_resolver = context_resolver or ContextResolver()
        self.timezone_name = timezone_name or settings.BOOKING_TIMEZONE

    def get_timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except Exception:
            return ZoneInfo("Asia/Dubai")

    def evaluate(
        self,
        message: str,
        session: SessionState,
        current_time: Optional[datetime] = None,
    ) -> Phase5Resolution:
        """
        Evaluates the message in the context of active session workflows.
        Uses injected clock current_time (in Asia/Dubai) for relative date/time parsing.
        """
        tz = self.get_timezone()
        if current_time is None:
            now = datetime.now(tz)
        else:
            if current_time.tzinfo is None:
                now = current_time.replace(tzinfo=tz)
            else:
                now = current_time.astimezone(tz)

        msg_clean = message.strip()
        msg_lower = msg_clean.lower()

        # -------------------------------------------------------------
        # 1. Check Cancellation across active drafts
        # -------------------------------------------------------------
        if self._is_cancellation(msg_lower):
            if session.pending_booking is not None:
                return Phase5Resolution(action=Phase5Action.CANCEL_BOOKING)
            if session.pending_lead is not None:
                return Phase5Resolution(action=Phase5Action.CANCEL_LEAD)

        # -------------------------------------------------------------
        # 2. Check Active Pending Booking Continuation
        # -------------------------------------------------------------
        if session.pending_booking is not None:
            draft = session.pending_booking
            if draft.status == WorkflowStatus.AWAITING_CONFIRMATION:
                if self._is_confirmation(msg_lower):
                    return Phase5Resolution(action=Phase5Action.CONFIRM_BOOKING)
                # Check if user provides alternative date/time
                date_val, time_val, is_ambig, raw_d, raw_t = self.parse_datetime_expression(msg_lower, now)
                if date_val or time_val or is_ambig:
                    if is_ambig:
                        return Phase5Resolution(
                            action=Phase5Action.CONTINUE_BOOKING,
                            is_ambiguous_time=True,
                            clarification_prompt="Do you mean AM or PM? (Our business hours are 8:00 AM to 8:00 PM Asia/Dubai)."
                        )
                    return Phase5Resolution(
                        action=Phase5Action.CONTINUE_BOOKING,
                        date_val=date_val,
                        time_val=time_val,
                        raw_date_str=raw_d,
                        raw_time_str=raw_t
                    )
            elif draft.status == WorkflowStatus.COLLECTING:
                # If collecting, check if message provides date/time or contact
                date_val, time_val, is_ambig, raw_d, raw_t = self.parse_datetime_expression(msg_lower, now)
                name, phone, email = self._extract_contact_info(msg_clean)
                if date_val or time_val or is_ambig or name or phone or email:
                    if is_ambig:
                        return Phase5Resolution(
                            action=Phase5Action.CONTINUE_BOOKING,
                            is_ambiguous_time=True,
                            clarification_prompt="Do you mean AM or PM? (Our business hours are 8:00 AM to 8:00 PM Asia/Dubai)."
                        )
                    return Phase5Resolution(
                        action=Phase5Action.CONTINUE_BOOKING,
                        date_val=date_val,
                        time_val=time_val,
                        raw_date_str=raw_d,
                        raw_time_str=raw_t,
                        extracted_name=name,
                        extracted_phone=phone,
                        extracted_email=email,
                    )

        # -------------------------------------------------------------
        # 3. Check Active Pending Lead Continuation
        # -------------------------------------------------------------
        if session.pending_lead is not None:
            ldraft = session.pending_lead
            if ldraft.status == WorkflowStatus.AWAITING_CONFIRMATION:
                if self._is_confirmation(msg_lower):
                    return Phase5Resolution(action=Phase5Action.CONFIRM_LEAD)
            elif ldraft.status == WorkflowStatus.COLLECTING:
                name, phone, email = self._extract_contact_info(msg_clean)
                min_b, max_b = self._extract_budget(msg_lower)
                reqs = self._extract_requirements(msg_lower)
                if name or phone or email or min_b or max_b or reqs:
                    return Phase5Resolution(
                        action=Phase5Action.CONTINUE_LEAD,
                        extracted_name=name,
                        extracted_phone=phone,
                        extracted_email=email,
                        extracted_min_budget=min_b,
                        extracted_max_budget=max_b,
                        extracted_requirements=reqs,
                    )

        # -------------------------------------------------------------
        # 4. Check Explicit New Booking Action
        # -------------------------------------------------------------
        if self._is_booking_intent(msg_lower):
            date_val, time_val, is_ambig, raw_d, raw_t = self.parse_datetime_expression(msg_lower, now)
            if is_ambig:
                return Phase5Resolution(
                    action=Phase5Action.START_BOOKING,
                    is_ambiguous_time=True,
                    clarification_prompt="Do you mean AM or PM? (Our business hours are 8:00 AM to 8:00 PM Asia/Dubai)."
                )
            name, phone, email = self._extract_contact_info(msg_clean)
            return Phase5Resolution(
                action=Phase5Action.START_BOOKING,
                date_val=date_val,
                time_val=time_val,
                raw_date_str=raw_d,
                raw_time_str=raw_t,
                extracted_name=name,
                extracted_phone=phone,
                extracted_email=email,
            )

        # -------------------------------------------------------------
        # 5. Check Explicit New Lead Action
        # -------------------------------------------------------------
        if self._is_lead_intent(msg_lower):
            name, phone, email = self._extract_contact_info(msg_clean)
            min_b, max_b = self._extract_budget(msg_lower)
            reqs = self._extract_requirements(msg_lower)
            return Phase5Resolution(
                action=Phase5Action.START_LEAD,
                extracted_name=name,
                extracted_phone=phone,
                extracted_email=email,
                extracted_min_budget=min_b,
                extracted_max_budget=max_b,
                extracted_requirements=reqs,
            )

        return Phase5Resolution(action=Phase5Action.NOT_PHASE5)

    def parse_datetime_expression(
        self,
        text: str,
        now: datetime,
    ) -> Tuple[Optional[date], Optional[time], bool, Optional[str], Optional[str]]:
        """
        Extracts typed date, typed time, ambiguity flag, raw date str, and raw time str.
        Uses injected clock 'now' in Asia/Dubai for relative calculations.
        """
        text = text.lower().strip()
        date_val: Optional[date] = None
        time_val: Optional[time] = None
        is_ambiguous_time: bool = False
        raw_date_str: Optional[str] = None
        raw_time_str: Optional[str] = None

        # 1. Parse Date
        # A. Relative dates
        if re.search(r"\bday\s+after\s+tomorrow\b", text):
            date_val = now.date() + timedelta(days=2)
            raw_date_str = "day after tomorrow"
        elif re.search(r"\btomorrow\b", text):
            date_val = now.date() + timedelta(days=1)
            raw_date_str = "tomorrow"
        elif re.search(r"\btoday\b", text):
            date_val = now.date()
            raw_date_str = "today"
        else:
            # B. Bare or qualified weekday
            weekday_match = re.search(
                r"\b(?:on\s+|this\s+|next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
                text
            )
            if weekday_match:
                w_str = weekday_match.group(1).lower()
                target_w = self.WEEKDAY_MAP.get(w_str)
                if target_w is not None:
                    current_w = now.weekday()
                    days_ahead = (target_w - current_w) % 7
                    if days_ahead == 0:
                        # If today is Saturday and user asks Saturday, schedule next Saturday (7 days)
                        days_ahead = 7
                    date_val = now.date() + timedelta(days=days_ahead)
                    raw_date_str = weekday_match.group(0)

            # C. Month + Day (e.g. August 29, 29th August, 2026-08-29)
            if not date_val:
                # ISO date
                iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
                if iso_match:
                    try:
                        date_val = date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
                        raw_date_str = iso_match.group(0)
                    except ValueError:
                        pass

            if not date_val:
                # "August 29", "Aug 29th"
                month_first = re.search(
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b",
                    text
                )
                if month_first:
                    m_str = month_first.group(1).lower()
                    day_int = int(month_first.group(2))
                    year_int = int(month_first.group(3)) if month_first.group(3) else now.year
                    m_int = self.MONTH_MAP.get(m_str, now.month)
                    try:
                        candidate_date = date(year_int, m_int, day_int)
                        # If no year given and candidate is in the past, roll to next year
                        if not month_first.group(3) and candidate_date < now.date():
                            candidate_date = date(year_int + 1, m_int, day_int)
                        date_val = candidate_date
                        raw_date_str = month_first.group(0)
                    except ValueError:
                        pass

            if not date_val:
                # "29th August", "29 August 2026"
                day_first = re.search(
                    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+(\d{4}))?\b",
                    text
                )
                if day_first:
                    day_int = int(day_first.group(1))
                    m_str = day_first.group(2).lower()
                    year_int = int(day_first.group(3)) if day_first.group(3) else now.year
                    m_int = self.MONTH_MAP.get(m_str, now.month)
                    try:
                        candidate_date = date(year_int, m_int, day_int)
                        if not day_first.group(3) and candidate_date < now.date():
                            candidate_date = date(year_int + 1, m_int, day_int)
                        date_val = candidate_date
                        raw_date_str = day_first.group(0)
                    except ValueError:
                        pass

        # 2. Parse Time
        # A. Unambiguous 12-hour format: "3 PM", "3:30 PM", "11 AM", "11:00 am"
        time_12h = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
        if time_12h:
            hour = int(time_12h.group(1))
            minute = int(time_12h.group(2)) if time_12h.group(2) else 0
            meridiem = time_12h.group(3).lower()
            if meridiem == "pm" and hour < 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            try:
                time_val = time(hour, minute)
                raw_time_str = time_12h.group(0)
            except ValueError:
                pass

        # B. Unambiguous 24-hour format: "15:00", "08:30", "20:00"
        if not time_val and not is_ambiguous_time:
            time_24h = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
            if time_24h:
                hour = int(time_24h.group(1))
                minute = int(time_24h.group(2))
                try:
                    time_val = time(hour, minute)
                    raw_time_str = time_24h.group(0)
                except ValueError:
                    pass

        # C. Ambiguous 12-hour without AM/PM: "at 4", "at 3:00", "4 o'clock"
        if not time_val:
            ambig_match = re.search(r"\b(?:at\s+|around\s+)?(\d{1,2})(?::(\d{2}))?\s*(?:o'?clock)?\b", text)
            if ambig_match:
                # Check if it was part of an explicit price or year (e.g. 150000, 2020)
                matched_num = int(ambig_match.group(1))
                if 1 <= matched_num <= 12 and ("at" in text or "clock" in text or re.search(r"\b\d{1,2}\s*(?:o'?clock)\b", text)):
                    is_ambiguous_time = True
                    raw_time_str = ambig_match.group(0)

        return date_val, time_val, is_ambiguous_time, raw_date_str, raw_time_str

    @staticmethod
    def _is_booking_intent(text: str) -> bool:
        booking_keywords = [
            "test drive", "test-drive", "book a test", "book test",
            "book a viewing", "book viewing", "schedule viewing",
            "schedule a test", "schedule test", "view this car",
            "view the car", "book this", "test drive this", "test drive the",
        ]
        return any(k in text for k in booking_keywords)

    @staticmethod
    def _is_lead_intent(text: str) -> bool:
        lead_keywords = [
            "submit an enquiry", "submit enquiry", "register my enquiry",
            "register enquiry", "salesperson call me", "sales person call me",
            "contact me about buying", "interested in purchasing",
            "request a callback", "have someone call me", "dealer contact me",
        ]
        return any(k in text for k in lead_keywords)

    @staticmethod
    def _is_confirmation(text: str) -> bool:
        confirm_patterns = [
            r"^\s*yes\b", r"^\s*confirm\b", r"^\s*please confirm\b",
            r"^\s*yes please\b", r"^\s*proceed\b", r"^\s*sounds good\b",
            r"^\s*submit\b", r"^\s*yes confirm\b", r"^\s*ok\b", r"^\s*sure\b",
        ]
        return any(re.search(p, text) for p in confirm_patterns)

    @staticmethod
    def _is_cancellation(text: str) -> bool:
        cancel_patterns = [
            r"\bcancel\b", r"\bnever mind\b", r"\bnevermind\b",
            r"\bforget it\b", r"\bstop\b", r"\bno thanks\b",
        ]
        return any(re.search(p, text) for p in cancel_patterns)

    @staticmethod
    def _extract_contact_info(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extracts customer name, phone number, and email from user input."""
        name: Optional[str] = None
        phone: Optional[str] = None
        email: Optional[str] = None

        # Email
        email_match = re.search(r"\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b", text)
        if email_match:
            email = email_match.group(1).strip()

        # Phone: minimum 7 digits, supports +, dashes, spaces
        phone_match = re.search(r"(\+?\d[\d\s\-]{6,15}\d)", text)
        if phone_match:
            candidate_phone = phone_match.group(1).strip()
            digits = re.sub(r"\D", "", candidate_phone)
            # Filter out year/price numbers like 2018 or 150000
            if len(digits) >= 7 and not (len(digits) == 4 and digits.startswith("20")):
                phone = candidate_phone

        # Name: "My name is John Doe", "Name: Alex Smith", "I am John"
        name_match = re.search(r"\b(?:my\s+name\s+is|name\s*[:\-]|i\s+am)\s+([a-zA-Z\s]{2,30})\b", text, re.IGNORECASE)
        if name_match:
            candidate_name = name_match.group(1).strip()
            # Exclude common non-name words
            if candidate_name.lower() not in ("looking", "interested", "ready", "here"):
                name = candidate_name.title()

        return name, phone, email

    @staticmethod
    def _extract_budget(text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extracts minimum and maximum budget from text."""
        min_b: Optional[float] = None
        max_b: Optional[float] = None

        # Max budget: "under 150k", "under AED 150,000", "budget 120000", "max 200k"
        max_match = re.search(r"\b(?:under|below|max|up\s+to|budget\s+(?:of\s+)?(?:aed\s+)?)\s*(?:aed\s+)?(\d{1,3}(?:,\d{3})+|\d+)\s*(k)?\b", text)
        if max_match:
            val_str = max_match.group(1).replace(",", "")
            val = float(val_str)
            if max_match.group(2) and max_match.group(2).lower() == "k":
                val *= 1000.0
            max_b = val

        # Min budget: "above 50k", "from 60,000"
        min_match = re.search(r"\b(?:above|over|from|min|minimum)\s*(?:aed\s+)?(\d{1,3}(?:,\d{3})+|\d+)\s*(k)?\b", text)
        if min_match:
            val_str = min_match.group(1).replace(",", "")
            val = float(val_str)
            if min_match.group(2) and min_match.group(2).lower() == "k":
                val *= 1000.0
            min_b = val

        return min_b, max_b

    @staticmethod
    def _extract_requirements(text: str) -> Optional[str]:
        """Extracts vehicle preferences/requirements if mentioned."""
        reqs: List[str] = []
        if "gcc" in text:
            reqs.append("GCC Specs")
        if "suv" in text:
            reqs.append("SUV")
        if "sedan" in text:
            reqs.append("Sedan")
        if "coupe" in text:
            reqs.append("Coupe")
        if "warranty" in text:
            reqs.append("Under Warranty")
        if "low mileage" in text or "under 50k km" in text:
            reqs.append("Low Mileage")

        # Explicit interest: "interested in Nissan Patrol", "looking for Land Cruiser"
        interest_match = re.search(r"\b(?:interested in|looking for|enquiry for (?:a|an)?)\s+([a-zA-Z0-9\s]{3,30})\b", text)
        if interest_match:
            cand = interest_match.group(1).strip().title()
            if cand.lower() not in ("a car", "a vehicle", "an enquiry", "buying a car", "contact me", "a salesperson"):
                if cand not in reqs:
                    reqs.append(cand)

        return ", ".join(reqs) if reqs else None
