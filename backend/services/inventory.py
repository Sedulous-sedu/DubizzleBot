"""Inventory search and dataset retrieval module using pandas and regex enrichment."""

import re
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from backend.config import settings
from backend.models.car import CarFilter, CarListing

class InventoryService:
    """Service managing car inventory dataset search, deterministic filtering, and regex enrichment."""

    def __init__(self, dataset_path: str = settings.DATASET_PATH):
        self.dataset_path = dataset_path
        self._df: Optional[pd.DataFrame] = None
        self._load_and_enrich_dataset()

    def _load_and_enrich_dataset(self):
        """Loads dataset reproducibly, preserving original fields and enriching with derived attributes and provenance."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path}")

        xl = pd.ExcelFile(self.dataset_path)
        sheet_name = "cleaned dataset" if "cleaned dataset" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet_name)

        # Preserve original source display fields intact
        df['Listing_ID'] = df['Listing_ID'].astype(int)
        df['year'] = df['year'].astype(int)
        df['make'] = df['make'].astype(str)
        df['model'] = df['model'].astype(str)
        df['trim'] = df['trim'].astype(str)
        df['title'] = df['title'].astype(str)
        df['description'] = df['description'].astype(str)
        df['photo_url'] = df['photo_url'].astype(str)

        # Create normalized internal search fields (lowercased & stripped for matching)
        df['_make_clean'] = df['make'].str.strip().str.lower()
        df['_model_clean'] = df['model'].str.strip().str.lower()
        df['_trim_clean'] = df['trim'].str.strip().str.lower()
        df['_title_clean'] = df['title'].str.strip().str.lower()
        df['_description_clean'] = df['description'].str.strip().str.lower()

        # Lists for derived fields and provenance
        prices = []
        monthlies = []
        mileages = []
        specs = []
        has_pos_warranties = []
        warranty_statuses = []
        body_types = []
        provenances = []

        for _, row in df.iterrows():
            text = f"{row['title']}\n{row['description']}"
            provenance = {}

            # 1. Cash Price
            price_val, price_snip = self._extract_cash_price(text)
            prices.append(price_val)
            if price_snip:
                provenance['price_aed'] = {"value": price_val, "source_snippet": price_snip}

            # 2. Monthly Payment
            monthly_val, monthly_snip = self._extract_monthly_payment(text)
            monthlies.append(monthly_val)
            if monthly_snip:
                provenance['monthly_payment_aed'] = {"value": monthly_val, "source_snippet": monthly_snip}

            # 3. Mileage
            mileage_val, mileage_snip = self._extract_mileage(text)
            mileages.append(mileage_val)
            if mileage_snip:
                provenance['mileage_km'] = {"value": mileage_val, "source_snippet": mileage_snip}

            # 4. Regional Specs
            spec_val, spec_snip = self._extract_regional_specs(text)
            specs.append(spec_val)
            if spec_snip:
                provenance['regional_specs'] = {"value": spec_val, "source_snippet": spec_snip}

            # 5. Warranty
            pos_warr, warr_status, warr_snip = self._extract_warranty(text)
            has_pos_warranties.append(pos_warr)
            warranty_statuses.append(warr_status)
            if warr_snip:
                provenance['warranty'] = {
                    "has_positive_warranty": pos_warr,
                    "status": warr_status,
                    "source_snippet": warr_snip
                }

            # 6. Body Type (strictly grounded in text, model, or title)
            btype = self._extract_body_type(row['make'], row['model'], row['title'], row['description'])
            body_types.append(btype)

            provenances.append(provenance if provenance else None)

        df['price_aed'] = prices
        df['monthly_payment_aed'] = monthlies
        df['mileage_km'] = mileages
        df['regional_specs'] = specs
        df['has_positive_warranty'] = has_pos_warranties
        df['warranty_status'] = warranty_statuses
        df['body_type'] = body_types
        df['provenance'] = provenances

        self._df = df

    @staticmethod
    def _extract_cash_price(text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract cash vehicle price in AED prioritizing precision over recall.
        Returns None if currency/price context is missing or ambiguous.
        Excludes phone numbers, year numbers, mileage, monthly installments, down payments, service costs.
        """
        # Pattern 1: Explicit cash statement: "AED 119,750 in cash", "Cash Price: AED 79,999"
        p1 = re.search(r'((?:AED|DHS)\s*([\d,]{4,10}(?:\.\d{2})?)\s*(?:in cash|cash price|cash\b))', text, re.IGNORECASE)
        if p1:
            try:
                val = float(p1.group(2).replace(',', ''))
                if 5000 <= val <= 5000000:  # Reasonable car price bounds
                    return val, p1.group(1)
            except ValueError:
                pass

        p2 = re.search(r'((?:cash price|vehicle price|car price)[:\s]*(?:AED|DHS)?\s*([\d,]{4,10}(?:\.\d{2})?))', text, re.IGNORECASE)
        if p2:
            try:
                val = float(p2.group(2).replace(',', ''))
                if 5000 <= val <= 5000000:
                    return val, p2.group(1)
            except ValueError:
                pass

        # Pattern 2: Standalone AED lines where line context excludes monthly/down-payment/phone/stock numbers/service price
        lines = text.split('\n')
        for line in lines:
            if re.search(r'(?:monthly|/month|/mo|per month|per mo|down-payment|down payment|financing|ref#|ref:|a month|pm\b|/pm|phone|call|tel|mobile|whatsapp|service price|service cost|service contract)', line, re.IGNORECASE):
                continue
            price_match = re.search(r'((?:AED|DHS)\s*([\d,]{5,10}(?:\.\d{2})?))', line, re.IGNORECASE)
            if price_match:
                try:
                    val = float(price_match.group(2).replace(',', ''))
                    if 5000 <= val <= 5000000:
                        return val, price_match.group(1)
                except ValueError:
                    pass
        return None, None

    @staticmethod
    def _extract_monthly_payment(text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract monthly payment rate in AED.
        Guards against showroom operating hours such as '8am to 9pm' or bare 'PM'.
        Requires explicit keywords: monthly, /month, per month, a month, or AED ... pm.
        """
        # 1. Match explicit "AED [amount] monthly" / "AED [amount] / month" / "AED [amount] per month"
        m1 = re.search(r'(((?:AED|DHS)\s*([\d,]+(?:\.\d{2})?))\s*(?:monthly|/month|per month|a month|monthy))', text, re.IGNORECASE)
        if m1:
            try:
                val = float(m1.group(3).replace(',', ''))
                if val > 100:  # Filter out interest rates like 8.00%
                    return val, m1.group(1)
            except ValueError:
                pass

        # 2. Match "[amount] AED monthly" / "From 1099 Pm" / "AED [amount] pm"
        m2 = re.search(r'((?:AED|DHS|From)?\s*([\d,]{3,6}(?:\.\d{2})?)\s*(?:monthly|/month|per month|a month|pm\b))', text, re.IGNORECASE)
        if m2:
            # Exclude time phrases like "9 pm", "8 pm", "12 pm"
            snip = m2.group(1)
            if not re.search(r'\b(?:1[0-2]|[1-9])\s*pm\b', snip, re.IGNORECASE) and not re.search(r'\b(?:to|until|till|am)\s*\d+\s*pm\b', text, re.IGNORECASE):
                try:
                    val = float(m2.group(2).replace(',', ''))
                    if val > 100:
                        return val, snip
                except ValueError:
                    pass

        return None, None

    @staticmethod
    def _extract_mileage(text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract odometer reading in KM.
        Excludes service interval limits ('service up to 30,000 km') and electric range ('electric range 650 km').
        Prefers explicit patterns: Mileage:, Odometer:, km driven, km done.
        """
        lines = text.split('\n')
        for line in lines:
            # Exclude service interval, warranty limits, or range/electric range
            if re.search(r'(?:service|warranty|maintenance|range|electric range).{0,30}\d+[\d,]*\s*km', line, re.IGNORECASE):
                if not re.search(r'(?:odometer|mileage|km done|kms done|done\s*[\d,]+)', line, re.IGNORECASE):
                    continue

            # Explicit odometer/mileage patterns
            m_exp = re.search(r'(((?:odometer|mileage|done|done\s*[:\-]?)\s*[:\-]?\s*([\d,]+))\s*(?:km|kms|kilometers|k km)\b)', line, re.IGNORECASE)
            if m_exp:
                try:
                    val = int(m_exp.group(3).replace(',', ''))
                    return val, m_exp.group(1)
                except ValueError:
                    pass

            # Standalone mileage like "56,000 KM" or "0 KM" or "120,000 km"
            m_std = re.search(r'(\b([\d,]{1,7})\s*(?:km|kms|kilometers)\b)', line, re.IGNORECASE)
            if m_std:
                prefix = line[:m_std.start()]
                if not re.search(r'(?:service|warranty|maintenance|range|electric range)\s*(?:up to|for|contract|every|or|of)?\s*$', prefix, re.IGNORECASE):
                    try:
                        val = int(m_std.group(2).replace(',', ''))
                        return val, m_std.group(1)
                    except ValueError:
                        pass
        return None, None

    @staticmethod
    def _extract_regional_specs(text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract regional specification tag.
        Only extracted when there is explicit textual evidence that the term refers to VEHICLE SPECIFICATION.
        Bare country/nationality words ('Owner relocated to USA', 'Korean owner', 'Japanese owner', 'contact us') return None.
        Returns actual matched substring as provenance snippet.
        """
        # 1. GCC Spec
        gcc_match = re.search(r'(\b(?:GCC Specs?|GCC Specifications?|GCC)\b)', text, re.IGNORECASE)
        if gcc_match:
            return "GCC", gcc_match.group(1)

        # 2. USA / American Spec
        usa_match = re.search(r'(\b(?:USA Specs?|US Specs?|American Specs?|American Specifications?|USA Specifications?)\b)', text, re.IGNORECASE)
        if usa_match:
            return "USA", usa_match.group(1)

        # 3. Korean Spec
        korean_match = re.search(r'(\b(?:Korea Specs?|Korean Specs?|Korea Specifications?|Korean Specifications?)\b)', text, re.IGNORECASE)
        if korean_match:
            return "Korean", korean_match.group(1)

        # 4. Japanese Spec
        japan_match = re.search(r'(\b(?:Japan Specs?|Japanese Specs?|Japan Specifications?|Japanese Specifications?)\b)', text, re.IGNORECASE)
        if japan_match:
            return "Japanese", japan_match.group(1)

        # 5. European Spec
        euro_match = re.search(r'(\b(?:Euro Specs?|European Specs?|German Specs?|European Specifications?)\b)', text, re.IGNORECASE)
        if euro_match:
            return "European", euro_match.group(1)

        # 6. Canadian Spec
        canadian_match = re.search(r'(\b(?:Canada Specs?|Canadian Specs?|Canadian Specifications?)\b)', text, re.IGNORECASE)
        if canadian_match:
            return "Canadian", canadian_match.group(1)

        # 7. UK Spec
        uk_match = re.search(r'(\b(?:UK Specs?|UK Specifications?)\b)', text, re.IGNORECASE)
        if uk_match:
            return "UK", uk_match.group(1)

        # 8. Russian Spec
        russian_match = re.search(r'(\b(?:Russia Specs?|Russian Specs?|Russian Specifications?)\b)', text, re.IGNORECASE)
        if russian_match:
            return "Russian", russian_match.group(1)

        # 9. Singapore Spec
        singapore_match = re.search(r'(\b(?:Singapore Specs?|Singapore Specifications?)\b)', text, re.IGNORECASE)
        if singapore_match:
            return "Singapore", singapore_match.group(1)

        # 10. Other / Custom Spec
        other_match = re.search(r'(\b(?:Other Specs?|Other Specifications?)\b)', text, re.IGNORECASE)
        if other_match:
            return "Other", other_match.group(1)

        custom_match = re.search(r'(\b(?:Custom Specs?|Custom Specifications?)\b)', text, re.IGNORECASE)
        if custom_match:
            return "Custom", custom_match.group(1)

        return None, None

    @staticmethod
    def _extract_warranty(text: str) -> Tuple[Optional[bool], Optional[str], Optional[str]]:
        """
        Extract warranty status.
        Only explicit existing warranty statements establish has_positive_warranty=True.
        Company/dealer names alone ('Gargash maintained vehicle') do NOT establish active warranty.
        Phrases like 'warranty can be arranged' or 'warranty available' set has_positive_warranty=False.
        Missing warranty facts return None for status and has_positive_warranty.
        """
        # 1. Explicit negative warranty phrases
        neg_match = re.search(r'(\b(?:no warranty|without warranty|warranty expired|out of warranty|no agency warranty|expired warranty)\b)', text, re.IGNORECASE)
        if neg_match:
            return False, "No Warranty / Expired", neg_match.group(1)

        # 2. Warranty options / arrangement (not active existing warranty)
        option_match = re.search(r'(\b(?:warranty can be arranged|warranty available|warranty option|warranty options|warranty package available)\b)', text, re.IGNORECASE)
        if option_match:
            return False, "Warranty Option Available (Not Active)", option_match.group(1)

        # 3. Explicit positive agency / manufacturer / dealer warranty
        agency_match = re.search(r'(\b(?:agency warranty|gargash warranty|gargash auto warranty|manufacturer warranty|dealer warranty)\b.{0,30})', text, re.IGNORECASE)
        if agency_match:
            return True, "Agency Warranty", agency_match.group(1).strip()

        # 4. Third-party active warranty
        third_match = re.search(r'(\b(?:3rd party|third party).{0,30}warranty\b|\bwarranty.{0,20}(?:3rd party|third party)\b)', text, re.IGNORECASE)
        if third_match:
            return True, "Third Party Warranty", third_match.group(1).strip()

        # 5. Active positive warranty mention (e.g. "under warranty", "warranty till 2027", "1 year warranty")
        active_match = re.search(r'(\b(?:under warranty|under agancy warranty|warranty till|with warranty|warranty dec|warranty jan|warranty 20\d\d|\d+ year[s]? warranty|\d+ year[s] agency warranty)\b)', text, re.IGNORECASE)
        if active_match:
            return True, "Under Warranty", active_match.group(1).strip()

        return None, None, None

    @staticmethod
    def _extract_body_type(make: str, model: str, title: str, description: str) -> Optional[str]:
        """
        Infer vehicle body type ONLY from explicit evidence found in title, model, or description.
        '4x4' by itself is drivetrain terminology and does NOT mean SUV.
        Does NOT infer body type from external model knowledge.
        Returns None if not explicitly grounded in text.
        """
        combined = f"{make} {model} {title} {description}".lower()
        if re.search(r'\b(pickup|pickup truck|truck)\b', combined):
            return "Pickup"
        elif re.search(r'\b(suv|crossover)\b', combined):
            return "SUV"
        elif re.search(r'\b(coupe|cabriolet)\b', combined):
            return "Coupe"
        elif re.search(r'\b(convertible|soft top|volante|spider)\b', combined):
            return "Convertible"
        elif re.search(r'\b(sedan|saloon)\b', combined):
            return "Sedan"
        elif re.search(r'\b(hatchback)\b', combined):
            return "Hatchback"
        elif re.search(r'\b(van)\b', combined):
            return "Van"
        elif re.search(r'\b(station wagon|wagon)\b', combined):
            return "Station Wagon"
        return None

    def search_cars(
        self,
        make: Optional[str] = None,
        model: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        min_price_aed: Optional[float] = None,
        max_price_aed: Optional[float] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_mileage_km: Optional[int] = None,
        max_mileage_km: Optional[int] = None,
        min_mileage: Optional[int] = None,
        max_mileage: Optional[int] = None,
        min_monthly_aed: Optional[float] = None,
        max_monthly_aed: Optional[float] = None,
        min_monthly_payment: Optional[float] = None,
        max_monthly_payment: Optional[float] = None,
        regional_specs: Optional[str] = None,
        warranty: Optional[bool] = None,
        keywords: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic inventory search method.
        All specified filters are evaluated as boolean AND conditions.
        Results are deterministically sorted by Listing_ID.
        Missing values are strictly excluded when evaluating numeric thresholds.
        """
        if self._df is None:
            return []

        # Handle alias parameters gracefully
        eff_min_price = min_price_aed if min_price_aed is not None else min_price
        eff_max_price = max_price_aed if max_price_aed is not None else max_price
        eff_min_mileage = min_mileage_km if min_mileage_km is not None else min_mileage
        eff_max_mileage = max_mileage_km if max_mileage_km is not None else max_mileage
        eff_min_monthly = min_monthly_aed if min_monthly_aed is not None else min_monthly_payment
        eff_max_monthly = max_monthly_aed if max_monthly_aed is not None else max_monthly_payment

        filtered = self._df.copy()

        # 1. Make filter (case-insensitive & whitespace tolerant)
        if make:
            make_clean = make.strip().lower()
            filtered = filtered[filtered['_make_clean'].str.contains(re.escape(make_clean), na=False)]

        # 2. Model filter (case-insensitive & whitespace tolerant)
        if model:
            model_clean = model.strip().lower()
            filtered = filtered[filtered['_model_clean'].str.contains(re.escape(model_clean), na=False)]

        # 3. Year filters
        if min_year is not None:
            filtered = filtered[filtered['year'] >= min_year]
        if max_year is not None:
            filtered = filtered[filtered['year'] <= max_year]

        # 4. Cash Price filters (excludes listings without extracted price)
        if eff_min_price is not None:
            filtered = filtered[
                filtered['price_aed'].notnull() &
                (filtered['price_aed'] >= eff_min_price)
            ]
        if eff_max_price is not None:
            filtered = filtered[
                filtered['price_aed'].notnull() &
                (filtered['price_aed'] <= eff_max_price)
            ]

        # 5. Monthly Payment filters
        if eff_min_monthly is not None:
            filtered = filtered[
                filtered['monthly_payment_aed'].notnull() &
                (filtered['monthly_payment_aed'] >= eff_min_monthly)
            ]
        if eff_max_monthly is not None:
            filtered = filtered[
                filtered['monthly_payment_aed'].notnull() &
                (filtered['monthly_payment_aed'] <= eff_max_monthly)
            ]

        # 6. Mileage filters (excludes listings without extracted mileage)
        if eff_min_mileage is not None:
            filtered = filtered[
                filtered['mileage_km'].notnull() &
                (filtered['mileage_km'] >= eff_min_mileage)
            ]
        if eff_max_mileage is not None:
            filtered = filtered[
                filtered['mileage_km'].notnull() &
                (filtered['mileage_km'] <= eff_max_mileage)
            ]

        # 7. Regional Specs filter
        if regional_specs:
            spec_clean = regional_specs.strip().upper()
            filtered = filtered[
                filtered['regional_specs'].notnull() &
                (filtered['regional_specs'].str.upper() == spec_clean)
            ]

        # 8. Warranty filter (True requires active warranty, False requires explicit non-active/negative)
        if warranty is True:
            filtered = filtered[filtered['has_positive_warranty'] == True]
        elif warranty is False:
            filtered = filtered[filtered['has_positive_warranty'] == False]

        # 9. Free-text keyword search across useful original text fields
        if keywords:
            kw_tokens = [k.strip().lower() for k in keywords.strip().split() if k.strip()]
            for kw in kw_tokens:
                cond = (
                    filtered['_title_clean'].str.contains(re.escape(kw), na=False) |
                    filtered['_description_clean'].str.contains(re.escape(kw), na=False) |
                    filtered['_make_clean'].str.contains(re.escape(kw), na=False) |
                    filtered['_model_clean'].str.contains(re.escape(kw), na=False) |
                    filtered['_trim_clean'].str.contains(re.escape(kw), na=False)
                )
                filtered = filtered[cond]

        # Deterministic sorting by Listing_ID ascending
        filtered = filtered.sort_values(by='Listing_ID', ascending=True)

        # Apply limit if specified
        if limit is not None and limit > 0:
            filtered = filtered.head(limit)

        # Convert to dictionary records preserving original display values
        records = []
        for _, row in filtered.iterrows():
            rec = {
                "listing_id": int(row['Listing_ID']),
                "year": int(row['year']),
                "make": str(row['make']),
                "model": str(row['model']),
                "trim": str(row['trim']),
                "title": str(row['title']),
                "description": str(row['description']),
                "photo_url": str(row['photo_url']),
                "price_aed": float(row['price_aed']) if pd.notnull(row['price_aed']) else None,
                "monthly_payment_aed": float(row['monthly_payment_aed']) if pd.notnull(row['monthly_payment_aed']) else None,
                "mileage_km": int(row['mileage_km']) if pd.notnull(row['mileage_km']) else None,
                "regional_specs": str(row['regional_specs']) if pd.notnull(row['regional_specs']) else None,
                "has_positive_warranty": bool(row['has_positive_warranty']) if pd.notnull(row['has_positive_warranty']) else None,
                "warranty_status": str(row['warranty_status']) if pd.notnull(row['warranty_status']) else None,
                "body_type": str(row['body_type']) if pd.notnull(row['body_type']) else None,
                "provenance": row['provenance']
            }
            records.append(rec)

        return records

    def search(self, filters: CarFilter) -> List[Dict[str, Any]]:
        """Wrapper method mapping CarFilter schema parameters to search_cars()."""
        return self.search_cars(
            make=filters.make,
            model=filters.model,
            min_year=filters.min_year,
            max_year=filters.max_year,
            min_price_aed=filters.min_price_aed,
            max_price_aed=filters.max_price_aed,
            min_price=filters.min_price,
            max_price=filters.max_price,
            min_mileage_km=filters.min_mileage_km,
            max_mileage_km=filters.max_mileage_km,
            min_mileage=filters.min_mileage,
            max_mileage=filters.max_mileage,
            min_monthly_aed=filters.min_monthly_aed,
            max_monthly_aed=filters.max_monthly_aed,
            min_monthly_payment=filters.min_monthly_payment,
            max_monthly_payment=filters.max_monthly_payment,
            regional_specs=filters.regional_specs,
            warranty=filters.warranty,
            keywords=filters.keywords,
            limit=filters.limit
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns statistical overview of the loaded car inventory dataset."""
        if self._df is None:
            return {}

        return {
            "total_listings": len(self._df),
            "unique_makes": int(self._df['make'].nunique()),
            "make_counts": {str(k): int(v) for k, v in self._df['make'].value_counts().items()},
            "min_year": int(self._df['year'].min()),
            "max_year": int(self._df['year'].max()),
            "spec_counts": {str(k) if pd.notnull(k) else "Unspecified": int(v) for k, v in self._df['regional_specs'].value_counts(dropna=False).items()},
            "warranty_counts": {str(k) if pd.notnull(k) else "Unspecified": int(v) for k, v in self._df['warranty_status'].value_counts(dropna=False).items()},
            "body_type_counts": {str(k) if pd.notnull(k) else "Unspecified": int(v) for k, v in self._df['body_type'].value_counts(dropna=False).items()},
        }

    def get_by_listing_id(self, listing_id: int) -> Optional[CarListing]:
        """Direct, read-only lookup of a single CarListing by Listing_ID from the loaded dataset."""
        if self._df is None or self._df.empty:
            return None
        try:
            target_id = int(listing_id)
        except (ValueError, TypeError):
            return None

        matches = self._df[self._df['Listing_ID'] == target_id]
        if matches.empty:
            return None

        row = matches.iloc[0]
        rec = {
            "listing_id": int(row['Listing_ID']),
            "year": int(row['year']),
            "make": str(row['make']),
            "model": str(row['model']),
            "trim": str(row['trim']),
            "title": str(row['title']),
            "description": str(row['description']),
            "photo_url": str(row['photo_url']),
            "price_aed": float(row['price_aed']) if pd.notnull(row['price_aed']) else None,
            "monthly_payment_aed": float(row['monthly_payment_aed']) if pd.notnull(row['monthly_payment_aed']) else None,
            "mileage_km": int(row['mileage_km']) if pd.notnull(row['mileage_km']) else None,
            "regional_specs": str(row['regional_specs']) if pd.notnull(row['regional_specs']) else None,
            "has_positive_warranty": bool(row['has_positive_warranty']) if pd.notnull(row['has_positive_warranty']) else None,
            "warranty_status": str(row['warranty_status']) if pd.notnull(row['warranty_status']) else None,
            "body_type": str(row['body_type']) if pd.notnull(row['body_type']) else None,
            "provenance": row['provenance']
        }
        return CarListing.model_validate(rec)

