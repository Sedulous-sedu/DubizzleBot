# Demo: Multi-Turn Inventory Search with Contextual Reference Resolution

This transcript demonstrates:
1. Natural-language inventory search with structured filter extraction
2. Ordinal reference resolution across turns ("the second one")
3. Session-level short-term memory (`ContextResolver`)

---

## Session Details

- **User:** `demo_user`
- **Session:** fresh session (new session_id assigned)
- **Backend:** `uv run uvicorn backend.main:app --reload`
- **Contact info:** synthetic only

---

## Transcript

**User:** Show me Bentleys

**DubizzleBot:**
> I found **7 Bentley** vehicles in our UAE inventory.
>
> ---
> **[1] Listing #9** — 2020 Bentley Continental GT
> 💰 AED 459,999 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 5,200 km
>
> **[2] Listing #17** — 2020 Bentley Continental GT
> 💰 AED 489,000 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 318 km
>
> **[3] Listing #28** — 2022 Bentley Bentayga
> 💰 AED 649,000 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 12,100 km
>
> **[4] Listing #41** — 2021 Bentley Flying Spur
> 💰 AED 539,900 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 8,750 km
>
> **[5] Listing #52** — 2019 Bentley Mulsanne
> 💰 AED 599,000 · 📍 GCC Spec · ⚠️ Warranty Expired · 🛣 22,000 km
>
> **[6] Listing #64** — 2020 Bentley Continental GT Convertible
> 💰 AED 620,000 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 6,800 km
>
> **[7] Listing #78** — 2021 Bentley Bentayga V8
> 💰 AED 610,000 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 9,400 km

---

**User:** What's the mileage on the second one?

**DubizzleBot:**
> The second vehicle — **Listing #17, 2020 Bentley Continental GT** — has a recorded mileage of **318 km**.
>
> This is essentially a nearly-new vehicle. Would you like more details, or are you interested in arranging a test drive?

---

**User:** What's the price of the third one?

**DubizzleBot:**
> **Listing #28, 2022 Bentley Bentayga** is listed at **AED 649,000**.
>
> Would you like to compare it with the second vehicle (AED 489,000), or shall I show you similar models within a different budget?

---

## Key Observations

- **Structured extraction**: "Show me Bentleys" → `{"make": "Bentley"}` passed to `InventoryService`. No LLM involvement in result selection.
- **Ordinal resolution**: "the second one" → `ContextResolver` resolves to `index=1` in the stored result set → Listing #17.
- **Third follow-up**: "the third one" → resolved to `index=2` → Listing #28. The entire 7-result context window remains accessible.
- **Facts are real**: 318 km mileage is extracted directly from the dataset field. The LLM does not fabricate it.
