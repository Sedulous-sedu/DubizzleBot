# Demo: Returning-User Persistent Memory Across Sessions

This transcript demonstrates:
1. Explicit favourite saved in Session A
2. New Conversation button clicked (new `session_id`, chat cleared)
3. Returning-user recognition in Session B — `LongTermMemoryResolver` retrieves SQLite record
4. Listing #17 re-hydrated from live inventory

---

## Session Details

- **User:** `demo_user`
- **Contact info:** synthetic only
- **Backend:** `uv run uvicorn backend.main:app --reload`

---

## Session A — Save a Favourite

**User:** Show me Bentleys

**DubizzleBot:**
> I found **7 Bentley** vehicles in our UAE inventory.
>
> **[1] Listing #9** — 2020 Bentley Continental GT, AED 459,999, 5,200 km
> **[2] Listing #17** — 2020 Bentley Continental GT, AED 489,000, **318 km**
> *(… remaining 5 results …)*

---

**User:** I like the second one

**DubizzleBot:**
> Got it! I've saved **Listing #17** (2020 Bentley Continental GT, 318 km, AED 489,000) to your favourites.
>
> You can recall it in any future conversation by asking "What cars did I like?"

---

## ↕️ User clicks ➕ New Conversation

A new `session_id` UUID is generated and shown in the sidebar. The chat window is cleared. The `user_id` remains `demo_user`. The SQLite record for Listing #17 is untouched.

---

## Session B — Recall Across Sessions

**User:** What cars did I like?

**DubizzleBot:**
> Welcome back! Based on your saved favourites, you previously showed interest in:
>
> ---
> **Listing #17** — 2020 Bentley Continental GT
> 💰 AED 489,000 · 📍 GCC Spec · 🔧 Under Warranty · 🛣 318 km
>
> Would you like to arrange a test drive for this vehicle, or explore similar options?

---

**User:** Yes, I want to test drive it — next Saturday at 2 PM

**DubizzleBot:**
> I'd be happy to arrange a viewing for you!
>
> **Proposed appointment:**
> - 🚗 **Vehicle:** Listing #17 — 2020 Bentley Continental GT
> - 📅 **Date:** Saturday, 30 August 2026
> - 🕐 **Time:** 14:00 Asia/Dubai
>
> All our showrooms are open Monday–Saturday, 08:00–20:00. Shall I confirm this booking?

---

**User:** Yes, confirm

**DubizzleBot:**
> ✅ **Booking Confirmed!**
>
> Your test-drive viewing has been scheduled:
> - 🎫 **Reference:** #BK-A3F2C9
> - 🚗 Listing #17 — 2020 Bentley Continental GT
> - 📅 Saturday, 30 August 2026 at 14:00
> - 📋 A copy of your booking details has been saved.

---

## Key Observations

- **Persistent memory boundary**: The `user_id` (`demo_user`) bridges sessions. The `session_id` does not.
- **`LongTermMemoryResolver`** reads the `user_preferences` SQLite table keyed on `user_id`.
- **Listing re-hydration**: Listing #17 is fetched from live inventory (not stored verbatim) to ensure the response reflects current dataset state.
- **Invariant**: The "New Conversation" action does **not** delete persistent memory. Only an explicit "remove from favourites" would do so.
- **Booking reference**: `#BK-A3F2C9` is a stable identifier persisted to SQLite — retrying the same booking_id returns the existing record without overwriting it.
