"""
payments.py — Razorpay integration
"""

import os
import hmac
import hashlib
import razorpay
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from supabase import create_client
from app.auth import verify_token

router = APIRouter()

def get_sb():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

def get_razorpay():
    return razorpay.Client(
        auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
    )


@router.post("/payments/create-link")
async def create_payment_link(user_id: str = Depends(verify_token)):
    """
    Creates a fresh Razorpay payment link with user_id embedded in notes.
    Called when user taps Subscribe in the app.
    """
    client  = get_razorpay()
    sb      = get_sb()

    # Get user profile for prefilling
    profile = sb.table("profiles")\
        .select("display_name")\
        .eq("id", user_id).single().execute()
    name = profile.data.get("display_name", "") if profile.data else ""

    try:
        link = client.payment_link.create({
            "amount":      14900,   # ₹149 in paise
            "currency":    "INR",
            "description": "Astro Medha Premium — Monthly",
            "notes": {
                "user_id": user_id,   # ← this is how we know who paid
            },
            "customer": {
                "name": name,
            },
            "notify": {
                "sms":   False,
                "email": False,
            },
            "reminder_enable": False,
            "callback_url":    os.getenv("APP_SCHEME", "medha://") + "payment-success",
            "callback_method": "get",
        })
        return {"payment_url": link["short_url"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/payments/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay calls this when payment is completed.
    Automatically activates premium for the user.
    """
    body      = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    secret    = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # Verify the webhook is genuinely from Razorpay
    expected = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    payload = request.json() if callable(request.json) else {}
    try:
        import json
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = payload.get("event", "")

    if event == "payment_link.paid":
        try:
            notes   = payload["payload"]["payment_link"]["entity"].get("notes", {})
            user_id = notes.get("user_id")

            if not user_id:
                return {"status": "no user_id in notes"}

            sb = get_sb()
            sb.table("profiles").update({
                "is_premium":    True,
                "premium_since": datetime.now().isoformat(),
            }).eq("id", user_id).execute()

            print(f"Premium activated for user: {user_id}")

        except Exception as e:
            print(f"Webhook processing error: {e}")

    return {"status": "ok"}