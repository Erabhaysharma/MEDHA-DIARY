"""
payments.py — Razorpay integration
"""

import os
import hmac
import hashlib
import json
import razorpay
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from supabase import create_client
from app.auth import verify_token

router = APIRouter()


def get_sb():
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY")
    )


def get_razorpay():
    return razorpay.Client(
        auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET"))
    )


@router.post("/payments/create-link")
async def create_payment_link(user_id: str = Depends(verify_token)):
    client = get_razorpay()
    sb     = get_sb()

    profile = sb.table("profiles")\
        .select("display_name")\
        .eq("id", user_id).single().execute()
    name = (profile.data or {}).get("display_name", "") or ""

    try:
        link = client.payment_link.create({
            "amount":      100,
            "currency":    "INR",
            "description": "Astro Medha Premium — Monthly",
            "notes": {
                "user_id": user_id,
            },
            "customer": {
                "name": name,
            },
            "notify": {
                "sms":   False,
                "email": False,
            },
            "reminder_enable": False,
            "options": {
                "checkout": {
                    "method": {
                        "upi":        1,
                        "card":       1,
                        "netbanking": 0,
                        "wallet":     0,
                    },
                    "prefill": { "name": name },
                    "theme":   { "color": "#C8A96E" },
                }
            },
        })
        return {"payment_url": link["short_url"]}

    except Exception as e:
        print(f"Razorpay create-link error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/payments/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay calls this automatically when payment completes.
    Verifies signature then activates premium.
    """
    body      = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    secret    = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    print(f"Webhook received. Event body length: {len(body)}")
    print(f"Signature header: {signature[:20]}...")

    # ── Verify signature ──
    if secret:
        expected = hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            print("Webhook signature mismatch — rejecting")
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        print("WARNING: RAZORPAY_WEBHOOK_SECRET not set — skipping signature check")

    # ── Parse body ──
    try:
        payload = json.loads(body)
    except Exception as e:
        print(f"Webhook JSON parse error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = payload.get("event", "")
    print(f"Webhook event: {event}")

    # ── Handle payment_link.paid ──
    if event == "payment_link.paid":
        try:
            payment_link_entity = payload["payload"]["payment_link"]["entity"]
            notes               = payment_link_entity.get("notes", {})
            user_id             = notes.get("user_id")
            amount_paid         = payment_link_entity.get("amount_paid", 0)

            print(f"Payment received. user_id={user_id}, amount={amount_paid}")

            if not user_id:
                print("ERROR: No user_id in payment notes")
                return {"status": "no user_id"}

            sb = get_sb()
            result = sb.table("profiles").update({
                "is_premium":    True,
                "premium_since": datetime.now().isoformat(),
            }).eq("id", user_id).execute()

            print(f"Premium activated for user: {user_id}")
            print(f"Supabase update result: {result.data}")

        except Exception as e:
            print(f"Webhook processing error: {e}")
            # Still return 200 so Razorpay doesn't retry endlessly
            return {"status": "error", "detail": str(e)}

    return {"status": "ok", "event": event}


@router.get("/payments/verify-premium")
async def verify_premium(user_id: str = Depends(verify_token)):
    """
    Frontend polls this after returning from payment.
    Returns current premium status fresh from DB.
    """
    sb  = get_sb()
    res = sb.table("profiles")\
        .select("is_premium, astro_onboarded, premium_since")\
        .eq("id", user_id).single().execute()
    data = res.data or {}
    return {
        "is_premium":      data.get("is_premium", False),
        "astro_onboarded": data.get("astro_onboarded", False),
        "premium_since":   data.get("premium_since"),
    }