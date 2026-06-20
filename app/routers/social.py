from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.auth import verify_token
from supabase import create_client
import os

router = APIRouter()

def get_sb():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))


class PostCreate(BaseModel):
    entry_id:     str
    caption:      str
    is_anonymous: bool = False


class CommentCreate(BaseModel):
    content: str
@router.post("/social/post")
async def create_post(body: PostCreate, user_id: str = Depends(verify_token)):
    sb = get_sb()
    entry = sb.table("diary_entries").select("id").eq("id", body.entry_id).eq("user_id", user_id).single().execute()
    if not entry.data:
        raise HTTPException(status_code=404, detail="Entry not found")
    post = sb.table("social_posts").insert({
        "user_id":      user_id,
        "entry_id":     body.entry_id,
        "caption":      body.caption,
        "is_anonymous": body.is_anonymous,
    }).execute()
    sb.rpc("increment_shares", {"uid": user_id}).execute()
    return post.data[0]

@router.get("/social/feed")
async def get_feed(page: int = 0, user_id: str = Depends(verify_token)):
    sb = get_sb()
    limit  = 20
    offset = page * limit

    posts = sb.table("social_posts") \
        .select("*") \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1) \
        .execute()

    if not posts.data:
        return []

    result = []
    for p in posts.data:
        post_id = p["id"]

        # Fetch entry separately
        entry_row = sb.table("diary_entries") \
            .select("content, entry_date, mood_label") \
            .eq("id", p["entry_id"]) \
            .single() \
            .execute()
        entry = entry_row.data or {}

        # Fetch profile separately
        profile_row = sb.table("profiles") \
            .select("display_name") \
            .eq("id", p["user_id"]) \
            .single() \
            .execute()
        profile = profile_row.data or {}

        likes    = sb.table("post_likes").select("user_id", count="exact").eq("post_id", post_id).execute()
        comments = sb.table("post_comments").select("id", count="exact").eq("post_id", post_id).execute()
        my_like  = sb.table("post_likes").select("user_id").eq("post_id", post_id).eq("user_id", user_id).execute()

        result.append({
            "id":             post_id,
            "user_id":        p["user_id"],
            "entry_id":       p["entry_id"],
            "caption":        p["caption"],
            "is_anonymous":   p["is_anonymous"],
            "views":          p["views"],
            "created_at":     p["created_at"],
            "entry_content":  entry.get("content", ""),
            "entry_date":     entry.get("entry_date", ""),
            "mood_label":     entry.get("mood_label"),
            "display_name":   None if p["is_anonymous"] else profile.get("display_name"),
            "likes_count":    likes.count or 0,
            "comments_count": comments.count or 0,
            "liked_by_me":    len(my_like.data) > 0,
        })

    return result

@router.post("/social/post/{post_id}/like")
async def toggle_like(post_id: str, user_id: str = Depends(verify_token)):
    sb = get_sb()
    existing = sb.table("post_likes").select("user_id").eq("post_id", post_id).eq("user_id", user_id).execute()
    if existing.data:
        sb.table("post_likes").delete().eq("post_id", post_id).eq("user_id", user_id).execute()
        liked = False
    else:
        sb.table("post_likes").insert({"post_id": post_id, "user_id": user_id}).execute()
        liked = True
    count = sb.table("post_likes").select("user_id", count="exact").eq("post_id", post_id).execute()
    return {"liked": liked, "likes_count": count.count or 0}


@router.post("/social/post/{post_id}/view")
async def increment_view(post_id: str, user_id: str = Depends(verify_token)):
    sb = get_sb()
    sb.rpc("increment_post_views", {"pid": post_id}).execute()
    return {"ok": True}


@router.get("/social/post/{post_id}/comments")
async def get_comments(post_id: str, user_id: str = Depends(verify_token)):
    sb = get_sb()
    rows = sb.table("post_comments") \
        .select("*, profiles(display_name)") \
        .eq("post_id", post_id) \
        .order("created_at") \
        .execute()
    return [
        {**r, "display_name": (r.get("profiles") or {}).get("display_name")}
        for r in (rows.data or [])
    ]


@router.post("/social/post/{post_id}/comment")
async def add_comment(post_id: str, body: CommentCreate, user_id: str = Depends(verify_token)):
    sb = get_sb()
    row = sb.table("post_comments").insert({
        "post_id": post_id,
        "user_id": user_id,
        "content": body.content,
    }).execute()
    return row.data[0]