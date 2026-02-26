from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import json
import re
import base64
import base64
from passlib.context import CryptContext
from jose import JWTError, jwt
from enum import Enum

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'soundchat-super-secret-key-change-in-production-2024')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create the main app
app = FastAPI(title="SoundChat API", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# Enums
# ================================

class MessageStatus(str, Enum):
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE_NOTE = "voice_note"
    DOCUMENT = "document"
    SOUND = "sound"
    STICKER = "sticker"
    GIF = "gif"
    LOCATION = "location"
    CONTACT = "contact"
    SYSTEM = "system"

class ChatType(str, Enum):
    DIRECT = "direct"
    GROUP = "group"

class CallType(str, Enum):
    VOICE = "voice"
    VIDEO = "video"

class CallStatus(str, Enum):
    INITIATED = "initiated"
    RINGING = "ringing"
    ONGOING = "ongoing"
    ENDED = "ended"
    MISSED = "missed"
    DECLINED = "declined"

class UserStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    TYPING = "typing"

# ================================
# Password & JWT Utilities
# ================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ================================
# Request/Response Models
# ================================

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return v.strip()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    user: Dict[str, Any]
    token: str
    token_type: str = "bearer"

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    status_message: Optional[str] = None
    picture: Optional[str] = None
    phone: Optional[str] = None

class CreateGroupRequest(BaseModel):
    name: str
    description: Optional[str] = None
    participants: List[str]
    picture: Optional[str] = None

class UpdateGroupRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    picture: Optional[str] = None

class SendMessageRequest(BaseModel):
    content: str
    message_type: MessageType = MessageType.TEXT
    sound_id: Optional[str] = None
    reply_to: Optional[str] = None
    media_url: Optional[str] = None
    media_data: Optional[str] = None  # Base64 encoded
    duration: Optional[float] = None  # For voice notes
    forwarded_from: Optional[str] = None

class MessageReactionRequest(BaseModel):
    emoji: str
    sound_id: Optional[str] = None

class UpdateSettingsRequest(BaseModel):
    master_volume: Optional[float] = None
    night_mode_enabled: Optional[bool] = None
    night_mode_start: Optional[str] = None
    night_mode_end: Optional[str] = None
    last_seen_visible: Optional[bool] = None
    read_receipts_enabled: Optional[bool] = None
    typing_indicator_enabled: Optional[bool] = None

class ChatSettingsRequest(BaseModel):
    is_muted: Optional[bool] = None
    mute_until: Optional[datetime] = None
    custom_notification_sound: Optional[str] = None
    sounds_enabled: Optional[bool] = None
    is_pinned: Optional[bool] = None
    is_archived: Optional[bool] = None

class BlockUserRequest(BaseModel):
    user_id: str
    reason: Optional[str] = None

class ReportUserRequest(BaseModel):
    user_id: str
    reason: str
    message_ids: Optional[List[str]] = None

class InitiateCallRequest(BaseModel):
    participant_ids: List[str]
    call_type: CallType = CallType.VOICE

class MoodAnalysisRequest(BaseModel):
    text: str

class MoodAnalysisResponse(BaseModel):
    mood: str
    suggested_emojis: List[str]
    suggested_sounds: List[Dict[str, str]]

class SearchRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    limit: int = 50

# ================================
# WebSocket Connection Manager
# ================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_status: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_status[user_id] = {
            "status": UserStatus.ONLINE,
            "last_seen": datetime.now(timezone.utc)
        }
        # Broadcast online status
        await self.broadcast_status(user_id, UserStatus.ONLINE)
        logger.info(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_status:
            self.user_status[user_id] = {
                "status": UserStatus.OFFLINE,
                "last_seen": datetime.now(timezone.utc)
            }
        logger.info(f"User {user_id} disconnected from WebSocket")
    
    def get_user_status(self, user_id: str) -> Dict[str, Any]:
        return self.user_status.get(user_id, {
            "status": UserStatus.OFFLINE,
            "last_seen": None
        })
    
    def is_online(self, user_id: str) -> bool:
        return user_id in self.active_connections
    
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                return False
        return False
    
    async def broadcast_to_chat(self, message: dict, chat_id: str, exclude_user: str = None):
        chat = await db.chats.find_one({"chat_id": chat_id}, {"_id": 0})
        if chat:
            for participant_id in chat["participants"]:
                if participant_id != exclude_user:
                    await self.send_personal_message(message, participant_id)
    
    async def broadcast_status(self, user_id: str, status: UserStatus):
        # Get all chats this user is part of
        chats = await db.chats.find({"participants": user_id}, {"_id": 0}).to_list(100)
        notified_users = set()
        for chat in chats:
            for participant_id in chat["participants"]:
                if participant_id != user_id and participant_id not in notified_users:
                    await self.send_personal_message({
                        "type": "user_status",
                        "user_id": user_id,
                        "status": status.value,
                        "last_seen": datetime.now(timezone.utc).isoformat()
                    }, participant_id)
                    notified_users.add(participant_id)

manager = ConnectionManager()

# ================================
# Auth Helper
# ================================

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Extract and validate user from JWT token"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.split(" ")[1]
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0, "hashed_password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

def serialize_datetime(obj):
    """Helper to serialize datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def serialize_doc(doc: dict) -> dict:
    """Serialize a MongoDB document for JSON response"""
    if not doc:
        return doc
    result = {}
    for key, value in doc.items():
        if key == "_id":
            continue
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, list):
            result[key] = [serialize_doc(item) if isinstance(item, dict) else serialize_datetime(item) for item in value]
        elif isinstance(value, dict):
            result[key] = serialize_doc(value)
        else:
            result[key] = value
    return result

# ================================
# Auth Endpoints
# ================================

@api_router.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user with email and password"""
    existing_user = await db.users.find_one({"email": req.email.lower()})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    hashed_password = get_password_hash(req.password)
    
    user_doc = {
        "user_id": user_id,
        "email": req.email.lower(),
        "name": req.name,
        "phone": req.phone,
        "hashed_password": hashed_password,
        "picture": None,
        "status_message": "Hey there! I am using SoundChat",
        "created_at": datetime.now(timezone.utc),
        "last_seen": datetime.now(timezone.utc),
        "is_online": False,
        "blocked_users": [],
        "settings": {
            "master_volume": 1.0,
            "night_mode_enabled": False,
            "night_mode_start": "22:00",
            "night_mode_end": "07:00",
            "last_seen_visible": True,
            "read_receipts_enabled": True,
            "typing_indicator_enabled": True,
            "notification_sound": "default",
            "vibration_enabled": True
        }
    }
    
    await db.users.insert_one(user_doc.copy())
    
    access_token = create_access_token(
        data={"sub": user_id, "email": req.email.lower()},
        expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    
    session_doc = {
        "session_id": str(uuid.uuid4()),
        "user_id": user_id,
        "token": access_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    user_response = serialize_doc(user_doc)
    del user_response["hashed_password"]
    
    return AuthResponse(user=user_response, token=access_token)

@api_router.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login with email and password"""
    user = await db.users.find_one({"email": req.email.lower()})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(req.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token = create_access_token(
        data={"sub": user["user_id"], "email": user["email"]},
        expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    
    # Update last seen
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": {"last_seen": datetime.now(timezone.utc)}}
    )
    
    session_doc = {
        "session_id": str(uuid.uuid4()),
        "user_id": user["user_id"],
        "token": access_token,
        "expires_at": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "created_at": datetime.now(timezone.utc)
    }
    await db.user_sessions.insert_one(session_doc)
    
    user_response = serialize_doc(user)
    if "hashed_password" in user_response:
        del user_response["hashed_password"]
    
    return AuthResponse(user=user_response, token=access_token)

@api_router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Get current authenticated user"""
    return serialize_doc(user)

@api_router.post("/auth/logout")
async def logout(request: Request):
    """Logout and invalidate token"""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        await db.user_sessions.delete_one({"token": token})
    return {"message": "Logged out successfully"}

# ================================
# User & Profile Endpoints
# ================================

@api_router.get("/users")
async def get_users(user: dict = Depends(get_current_user)):
    """Get all users except current user and blocked users"""
    blocked = user.get("blocked_users", [])
    users = await db.users.find(
        {"user_id": {"$ne": user["user_id"], "$nin": blocked}},
        {"_id": 0, "hashed_password": 0}
    ).to_list(100)
    
    result = []
    for u in users:
        user_data = serialize_doc(u)
        user_data["is_online"] = manager.is_online(u["user_id"])
        result.append(user_data)
    
    return result

@api_router.get("/users/blocked")
async def get_blocked_users(user: dict = Depends(get_current_user)):
    """Get list of blocked users"""
    blocked_ids = user.get("blocked_users", [])
    if not blocked_ids:
        return []
    
    blocked_users = await db.users.find(
        {"user_id": {"$in": blocked_ids}},
        {"_id": 0, "user_id": 1, "name": 1, "picture": 1}
    ).to_list(100)
    
    return [serialize_doc(u) for u in blocked_users]

@api_router.get("/users/{user_id}")
async def get_user(user_id: str, user: dict = Depends(get_current_user)):
    """Get a specific user"""
    target_user = await db.users.find_one(
        {"user_id": user_id}, 
        {"_id": 0, "hashed_password": 0}
    )
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    result = serialize_doc(target_user)
    result["is_online"] = manager.is_online(user_id)
    
    # Check privacy settings
    if not target_user.get("settings", {}).get("last_seen_visible", True):
        result["last_seen"] = None
    
    return result

@api_router.get("/users/{user_id}/status")
async def get_user_status(user_id: str, user: dict = Depends(get_current_user)):
    """Get user's online status"""
    target_user = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    status = manager.get_user_status(user_id)
    
    # Check privacy settings
    if not target_user.get("settings", {}).get("last_seen_visible", True):
        return {
            "user_id": user_id,
            "is_online": status["status"] == UserStatus.ONLINE,
            "last_seen": None
        }
    
    return {
        "user_id": user_id,
        "is_online": status["status"] == UserStatus.ONLINE,
        "status": status["status"].value if isinstance(status["status"], UserStatus) else status["status"],
        "last_seen": serialize_datetime(status.get("last_seen"))
    }

@api_router.patch("/profile")
async def update_profile(req: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    """Update user profile"""
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": update_data}
    )
    
    updated_user = await db.users.find_one(
        {"user_id": user["user_id"]},
        {"_id": 0, "hashed_password": 0}
    )
    return serialize_doc(updated_user)

@api_router.post("/users/block")
async def block_user(req: BlockUserRequest, user: dict = Depends(get_current_user)):
    """Block a user"""
    if req.user_id == user["user_id"]:
        raise HTTPException(status_code=400, detail="Cannot block yourself")
    
    target_user = await db.users.find_one({"user_id": req.user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$addToSet": {"blocked_users": req.user_id}}
    )
    
    # Log the block
    await db.blocked_users.insert_one({
        "blocker_id": user["user_id"],
        "blocked_id": req.user_id,
        "reason": req.reason,
        "created_at": datetime.now(timezone.utc)
    })
    
    return {"message": f"User {req.user_id} blocked successfully"}

@api_router.post("/users/unblock")
async def unblock_user(req: BlockUserRequest, user: dict = Depends(get_current_user)):
    """Unblock a user"""
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$pull": {"blocked_users": req.user_id}}
    )
    return {"message": f"User {req.user_id} unblocked successfully"}

@api_router.post("/users/report")
async def report_user(req: ReportUserRequest, user: dict = Depends(get_current_user)):
    """Report a user"""
    target_user = await db.users.find_one({"user_id": req.user_id})
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    report_doc = {
        "report_id": f"report_{uuid.uuid4().hex[:12]}",
        "reporter_id": user["user_id"],
        "reported_id": req.user_id,
        "reason": req.reason,
        "message_ids": req.message_ids or [],
        "status": "pending",
        "created_at": datetime.now(timezone.utc)
    }
    await db.reports.insert_one(report_doc)
    
    return {"message": "Report submitted successfully", "report_id": report_doc["report_id"]}

# ================================
# Chat Endpoints
# ================================

@api_router.post("/chats")
async def create_chat(
    participant_id: str = None,
    chat_type: ChatType = ChatType.DIRECT,
    name: str = None,
    user: dict = Depends(get_current_user)
):
    """Create a new direct or group chat"""
    if chat_type == ChatType.DIRECT:
        if not participant_id:
            raise HTTPException(status_code=400, detail="participant_id required for direct chat")
        
        # Check if direct chat already exists
        existing_chat = await db.chats.find_one({
            "chat_type": ChatType.DIRECT.value,
            "participants": {"$all": [user["user_id"], participant_id]}
        }, {"_id": 0})
        
        if existing_chat:
            return serialize_doc(existing_chat)
        
        participants = [user["user_id"], participant_id]
    else:
        participants = [user["user_id"]]
    
    chat_doc = {
        "chat_id": f"chat_{uuid.uuid4().hex[:12]}",
        "chat_type": chat_type.value,
        "participants": participants,
        "name": name,
        "description": None,
        "picture": None,
        "created_by": user["user_id"],
        "created_at": datetime.now(timezone.utc),
        "last_message_at": None,
        "last_message": None,
        "settings": {},
        "admins": [user["user_id"]] if chat_type == ChatType.GROUP else [],
        "is_muted": False,
        "is_pinned": False,
        "is_archived": False,
        "sounds_enabled": True
    }
    
    await db.chats.insert_one(chat_doc.copy())
    
    # Create chat settings for participants
    for pid in participants:
        await db.chat_settings.update_one(
            {"chat_id": chat_doc["chat_id"], "user_id": pid},
            {"$set": {
                "chat_id": chat_doc["chat_id"],
                "user_id": pid,
                "is_muted": False,
                "mute_until": None,
                "is_pinned": False,
                "is_archived": False,
                "sounds_enabled": True,
                "custom_notification_sound": None,
                "unread_count": 0
            }},
            upsert=True
        )
    
    return serialize_doc(chat_doc)

@api_router.get("/chats")
async def get_chats(
    archived: bool = False,
    user: dict = Depends(get_current_user)
):
    """Get all chats for current user"""
    # Get chat settings for the user
    chat_settings = await db.chat_settings.find(
        {"user_id": user["user_id"]},
        {"_id": 0}
    ).to_list(1000)
    
    settings_map = {cs["chat_id"]: cs for cs in chat_settings}
    
    # Filter based on archived status
    chat_ids_to_show = [
        cs["chat_id"] for cs in chat_settings 
        if cs.get("is_archived", False) == archived
    ]
    
    if not archived:
        # Also include chats without settings (new chats)
        all_chats = await db.chats.find(
            {"participants": user["user_id"]},
            {"_id": 0, "chat_id": 1}
        ).to_list(1000)
        all_chat_ids = {c["chat_id"] for c in all_chats}
        chats_with_settings = set(settings_map.keys())
        new_chat_ids = all_chat_ids - chats_with_settings
        chat_ids_to_show = list(set(chat_ids_to_show) | new_chat_ids)
    
    chats = await db.chats.find(
        {"chat_id": {"$in": chat_ids_to_show}, "participants": user["user_id"]},
        {"_id": 0}
    ).sort("last_message_at", -1).to_list(100)
    
    enriched_chats = []
    for chat in chats:
        chat_data = serialize_doc(chat)
        
        # Add chat settings
        chat_settings_data = settings_map.get(chat["chat_id"], {})
        chat_data["is_muted"] = chat_settings_data.get("is_muted", False)
        chat_data["is_pinned"] = chat_settings_data.get("is_pinned", False)
        chat_data["is_archived"] = chat_settings_data.get("is_archived", False)
        chat_data["sounds_enabled"] = chat_settings_data.get("sounds_enabled", True)
        chat_data["unread_count"] = chat_settings_data.get("unread_count", 0)
        
        # Get other participant info for direct chats
        if chat["chat_type"] == ChatType.DIRECT.value:
            other_participants = [p for p in chat["participants"] if p != user["user_id"]]
            if other_participants:
                other_user = await db.users.find_one(
                    {"user_id": other_participants[0]},
                    {"_id": 0, "hashed_password": 0}
                )
                if other_user:
                    other_user_data = serialize_doc(other_user)
                    other_user_data["is_online"] = manager.is_online(other_participants[0])
                    chat_data["other_user"] = other_user_data
        
        # Get last message
        last_message = await db.messages.find_one(
            {"chat_id": chat["chat_id"], "deleted_for_everyone": {"$ne": True}},
            {"_id": 0},
            sort=[("created_at", -1)]
        )
        if last_message:
            chat_data["last_message"] = serialize_doc(last_message)
        
        enriched_chats.append(chat_data)
    
    # Sort by pinned first, then by last message
    enriched_chats.sort(key=lambda x: (not x.get("is_pinned", False), x.get("last_message_at") or ""), reverse=True)
    
    return enriched_chats

@api_router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """Get a specific chat with details"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "participants": user["user_id"]},
        {"_id": 0}
    )
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat_data = serialize_doc(chat)
    
    # Get chat settings
    chat_settings = await db.chat_settings.find_one(
        {"chat_id": chat_id, "user_id": user["user_id"]},
        {"_id": 0}
    )
    if chat_settings:
        chat_data.update({
            "is_muted": chat_settings.get("is_muted", False),
            "is_pinned": chat_settings.get("is_pinned", False),
            "is_archived": chat_settings.get("is_archived", False),
            "sounds_enabled": chat_settings.get("sounds_enabled", True),
            "unread_count": chat_settings.get("unread_count", 0)
        })
    
    # Get participants info
    participants_info = []
    for pid in chat["participants"]:
        p_user = await db.users.find_one(
            {"user_id": pid},
            {"_id": 0, "user_id": 1, "name": 1, "picture": 1, "status_message": 1}
        )
        if p_user:
            p_data = serialize_doc(p_user)
            p_data["is_online"] = manager.is_online(pid)
            p_data["is_admin"] = pid in chat.get("admins", [])
            participants_info.append(p_data)
    
    chat_data["participants_info"] = participants_info
    
    # Get other user for direct chats
    if chat["chat_type"] == ChatType.DIRECT.value:
        other_participants = [p for p in chat["participants"] if p != user["user_id"]]
        if other_participants:
            other_user = await db.users.find_one(
                {"user_id": other_participants[0]},
                {"_id": 0, "hashed_password": 0}
            )
            if other_user:
                other_user_data = serialize_doc(other_user)
                other_user_data["is_online"] = manager.is_online(other_participants[0])
                chat_data["other_user"] = other_user_data
    
    return chat_data

@api_router.patch("/chats/{chat_id}/settings")
async def update_chat_settings(
    chat_id: str,
    req: ChatSettingsRequest,
    user: dict = Depends(get_current_user)
):
    """Update chat settings for current user"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "participants": user["user_id"]},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    await db.chat_settings.update_one(
        {"chat_id": chat_id, "user_id": user["user_id"]},
        {"$set": update_data},
        upsert=True
    )
    
    settings = await db.chat_settings.find_one(
        {"chat_id": chat_id, "user_id": user["user_id"]},
        {"_id": 0}
    )
    return serialize_doc(settings)

@api_router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user: dict = Depends(get_current_user)):
    """Delete/leave a chat"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "participants": user["user_id"]},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat["chat_type"] == ChatType.GROUP.value:
        # Remove user from participants
        await db.chats.update_one(
            {"chat_id": chat_id},
            {"$pull": {"participants": user["user_id"], "admins": user["user_id"]}}
        )
        # Add system message
        system_msg = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "chat_id": chat_id,
            "sender_id": "system",
            "content": f"{user['name']} left the group",
            "message_type": MessageType.SYSTEM.value,
            "created_at": datetime.now(timezone.utc)
        }
        await db.messages.insert_one(system_msg)
    else:
        # For direct chats, just archive
        await db.chat_settings.update_one(
            {"chat_id": chat_id, "user_id": user["user_id"]},
            {"$set": {"is_archived": True}},
            upsert=True
        )
    
    return {"message": "Chat deleted/left successfully"}

# ================================
# Group Chat Endpoints
# ================================

@api_router.post("/groups")
async def create_group(req: CreateGroupRequest, user: dict = Depends(get_current_user)):
    """Create a new group chat"""
    participants = list(set([user["user_id"]] + req.participants))
    
    chat_doc = {
        "chat_id": f"chat_{uuid.uuid4().hex[:12]}",
        "chat_type": ChatType.GROUP.value,
        "participants": participants,
        "name": req.name,
        "description": req.description,
        "picture": req.picture,
        "created_by": user["user_id"],
        "created_at": datetime.now(timezone.utc),
        "last_message_at": datetime.now(timezone.utc),
        "admins": [user["user_id"]],
        "settings": {
            "only_admins_can_send": False,
            "only_admins_can_edit": True
        }
    }
    
    await db.chats.insert_one(chat_doc.copy())
    
    # Create chat settings for all participants
    for pid in participants:
        await db.chat_settings.update_one(
            {"chat_id": chat_doc["chat_id"], "user_id": pid},
            {"$set": {
                "chat_id": chat_doc["chat_id"],
                "user_id": pid,
                "is_muted": False,
                "is_pinned": False,
                "is_archived": False,
                "sounds_enabled": True,
                "unread_count": 0
            }},
            upsert=True
        )
    
    # Add system message
    system_msg = {
        "message_id": f"msg_{uuid.uuid4().hex[:12]}",
        "chat_id": chat_doc["chat_id"],
        "sender_id": "system",
        "content": f"{user['name']} created the group \"{req.name}\"",
        "message_type": MessageType.SYSTEM.value,
        "created_at": datetime.now(timezone.utc)
    }
    await db.messages.insert_one(system_msg)
    
    return serialize_doc(chat_doc)

@api_router.patch("/groups/{chat_id}")
async def update_group(chat_id: str, req: UpdateGroupRequest, user: dict = Depends(get_current_user)):
    """Update group info (admin only)"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "chat_type": ChatType.GROUP.value},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if user["user_id"] not in chat.get("admins", []):
        raise HTTPException(status_code=403, detail="Only admins can update group info")
    
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if update_data:
        await db.chats.update_one(
            {"chat_id": chat_id},
            {"$set": update_data}
        )
    
    updated_chat = await db.chats.find_one({"chat_id": chat_id}, {"_id": 0})
    return serialize_doc(updated_chat)

@api_router.post("/groups/{chat_id}/participants")
async def add_group_participants(
    chat_id: str,
    participant_ids: List[str],
    user: dict = Depends(get_current_user)
):
    """Add participants to group"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "chat_type": ChatType.GROUP.value},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if user["user_id"] not in chat.get("admins", []):
        raise HTTPException(status_code=403, detail="Only admins can add participants")
    
    new_participants = [pid for pid in participant_ids if pid not in chat["participants"]]
    
    if new_participants:
        await db.chats.update_one(
            {"chat_id": chat_id},
            {"$addToSet": {"participants": {"$each": new_participants}}}
        )
        
        # Create chat settings for new participants
        for pid in new_participants:
            await db.chat_settings.update_one(
                {"chat_id": chat_id, "user_id": pid},
                {"$set": {
                    "chat_id": chat_id,
                    "user_id": pid,
                    "is_muted": False,
                    "is_pinned": False,
                    "sounds_enabled": True,
                    "unread_count": 0
                }},
                upsert=True
            )
        
        # Get names for system message
        added_users = await db.users.find(
            {"user_id": {"$in": new_participants}},
            {"_id": 0, "name": 1}
        ).to_list(100)
        added_names = ", ".join([u["name"] for u in added_users])
        
        system_msg = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "chat_id": chat_id,
            "sender_id": "system",
            "content": f"{user['name']} added {added_names}",
            "message_type": MessageType.SYSTEM.value,
            "created_at": datetime.now(timezone.utc)
        }
        await db.messages.insert_one(system_msg)
    
    return {"message": f"Added {len(new_participants)} participants"}

@api_router.delete("/groups/{chat_id}/participants/{participant_id}")
async def remove_group_participant(
    chat_id: str,
    participant_id: str,
    user: dict = Depends(get_current_user)
):
    """Remove a participant from group"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "chat_type": ChatType.GROUP.value},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Group not found")
    
    is_admin = user["user_id"] in chat.get("admins", [])
    is_self = participant_id == user["user_id"]
    
    if not is_admin and not is_self:
        raise HTTPException(status_code=403, detail="Only admins can remove participants")
    
    await db.chats.update_one(
        {"chat_id": chat_id},
        {"$pull": {"participants": participant_id, "admins": participant_id}}
    )
    
    removed_user = await db.users.find_one({"user_id": participant_id}, {"_id": 0, "name": 1})
    removed_name = removed_user["name"] if removed_user else "User"
    
    action = "left" if is_self else f"was removed by {user['name']}"
    system_msg = {
        "message_id": f"msg_{uuid.uuid4().hex[:12]}",
        "chat_id": chat_id,
        "sender_id": "system",
        "content": f"{removed_name} {action}",
        "message_type": MessageType.SYSTEM.value,
        "created_at": datetime.now(timezone.utc)
    }
    await db.messages.insert_one(system_msg)
    
    return {"message": f"Participant removed successfully"}

@api_router.post("/groups/{chat_id}/admins/{participant_id}")
async def make_group_admin(
    chat_id: str,
    participant_id: str,
    user: dict = Depends(get_current_user)
):
    """Make a participant an admin"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "chat_type": ChatType.GROUP.value},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Group not found")
    
    if user["user_id"] not in chat.get("admins", []):
        raise HTTPException(status_code=403, detail="Only admins can promote others")
    
    if participant_id not in chat["participants"]:
        raise HTTPException(status_code=400, detail="User is not a participant")
    
    await db.chats.update_one(
        {"chat_id": chat_id},
        {"$addToSet": {"admins": participant_id}}
    )
    
    return {"message": "User promoted to admin"}

# ================================
# Message Endpoints
# ================================

@api_router.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: str,
    req: SendMessageRequest,
    user: dict = Depends(get_current_user)
):
    """Send a message in a chat"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "participants": user["user_id"]},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Check if blocked (for direct chats)
    if chat["chat_type"] == ChatType.DIRECT.value:
        other_id = [p for p in chat["participants"] if p != user["user_id"]][0]
        other_user = await db.users.find_one({"user_id": other_id})
        if other_user and user["user_id"] in other_user.get("blocked_users", []):
            raise HTTPException(status_code=403, detail="You cannot send messages to this user")
    
    message_doc = {
        "message_id": f"msg_{uuid.uuid4().hex[:12]}",
        "chat_id": chat_id,
        "sender_id": user["user_id"],
        "content": req.content,
        "message_type": req.message_type.value,
        "sound_id": req.sound_id,
        "media_url": req.media_url,
        "media_data": req.media_data,
        "duration": req.duration,
        "reply_to": req.reply_to,
        "forwarded_from": req.forwarded_from,
        "created_at": datetime.now(timezone.utc),
        "status": MessageStatus.SENT.value,
        "delivered_to": [],
        "read_by": [user["user_id"]],
        "reactions": [],
        "is_starred": False,
        "is_edited": False,
        "deleted_for": [],
        "deleted_for_everyone": False
    }
    
    await db.messages.insert_one(message_doc.copy())
    
    # Update chat's last message
    await db.chats.update_one(
        {"chat_id": chat_id},
        {"$set": {
            "last_message_at": datetime.now(timezone.utc),
            "last_message": {
                "content": req.content,
                "message_type": req.message_type.value,
                "sender_id": user["user_id"],
                "created_at": datetime.now(timezone.utc)
            }
        }}
    )
    
    # Update unread count for other participants
    for pid in chat["participants"]:
        if pid != user["user_id"]:
            await db.chat_settings.update_one(
                {"chat_id": chat_id, "user_id": pid},
                {"$inc": {"unread_count": 1}},
                upsert=True
            )
    
    # Get reply message if exists
    reply_message = None
    if req.reply_to:
        reply_msg = await db.messages.find_one({"message_id": req.reply_to}, {"_id": 0})
        if reply_msg:
            reply_message = serialize_doc(reply_msg)
    
    message_response = serialize_doc(message_doc)
    message_response["sender"] = serialize_doc(user)
    message_response["reply_message"] = reply_message
    
    # Broadcast to other participants via WebSocket
    await manager.broadcast_to_chat(
        {"type": "new_message", "data": message_response},
        chat_id,
        exclude_user=user["user_id"]
    )
    
    # Update delivery status for online users
    for pid in chat["participants"]:
        if pid != user["user_id"] and manager.is_online(pid):
            await db.messages.update_one(
                {"message_id": message_doc["message_id"]},
                {"$addToSet": {"delivered_to": pid}}
            )
    
    return message_response

@api_router.get("/chats/{chat_id}/messages")
async def get_messages(
    chat_id: str,
    limit: int = 50,
    before: str = None,
    user: dict = Depends(get_current_user)
):
    """Get messages for a chat with pagination"""
    chat = await db.chats.find_one(
        {"chat_id": chat_id, "participants": user["user_id"]},
        {"_id": 0}
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    query = {
        "chat_id": chat_id,
        "deleted_for": {"$ne": user["user_id"]},
        "deleted_for_everyone": {"$ne": True}
    }
    
    if before:
        before_msg = await db.messages.find_one({"message_id": before})
        if before_msg:
            query["created_at"] = {"$lt": before_msg["created_at"]}
    
    messages = await db.messages.find(
        query,
        {"_id": 0}
    ).sort("created_at", -1).limit(limit).to_list(limit)
    
    messages.reverse()
    
    # Mark messages as read
    message_ids = [m["message_id"] for m in messages if user["user_id"] not in m.get("read_by", [])]
    if message_ids:
        await db.messages.update_many(
            {"message_id": {"$in": message_ids}},
            {"$addToSet": {"read_by": user["user_id"]}}
        )
        
        # Send read receipts via WebSocket
        for msg in messages:
            if msg["sender_id"] != user["user_id"]:
                await manager.send_personal_message({
                    "type": "message_read",
                    "message_id": msg["message_id"],
                    "chat_id": chat_id,
                    "read_by": user["user_id"]
                }, msg["sender_id"])
    
    # Reset unread count
    await db.chat_settings.update_one(
        {"chat_id": chat_id, "user_id": user["user_id"]},
        {"$set": {"unread_count": 0}}
    )
    
    # Enrich messages with sender info and reply messages
    result = []
    for msg in messages:
        msg_data = serialize_doc(msg)
        
        # Get sender info
        sender = await db.users.find_one(
            {"user_id": msg["sender_id"]},
            {"_id": 0, "user_id": 1, "name": 1, "picture": 1}
        )
        if sender:
            msg_data["sender"] = serialize_doc(sender)
        
        # Get reply message if exists
        if msg.get("reply_to"):
            reply_msg = await db.messages.find_one(
                {"message_id": msg["reply_to"]},
                {"_id": 0, "message_id": 1, "content": 1, "sender_id": 1, "message_type": 1}
            )
            if reply_msg:
                reply_sender = await db.users.find_one(
                    {"user_id": reply_msg["sender_id"]},
                    {"_id": 0, "name": 1}
                )
                reply_data = serialize_doc(reply_msg)
                reply_data["sender_name"] = reply_sender["name"] if reply_sender else "Unknown"
                msg_data["reply_message"] = reply_data
        
        result.append(msg_data)
    
    return result

@api_router.patch("/messages/{message_id}/status")
async def update_message_status(
    message_id: str,
    status: MessageStatus,
    user: dict = Depends(get_current_user)
):
    """Update message delivery/read status"""
    message = await db.messages.find_one({"message_id": message_id}, {"_id": 0})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    update = {}
    if status == MessageStatus.DELIVERED:
        update["$addToSet"] = {"delivered_to": user["user_id"]}
    elif status == MessageStatus.READ:
        update["$addToSet"] = {"read_by": user["user_id"]}
    
    if update:
        await db.messages.update_one({"message_id": message_id}, update)
        
        # Notify sender
        await manager.send_personal_message({
            "type": f"message_{status.value}",
            "message_id": message_id,
            "user_id": user["user_id"]
        }, message["sender_id"])
    
    return {"status": "updated"}

@api_router.post("/messages/{message_id}/reaction")
async def add_reaction(
    message_id: str,
    req: MessageReactionRequest,
    user: dict = Depends(get_current_user)
):
    """Add a reaction to a message"""
    message = await db.messages.find_one({"message_id": message_id}, {"_id": 0})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Check if user is in the chat
    chat = await db.chats.find_one(
        {"chat_id": message["chat_id"], "participants": user["user_id"]}
    )
    if not chat:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    reaction = {
        "user_id": user["user_id"],
        "emoji": req.emoji,
        "sound_id": req.sound_id,
        "created_at": datetime.now(timezone.utc)
    }
    
    # Remove existing reaction from this user and add new one
    await db.messages.update_one(
        {"message_id": message_id},
        {"$pull": {"reactions": {"user_id": user["user_id"]}}}
    )
    await db.messages.update_one(
        {"message_id": message_id},
        {"$push": {"reactions": reaction}}
    )
    
    # Broadcast reaction
    await manager.broadcast_to_chat({
        "type": "message_reaction",
        "message_id": message_id,
        "reaction": {
            "user_id": user["user_id"],
            "user_name": user["name"],
            "emoji": req.emoji,
            "sound_id": req.sound_id
        }
    }, message["chat_id"])
    
    return {"message": "Reaction added"}

@api_router.delete("/messages/{message_id}/reaction")
async def remove_reaction(message_id: str, user: dict = Depends(get_current_user)):
    """Remove reaction from a message"""
    await db.messages.update_one(
        {"message_id": message_id},
        {"$pull": {"reactions": {"user_id": user["user_id"]}}}
    )
    return {"message": "Reaction removed"}

@api_router.post("/messages/{message_id}/star")
async def toggle_star_message(message_id: str, user: dict = Depends(get_current_user)):
    """Toggle star on a message"""
    # Check if already starred
    starred = await db.starred_messages.find_one({
        "message_id": message_id,
        "user_id": user["user_id"]
    })
    
    if starred:
        await db.starred_messages.delete_one({
            "message_id": message_id,
            "user_id": user["user_id"]
        })
        return {"starred": False}
    else:
        await db.starred_messages.insert_one({
            "message_id": message_id,
            "user_id": user["user_id"],
            "created_at": datetime.now(timezone.utc)
        })
        return {"starred": True}

@api_router.get("/messages/starred")
async def get_starred_messages(user: dict = Depends(get_current_user)):
    """Get all starred messages"""
    starred = await db.starred_messages.find(
        {"user_id": user["user_id"]},
        {"_id": 0}
    ).to_list(100)
    
    message_ids = [s["message_id"] for s in starred]
    messages = await db.messages.find(
        {"message_id": {"$in": message_ids}},
        {"_id": 0}
    ).to_list(100)
    
    result = []
    for msg in messages:
        msg_data = serialize_doc(msg)
        msg_data["is_starred"] = True
        
        # Get chat info
        chat = await db.chats.find_one({"chat_id": msg["chat_id"]}, {"_id": 0, "name": 1, "chat_type": 1})
        if chat:
            msg_data["chat_name"] = chat.get("name")
            msg_data["chat_type"] = chat.get("chat_type")
        
        result.append(msg_data)
    
    return result

@api_router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str,
    for_everyone: bool = False,
    user: dict = Depends(get_current_user)
):
    """Delete a message"""
    message = await db.messages.find_one({"message_id": message_id}, {"_id": 0})
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    if for_everyone:
        # Only sender can delete for everyone
        if message["sender_id"] != user["user_id"]:
            raise HTTPException(status_code=403, detail="Only sender can delete for everyone")
        
        # Check time limit (e.g., 1 hour)
        msg_time = message["created_at"]
        if isinstance(msg_time, str):
            msg_time = datetime.fromisoformat(msg_time)
        # Ensure timezone-aware comparison
        if msg_time.tzinfo is None:
            msg_time = msg_time.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - msg_time > timedelta(hours=1):
            raise HTTPException(status_code=400, detail="Cannot delete message after 1 hour")
        
        await db.messages.update_one(
            {"message_id": message_id},
            {"$set": {
                "deleted_for_everyone": True,
                "content": "This message was deleted",
                "media_url": None,
                "media_data": None
            }}
        )
        
        # Broadcast deletion
        await manager.broadcast_to_chat({
            "type": "message_deleted",
            "message_id": message_id,
            "for_everyone": True
        }, message["chat_id"])
    else:
        # Delete only for this user
        await db.messages.update_one(
            {"message_id": message_id},
            {"$addToSet": {"deleted_for": user["user_id"]}}
        )
    
    return {"message": "Message deleted"}

@api_router.post("/messages/{message_id}/forward")
async def forward_message(
    message_id: str,
    chat_ids: List[str],
    user: dict = Depends(get_current_user)
):
    """Forward a message to multiple chats"""
    original_message = await db.messages.find_one({"message_id": message_id}, {"_id": 0})
    if not original_message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    forwarded_messages = []
    for chat_id in chat_ids:
        chat = await db.chats.find_one(
            {"chat_id": chat_id, "participants": user["user_id"]},
            {"_id": 0}
        )
        if not chat:
            continue
        
        new_message = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "chat_id": chat_id,
            "sender_id": user["user_id"],
            "content": original_message["content"],
            "message_type": original_message["message_type"],
            "sound_id": original_message.get("sound_id"),
            "media_url": original_message.get("media_url"),
            "media_data": original_message.get("media_data"),
            "forwarded_from": original_message["message_id"],
            "created_at": datetime.now(timezone.utc),
            "status": MessageStatus.SENT.value,
            "delivered_to": [],
            "read_by": [user["user_id"]],
            "reactions": []
        }
        
        await db.messages.insert_one(new_message.copy())
        forwarded_messages.append(serialize_doc(new_message))
        
        # Update chat's last message
        await db.chats.update_one(
            {"chat_id": chat_id},
            {"$set": {"last_message_at": datetime.now(timezone.utc)}}
        )
    
    return {"forwarded_count": len(forwarded_messages), "messages": forwarded_messages}

# ================================
# Search Endpoints
# ================================

@api_router.post("/search/messages")
async def search_messages(req: SearchRequest, user: dict = Depends(get_current_user)):
    """Search messages across chats"""
    query = {
        "content": {"$regex": req.query, "$options": "i"},
        "deleted_for": {"$ne": user["user_id"]},
        "deleted_for_everyone": {"$ne": True}
    }
    
    if req.chat_id:
        # Verify user has access to this chat
        chat = await db.chats.find_one(
            {"chat_id": req.chat_id, "participants": user["user_id"]}
        )
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        query["chat_id"] = req.chat_id
    else:
        # Only search in user's chats
        user_chats = await db.chats.find(
            {"participants": user["user_id"]},
            {"_id": 0, "chat_id": 1}
        ).to_list(1000)
        chat_ids = [c["chat_id"] for c in user_chats]
        query["chat_id"] = {"$in": chat_ids}
    
    messages = await db.messages.find(
        query,
        {"_id": 0}
    ).sort("created_at", -1).limit(req.limit).to_list(req.limit)
    
    result = []
    for msg in messages:
        msg_data = serialize_doc(msg)
        
        # Get chat info
        chat = await db.chats.find_one({"chat_id": msg["chat_id"]}, {"_id": 0, "name": 1, "chat_type": 1, "participants": 1})
        if chat:
            if chat["chat_type"] == ChatType.DIRECT.value:
                other_id = [p for p in chat["participants"] if p != user["user_id"]][0]
                other_user = await db.users.find_one({"user_id": other_id}, {"_id": 0, "name": 1})
                msg_data["chat_name"] = other_user["name"] if other_user else "Unknown"
            else:
                msg_data["chat_name"] = chat.get("name")
            msg_data["chat_type"] = chat.get("chat_type")
        
        result.append(msg_data)
    
    return result

@api_router.get("/search/chats")
async def search_chats(query: str, user: dict = Depends(get_current_user)):
    """Search chats by name"""
    # Search in group names
    groups = await db.chats.find({
        "participants": user["user_id"],
        "chat_type": ChatType.GROUP.value,
        "name": {"$regex": query, "$options": "i"}
    }, {"_id": 0}).to_list(50)
    
    # Search in user names for direct chats
    matching_users = await db.users.find({
        "name": {"$regex": query, "$options": "i"},
        "user_id": {"$ne": user["user_id"]}
    }, {"_id": 0, "user_id": 1}).to_list(50)
    
    matching_user_ids = [u["user_id"] for u in matching_users]
    
    direct_chats = await db.chats.find({
        "participants": {"$all": [user["user_id"]], "$in": matching_user_ids},
        "chat_type": ChatType.DIRECT.value
    }, {"_id": 0}).to_list(50)
    
    all_chats = groups + direct_chats
    result = []
    
    for chat in all_chats:
        chat_data = serialize_doc(chat)
        
        if chat["chat_type"] == ChatType.DIRECT.value:
            other_id = [p for p in chat["participants"] if p != user["user_id"]][0]
            other_user = await db.users.find_one({"user_id": other_id}, {"_id": 0, "hashed_password": 0})
            if other_user:
                chat_data["other_user"] = serialize_doc(other_user)
        
        result.append(chat_data)
    
    return result

# ================================
# Voice/Video Call Endpoints
# ================================

@api_router.post("/calls/initiate")
async def initiate_call(req: InitiateCallRequest, user: dict = Depends(get_current_user)):
    """Initiate a voice or video call"""
    call_doc = {
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "initiator_id": user["user_id"],
        "participant_ids": req.participant_ids,
        "call_type": req.call_type.value,
        "status": CallStatus.INITIATED.value,
        "started_at": datetime.now(timezone.utc),
        "answered_at": None,
        "ended_at": None,
        "duration": None,
        "quality_stats": None,
    }
    
    await db.calls.insert_one(call_doc.copy())
    
    # Notify participants
    for pid in req.participant_ids:
        await manager.send_personal_message({
            "type": "incoming_call",
            "call": serialize_doc(call_doc),
            "caller": {
                "user_id": user["user_id"],
                "name": user["name"],
                "picture": user.get("picture")
            }
        }, pid)
    
    return serialize_doc(call_doc)

@api_router.post("/calls/{call_id}/answer")
async def answer_call(call_id: str, user: dict = Depends(get_current_user)):
    """Answer a call"""
    call = await db.calls.find_one({"call_id": call_id}, {"_id": 0})
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if user["user_id"] not in call["participant_ids"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    now = datetime.now(timezone.utc)
    await db.calls.update_one(
        {"call_id": call_id},
        {"$set": {"status": CallStatus.ONGOING.value, "answered_at": now}}
    )
    
    # Notify initiator
    await manager.send_personal_message({
        "type": "call_answered",
        "call_id": call_id,
        "answered_by": user["user_id"]
    }, call["initiator_id"])
    
    return {"message": "Call answered"}

@api_router.post("/calls/{call_id}/decline")
async def decline_call(call_id: str, user: dict = Depends(get_current_user)):
    """Decline a call"""
    call = await db.calls.find_one({"call_id": call_id}, {"_id": 0})
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    now = datetime.now(timezone.utc)
    await db.calls.update_one(
        {"call_id": call_id},
        {"$set": {"status": CallStatus.DECLINED.value, "ended_at": now}}
    )

    # Increment missed call counter for all non-declining participants
    all_ids = [call["initiator_id"]] + call["participant_ids"]
    for pid in all_ids:
        if pid != user["user_id"]:
            await db.users.update_one({"user_id": pid}, {"$inc": {"missed_calls": 1}})
    
    # Notify initiator
    await manager.send_personal_message({
        "type": "call_declined",
        "call_id": call_id,
        "declined_by": user["user_id"]
    }, call["initiator_id"])
    
    return {"message": "Call declined"}

@api_router.post("/calls/{call_id}/end")
async def end_call(call_id: str, user: dict = Depends(get_current_user)):
    """End an ongoing call"""
    call = await db.calls.find_one({"call_id": call_id}, {"_id": 0})
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    answered_at = call.get("answered_at")
    if isinstance(answered_at, str):
        answered_at = datetime.fromisoformat(answered_at)
    
    duration = None
    if answered_at:
        if answered_at.tzinfo is None:
            answered_at = answered_at.replace(tzinfo=timezone.utc)
        duration = (datetime.now(timezone.utc) - answered_at).total_seconds()
    else:
        # Never answered — this is a missed/cancelled call
        started_at = call.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        if started_at:
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)

        # Mark missed for the non-ending party
        all_ids = [call["initiator_id"]] + call["participant_ids"]
        for pid in all_ids:
            if pid != user["user_id"]:
                await db.users.update_one({"user_id": pid}, {"$inc": {"missed_calls": 1}})

    now = datetime.now(timezone.utc)
    await db.calls.update_one(
        {"call_id": call_id},
        {"$set": {"status": CallStatus.ENDED.value, "ended_at": now, "duration": duration}}
    )
    
    # Notify all participants
    all_participants = [call["initiator_id"]] + call["participant_ids"]
    for pid in all_participants:
        if pid != user["user_id"]:
            await manager.send_personal_message({
                "type": "call_ended",
                "call_id": call_id,
                "ended_by": user["user_id"],
                "duration": duration
            }, pid)
    
    return {"message": "Call ended", "duration": duration}

@api_router.post("/calls/{call_id}/analytics")
async def save_call_analytics(call_id: str, request: Request, user: dict = Depends(get_current_user)):
    """Save call quality analytics"""
    body = await request.json()
    await db.calls.update_one(
        {"call_id": call_id},
        {"$set": {"quality_stats": body}}
    )
    return {"message": "Analytics saved"}

@api_router.get("/calls/missed-count")
async def get_missed_call_count(user: dict = Depends(get_current_user)):
    """Get missed call count"""
    u = await db.users.find_one({"user_id": user["user_id"]}, {"_id": 0, "missed_calls": 1})
    return {"count": u.get("missed_calls", 0) if u else 0}

@api_router.post("/calls/clear-missed")
async def clear_missed_calls(user: dict = Depends(get_current_user)):
    """Clear missed call counter"""
    await db.users.update_one({"user_id": user["user_id"]}, {"$set": {"missed_calls": 0}})
    return {"message": "Cleared"}

@api_router.get("/calls/history")
async def get_call_history(user: dict = Depends(get_current_user)):
    """Get call history"""
    calls = await db.calls.find({
        "$or": [
            {"initiator_id": user["user_id"]},
            {"participant_ids": user["user_id"]}
        ]
    }, {"_id": 0}).sort("started_at", -1).limit(50).to_list(50)
    
    result = []
    for call in calls:
        call_data = serialize_doc(call)
        
        # Get other participant info
        if call["initiator_id"] == user["user_id"]:
            other_ids = call["participant_ids"]
        else:
            other_ids = [call["initiator_id"]]
        
        other_users = await db.users.find(
            {"user_id": {"$in": other_ids}},
            {"_id": 0, "user_id": 1, "name": 1, "picture": 1}
        ).to_list(10)
        call_data["participants"] = [serialize_doc(u) for u in other_users]
        call_data["is_outgoing"] = call["initiator_id"] == user["user_id"]
        
        result.append(call_data)
    
    return result

# ================================
# Sound Pack Endpoints
# ================================

@api_router.get("/sound-packs")
async def get_sound_packs(user: dict = Depends(get_current_user)):
    """Get all available sound packs"""
    packs = await db.sound_packs.find({}, {"_id": 0}).to_list(100)
    
    if not packs:
        await seed_sound_packs()
        packs = await db.sound_packs.find({}, {"_id": 0}).to_list(100)
    
    return [serialize_doc(p) for p in packs]

@api_router.get("/sound-packs/{pack_id}")
async def get_sound_pack(pack_id: str, user: dict = Depends(get_current_user)):
    """Get a specific sound pack"""
    pack = await db.sound_packs.find_one({"pack_id": pack_id}, {"_id": 0})
    if not pack:
        raise HTTPException(status_code=404, detail="Sound pack not found")
    return serialize_doc(pack)

async def seed_sound_packs():
    """Seed initial sound packs"""
    sound_packs = [
        {
            "pack_id": "emotions",
            "name": "Emotions",
            "category": "basic",
            "is_premium": False,
            "sounds": [
                {"sound_id": "laugh_1", "name": "Haha", "emoji": "😂", "duration": 1.5, "description": "Funny laugh"},
                {"sound_id": "wow_1", "name": "Wow!", "emoji": "😮", "duration": 1.0, "description": "Surprised wow"},
                {"sound_id": "sad_1", "name": "Aww", "emoji": "😢", "duration": 1.2, "description": "Sad sigh"},
                {"sound_id": "love_1", "name": "Smooch", "emoji": "😘", "duration": 0.8, "description": "Kiss sound"},
                {"sound_id": "angry_1", "name": "Grr", "emoji": "😤", "duration": 1.0, "description": "Angry growl"},
                {"sound_id": "cool_1", "name": "Yeah", "emoji": "😎", "duration": 1.2, "description": "Cool yeah"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "pack_id": "reactions",
            "name": "Reactions",
            "category": "basic",
            "is_premium": False,
            "sounds": [
                {"sound_id": "clap_1", "name": "Clap", "emoji": "👏", "duration": 1.5, "description": "Applause"},
                {"sound_id": "cheer_1", "name": "Cheer", "emoji": "🎉", "duration": 2.0, "description": "Celebration"},
                {"sound_id": "thumbsup_1", "name": "Nice!", "emoji": "👍", "duration": 0.8, "description": "Approval sound"},
                {"sound_id": "fire_1", "name": "Fire", "emoji": "🔥", "duration": 1.0, "description": "On fire!"},
                {"sound_id": "mindblown_1", "name": "Mind Blown", "emoji": "🤯", "duration": 1.5, "description": "Explosion"},
                {"sound_id": "eye_roll_1", "name": "Eye Roll", "emoji": "🙄", "duration": 1.0, "description": "Whatever"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "pack_id": "fun_sounds",
            "name": "Fun Sounds",
            "category": "entertainment",
            "is_premium": False,
            "sounds": [
                {"sound_id": "boing_1", "name": "Boing", "emoji": "🏀", "duration": 0.5, "description": "Bouncy sound"},
                {"sound_id": "slide_1", "name": "Slide", "emoji": "🎵", "duration": 0.8, "description": "Slide whistle"},
                {"sound_id": "pop_1", "name": "Pop", "emoji": "💥", "duration": 0.3, "description": "Pop sound"},
                {"sound_id": "ding_1", "name": "Ding", "emoji": "🔔", "duration": 0.5, "description": "Bell ding"},
                {"sound_id": "whoosh_1", "name": "Whoosh", "emoji": "💨", "duration": 0.7, "description": "Swoosh sound"},
                {"sound_id": "tada_1", "name": "Ta-da!", "emoji": "✨", "duration": 1.0, "description": "Reveal sound"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "pack_id": "animals",
            "name": "Animals",
            "category": "entertainment",
            "is_premium": False,
            "sounds": [
                {"sound_id": "meow_1", "name": "Meow", "emoji": "🐱", "duration": 0.8, "description": "Cat meow"},
                {"sound_id": "woof_1", "name": "Woof", "emoji": "🐕", "duration": 0.6, "description": "Dog bark"},
                {"sound_id": "moo_1", "name": "Moo", "emoji": "🐄", "duration": 1.2, "description": "Cow moo"},
                {"sound_id": "quack_1", "name": "Quack", "emoji": "🦆", "duration": 0.5, "description": "Duck quack"},
                {"sound_id": "roar_1", "name": "Roar", "emoji": "🦁", "duration": 1.5, "description": "Lion roar"},
                {"sound_id": "chirp_1", "name": "Chirp", "emoji": "🐦", "duration": 0.8, "description": "Bird chirp"},
            ],
            "created_at": datetime.now(timezone.utc)
        }
    ]
    
    for pack in sound_packs:
        await db.sound_packs.update_one(
            {"pack_id": pack["pack_id"]},
            {"$set": pack},
            upsert=True
        )
    
    logger.info("Sound packs seeded successfully")


# ================================
# Sticker Pack Endpoints
# ================================

@api_router.get("/sticker-packs")
async def get_sticker_packs(user: dict = Depends(get_current_user)):
    """Get all available sticker packs"""
    packs = await db.sticker_packs.find({}, {"_id": 0}).to_list(100)
    if not packs:
        await seed_sticker_packs()
        packs = await db.sticker_packs.find({}, {"_id": 0}).to_list(100)
    return [serialize_doc(p) for p in packs]

async def seed_sticker_packs():
    """Seed built-in sticker packs with sound mappings"""
    sticker_packs = [
        {
            "pack_id": "sound_vibes",
            "name": "Sound Vibes",
            "icon": "speaker_high",
            "stickers": [
                {"sticker_id": "sv1", "art": "\U0001f602", "name": "LOL", "sound_id": "laugh_1"},
                {"sticker_id": "sv2", "art": "\U0001f62e", "name": "OMG", "sound_id": "wow_1"},
                {"sticker_id": "sv3", "art": "\U0001f622", "name": "Sad", "sound_id": "sad_1"},
                {"sticker_id": "sv4", "art": "\U0001f618", "name": "Kiss", "sound_id": "love_1"},
                {"sticker_id": "sv5", "art": "\U0001f624", "name": "Angry", "sound_id": "angry_1"},
                {"sticker_id": "sv6", "art": "\U0001f60e", "name": "Cool", "sound_id": "cool_1"},
                {"sticker_id": "sv7", "art": "\U0001f44f", "name": "Clap", "sound_id": "clap_1"},
                {"sticker_id": "sv8", "art": "\U0001f389", "name": "Party", "sound_id": "cheer_1"},
                {"sticker_id": "sv9", "art": "\U0001f525", "name": "Fire", "sound_id": "fire_1"},
                {"sticker_id": "sv10", "art": "\U0001f92f", "name": "Mind Blown", "sound_id": "mindblown_1"},
                {"sticker_id": "sv11", "art": "\U0001f44d", "name": "Nice", "sound_id": "thumbsup_1"},
                {"sticker_id": "sv12", "art": "\u2728", "name": "Magic", "sound_id": "tada_1"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "pack_id": "animals_sound",
            "name": "Animal Sounds",
            "icon": "paw",
            "stickers": [
                {"sticker_id": "as1", "art": "\U0001f431", "name": "Cat", "sound_id": "meow_1"},
                {"sticker_id": "as2", "art": "\U0001f436", "name": "Dog", "sound_id": "woof_1"},
                {"sticker_id": "as3", "art": "\U0001f404", "name": "Cow", "sound_id": "moo_1"},
                {"sticker_id": "as4", "art": "\U0001f986", "name": "Duck", "sound_id": "quack_1"},
                {"sticker_id": "as5", "art": "\U0001f981", "name": "Lion", "sound_id": "roar_1"},
                {"sticker_id": "as6", "art": "\U0001f426", "name": "Bird", "sound_id": "chirp_1"},
                {"sticker_id": "as7", "art": "\U0001f438", "name": "Frog"},
                {"sticker_id": "as8", "art": "\U0001f435", "name": "Monkey"},
                {"sticker_id": "as9", "art": "\U0001f98a", "name": "Fox"},
                {"sticker_id": "as10", "art": "\U0001f43b", "name": "Bear"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
        {
            "pack_id": "moods",
            "name": "Moods",
            "icon": "dizzy",
            "stickers": [
                {"sticker_id": "m1", "art": "\U0001f970", "name": "In Love"},
                {"sticker_id": "m2", "art": "\U0001f929", "name": "Star Eyes"},
                {"sticker_id": "m3", "art": "\U0001f973", "name": "Party"},
                {"sticker_id": "m4", "art": "\U0001f917", "name": "Hug"},
                {"sticker_id": "m5", "art": "\U0001f634", "name": "Sleepy"},
                {"sticker_id": "m6", "art": "\U0001f914", "name": "Thinking"},
                {"sticker_id": "m7", "art": "\U0001f97a", "name": "Pleading"},
                {"sticker_id": "m8", "art": "\U0001f608", "name": "Naughty"},
                {"sticker_id": "m9", "art": "\U0001f911", "name": "Rich"},
                {"sticker_id": "m10", "art": "\U0001f976", "name": "Freezing"},
            ],
            "created_at": datetime.now(timezone.utc)
        },
    ]
    for pack in sticker_packs:
        await db.sticker_packs.update_one(
            {"pack_id": pack["pack_id"]},
            {"$set": pack},
            upsert=True
        )
    logger.info("Sticker packs seeded successfully")


# ================================
# AI Mood Analysis Endpoint
# ================================

@api_router.post("/analyze-mood", response_model=MoodAnalysisResponse)
async def analyze_mood(req: MoodAnalysisRequest, user: dict = Depends(get_current_user)):
    """Analyze text mood and suggest emojis/sounds using GPT-5.2"""
    try:
        if not EMERGENT_LLM_KEY or LlmChat is None or UserMessage is None:
            return get_fallback_suggestions(req.text)
        
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"mood_{user['user_id']}_{uuid.uuid4().hex[:8]}",
            system_message="""You are a mood analyzer for a chat app. Analyze the given text and respond with ONLY a JSON object (no markdown, no explanation):
{
    "mood": "one of: happy, sad, excited, angry, loving, surprised, neutral, funny, cool",
    "suggested_emojis": ["emoji1", "emoji2", "emoji3"],
    "suggested_sounds": [
        {"sound_id": "sound_id_from_list", "name": "sound_name"},
        {"sound_id": "sound_id_from_list", "name": "sound_name"}
    ]
}

Available sounds:
- laugh_1 (Haha 😂), wow_1 (Wow! 😮), sad_1 (Aww 😢), love_1 (Smooch 😘), angry_1 (Grr 😤), cool_1 (Yeah 😎)
- clap_1 (Clap 👏), cheer_1 (Cheer 🎉), thumbsup_1 (Nice! 👍), fire_1 (Fire 🔥), mindblown_1 (Mind Blown 🤯)
- boing_1 (Boing 🏀), pop_1 (Pop 💥), ding_1 (Ding 🔔), whoosh_1 (Whoosh 💨), tada_1 (Ta-da! ✨)
- meow_1 (Meow 🐱), woof_1 (Woof 🐕), roar_1 (Roar 🦁)

Pick 2-3 emojis and 1-2 sounds that match the mood of the text."""
        ).with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(text=f"Analyze this message: \"{req.text}\"")
        response = await chat.send_message(user_message)
        
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response.split("```")[1]
                if cleaned_response.startswith("json"):
                    cleaned_response = cleaned_response[4:]
            
            result = json.loads(cleaned_response)
            return MoodAnalysisResponse(
                mood=result.get("mood", "neutral"),
                suggested_emojis=result.get("suggested_emojis", ["😊"]),
                suggested_sounds=result.get("suggested_sounds", [])
            )
        except json.JSONDecodeError:
            return get_fallback_suggestions(req.text)
            
    except Exception as e:
        logger.error(f"Mood analysis error: {e}")
        return get_fallback_suggestions(req.text)

def get_fallback_suggestions(text: str) -> MoodAnalysisResponse:
    """Fallback mood detection using simple keyword matching"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["haha", "lol", "funny", "😂", "joke", "hilarious"]):
        return MoodAnalysisResponse(
            mood="funny",
            suggested_emojis=["😂", "🤣", "😆"],
            suggested_sounds=[{"sound_id": "laugh_1", "name": "Haha"}]
        )
    elif any(word in text_lower for word in ["love", "miss", "heart", "❤", "😘", "adore"]):
        return MoodAnalysisResponse(
            mood="loving",
            suggested_emojis=["❤️", "😘", "🥰"],
            suggested_sounds=[{"sound_id": "love_1", "name": "Smooch"}]
        )
    elif any(word in text_lower for word in ["wow", "amazing", "incredible", "awesome", "great"]):
        return MoodAnalysisResponse(
            mood="excited",
            suggested_emojis=["😮", "🤩", "🔥"],
            suggested_sounds=[{"sound_id": "wow_1", "name": "Wow!"}, {"sound_id": "fire_1", "name": "Fire"}]
        )
    elif any(word in text_lower for word in ["sad", "sorry", "😢", "upset", "disappointed"]):
        return MoodAnalysisResponse(
            mood="sad",
            suggested_emojis=["😢", "😔", "💔"],
            suggested_sounds=[{"sound_id": "sad_1", "name": "Aww"}]
        )
    elif any(word in text_lower for word in ["congrat", "yay", "celebrate", "🎉", "won", "success"]):
        return MoodAnalysisResponse(
            mood="excited",
            suggested_emojis=["🎉", "🥳", "👏"],
            suggested_sounds=[{"sound_id": "cheer_1", "name": "Cheer"}, {"sound_id": "tada_1", "name": "Ta-da!"}]
        )
    elif any(word in text_lower for word in ["angry", "mad", "furious", "😤", "hate"]):
        return MoodAnalysisResponse(
            mood="angry",
            suggested_emojis=["😤", "😠", "💢"],
            suggested_sounds=[{"sound_id": "angry_1", "name": "Grr"}]
        )
    else:
        return MoodAnalysisResponse(
            mood="neutral",
            suggested_emojis=["😊", "👍", "✨"],
            suggested_sounds=[{"sound_id": "thumbsup_1", "name": "Nice!"}]
        )

# ================================
# Settings Endpoints
# ================================

@api_router.get("/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    """Get user settings"""
    return user.get("settings", {
        "master_volume": 1.0,
        "night_mode_enabled": False,
        "night_mode_start": "22:00",
        "night_mode_end": "07:00",
        "last_seen_visible": True,
        "read_receipts_enabled": True,
        "typing_indicator_enabled": True
    })

@api_router.patch("/settings")
async def update_settings(req: UpdateSettingsRequest, user: dict = Depends(get_current_user)):
    """Update user settings"""
    update_data = {f"settings.{k}": v for k, v in req.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    await db.users.update_one(
        {"user_id": user["user_id"]},
        {"$set": update_data}
    )
    
    updated_user = await db.users.find_one({"user_id": user["user_id"]}, {"_id": 0})
    return updated_user.get("settings", {})

# ================================
# WebSocket Endpoint
# ================================

@api_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time messaging"""
    await manager.connect(websocket, user_id)
    
    # Update user online status in DB
    await db.users.update_one(
        {"user_id": user_id},
        {"$set": {"is_online": True, "last_seen": datetime.now(timezone.utc)}}
    )
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "typing":
                chat_id = data.get("chat_id")
                if chat_id:
                    # Update user status
                    manager.user_status[user_id] = {
                        "status": UserStatus.TYPING,
                        "last_seen": datetime.now(timezone.utc),
                        "typing_in": chat_id
                    }
                    await manager.broadcast_to_chat(
                        {"type": "typing", "user_id": user_id, "chat_id": chat_id},
                        chat_id,
                        exclude_user=user_id
                    )
            
            elif data.get("type") == "stop_typing":
                chat_id = data.get("chat_id")
                if chat_id:
                    manager.user_status[user_id] = {
                        "status": UserStatus.ONLINE,
                        "last_seen": datetime.now(timezone.utc)
                    }
                    await manager.broadcast_to_chat(
                        {"type": "stop_typing", "user_id": user_id, "chat_id": chat_id},
                        chat_id,
                        exclude_user=user_id
                    )
            
            elif data.get("type") == "read":
                chat_id = data.get("chat_id")
                message_ids = data.get("message_ids", [])
                if chat_id and message_ids:
                    # Update messages as read
                    await db.messages.update_many(
                        {"message_id": {"$in": message_ids}},
                        {"$addToSet": {"read_by": user_id}}
                    )
                    await manager.broadcast_to_chat(
                        {"type": "read", "user_id": user_id, "chat_id": chat_id, "message_ids": message_ids},
                        chat_id,
                        exclude_user=user_id
                    )
            
            elif data.get("type") == "delivered":
                message_id = data.get("message_id")
                if message_id:
                    await db.messages.update_one(
                        {"message_id": message_id},
                        {"$addToSet": {"delivered_to": user_id}}
                    )
                    message = await db.messages.find_one({"message_id": message_id}, {"_id": 0})
                    if message:
                        await manager.send_personal_message(
                            {"type": "delivered", "message_id": message_id, "user_id": user_id},
                            message["sender_id"]
                        )
            
            # WebRTC Signaling
            elif data.get("type") == "webrtc_offer":
                target_id = data.get("target_id")
                if target_id:
                    await manager.send_personal_message({
                        "type": "webrtc_offer",
                        "from_id": user_id,
                        "call_id": data.get("call_id"),
                        "sdp": data.get("sdp")
                    }, target_id)

            elif data.get("type") == "webrtc_answer":
                target_id = data.get("target_id")
                if target_id:
                    await manager.send_personal_message({
                        "type": "webrtc_answer",
                        "from_id": user_id,
                        "call_id": data.get("call_id"),
                        "sdp": data.get("sdp")
                    }, target_id)

            elif data.get("type") == "webrtc_ice_candidate":
                target_id = data.get("target_id")
                if target_id:
                    await manager.send_personal_message({
                        "type": "webrtc_ice_candidate",
                        "from_id": user_id,
                        "call_id": data.get("call_id"),
                        "candidate": data.get("candidate")
                    }, target_id)

            elif data.get("type") == "webrtc_ice_restart":
                target_id = data.get("target_id")
                if target_id:
                    await manager.send_personal_message({
                        "type": "webrtc_ice_restart",
                        "from_id": user_id,
                        "call_id": data.get("call_id"),
                        "sdp": data.get("sdp")
                    }, target_id)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                    
    except WebSocketDisconnect:
        manager.disconnect(user_id)
        # Update user offline status in DB
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"is_online": False, "last_seen": datetime.now(timezone.utc)}}
        )
        # Broadcast offline status
        await manager.broadcast_status(user_id, UserStatus.OFFLINE)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(user_id)
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"is_online": False, "last_seen": datetime.now(timezone.utc)}}
        )

# ================================
# Health Check
# ================================

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0", "timestamp": datetime.now(timezone.utc).isoformat()}

# Include the router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    # User indexes
    await db.users.create_index("user_id", unique=True)
    await db.users.create_index("email", unique=True)
    await db.users.create_index("phone", sparse=True)
    
    # Session indexes
    await db.user_sessions.create_index("token")
    await db.user_sessions.create_index("user_id")
    
    # Chat indexes
    await db.chats.create_index("chat_id", unique=True)
    await db.chats.create_index("participants")
    await db.chats.create_index("chat_type")
    
    # Chat settings indexes
    await db.chat_settings.create_index([("chat_id", 1), ("user_id", 1)], unique=True)
    
    # Message indexes
    await db.messages.create_index("message_id", unique=True)
    await db.messages.create_index("chat_id")
    await db.messages.create_index("sender_id")
    await db.messages.create_index("created_at")
    await db.messages.create_index([("chat_id", 1), ("created_at", -1)])
    await db.messages.create_index([("content", "text")])
    
    # Starred messages index
    await db.starred_messages.create_index([("message_id", 1), ("user_id", 1)], unique=True)
    
    # Call indexes
    await db.calls.create_index("call_id", unique=True)
    await db.calls.create_index("initiator_id")
    await db.calls.create_index("participant_ids")
    
    # Sound pack indexes
    await db.sound_packs.create_index("pack_id", unique=True)
    
    # Block/Report indexes
    await db.blocked_users.create_index([("blocker_id", 1), ("blocked_id", 1)])
    await db.reports.create_index("report_id", unique=True)
    
    await db.sticker_packs.create_index("pack_id", unique=True)
    await seed_sound_packs()
    await seed_sticker_packs()
    logger.info("Database indexes created and packs seeded")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
