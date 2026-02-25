"""
SoundChat Backend API Tests
Tests all major endpoints: Auth, Chats, Messages, Groups, Sound Packs, Settings
"""
import pytest
import requests
import os
import uuid
import time

# Get base URL from environment or use default
BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', 'https://emoji-voice-app.preview.emergentagent.com').rstrip('/')

# Test credentials
TEST_USER1 = {"email": "alice@test.com", "password": "TestPass123"}
TEST_USER2 = {"email": "bob@test.com", "password": "TestPass123"}
EXISTING_CHAT_ID = "chat_697a859122cd"
ALICE_USER_ID = "user_28bcca91dfae"
BOB_USER_ID = "user_fd078634be09"


class TestAuthEndpoints:
    """Test authentication endpoints: register, login, me"""

    def test_login_success_alice(self):
        """POST /api/auth/login - Login with valid credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        assert response.status_code == 200, f"Login failed: {response.text}"
        
        data = response.json()
        assert "token" in data, "Response should contain token"
        assert "user" in data, "Response should contain user"
        assert data["user"]["email"] == TEST_USER1["email"]
        assert data["user"]["user_id"] == ALICE_USER_ID
        assert len(data["token"]) > 0

    def test_login_success_bob(self):
        """POST /api/auth/login - Login bob user"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER2)
        assert response.status_code == 200, f"Login failed: {response.text}"
        
        data = response.json()
        assert data["user"]["email"] == TEST_USER2["email"]
        assert data["user"]["user_id"] == BOB_USER_ID

    def test_login_invalid_credentials(self):
        """POST /api/auth/login - Should fail with wrong credentials"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "wrong@test.com",
            "password": "wrongpassword"
        })
        assert response.status_code == 401

    def test_get_me_with_valid_token(self):
        """GET /api/auth/me - Verify token returns user data"""
        # First login to get token
        login_resp = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        token = login_resp.json()["token"]
        
        # Call /me endpoint
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200, f"Get me failed: {response.text}"
        
        data = response.json()
        assert data["email"] == TEST_USER1["email"]
        assert data["user_id"] == ALICE_USER_ID

    def test_get_me_without_token(self):
        """GET /api/auth/me - Should fail without token"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401

    def test_register_duplicate_email(self):
        """POST /api/auth/register - Should fail with duplicate email"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": TEST_USER1["email"],
            "password": "TestPass123",
            "name": "Duplicate User"
        })
        assert response.status_code == 400
        assert "already registered" in response.json().get("detail", "").lower()


class TestChatEndpoints:
    """Test chat endpoints: create chat, list chats"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_get_chats_list(self, auth_token):
        """GET /api/chats - List user chats"""
        response = requests.get(
            f"{BASE_URL}/api/chats",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get chats failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list)
        # Alice should have at least the existing chat with Bob
        chat_ids = [c["chat_id"] for c in data]
        assert EXISTING_CHAT_ID in chat_ids, f"Expected chat {EXISTING_CHAT_ID} not found in {chat_ids}"

    def test_get_specific_chat(self, auth_token):
        """GET /api/chats/{chat_id} - Get specific chat details"""
        response = requests.get(
            f"{BASE_URL}/api/chats/{EXISTING_CHAT_ID}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get chat failed: {response.text}"
        
        data = response.json()
        assert data["chat_id"] == EXISTING_CHAT_ID
        assert data["chat_type"] == "direct"
        assert ALICE_USER_ID in data["participants"]
        assert BOB_USER_ID in data["participants"]

    def test_create_direct_chat_returns_existing(self, auth_token):
        """POST /api/chats - Creating duplicate direct chat returns existing one"""
        response = requests.post(
            f"{BASE_URL}/api/chats",
            headers={"Authorization": f"Bearer {auth_token}"},
            params={"participant_id": BOB_USER_ID, "chat_type": "direct"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["chat_id"] == EXISTING_CHAT_ID


class TestMessageEndpoints:
    """Test message endpoints: send and get messages"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_send_text_message(self, auth_token):
        """POST /api/chats/{chat_id}/messages - Send text message"""
        unique_content = f"Test message {uuid.uuid4().hex[:8]}"
        
        response = requests.post(
            f"{BASE_URL}/api/chats/{EXISTING_CHAT_ID}/messages",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "content": unique_content,
                "message_type": "text"
            }
        )
        assert response.status_code == 200, f"Send message failed: {response.text}"
        
        data = response.json()
        assert data["content"] == unique_content
        assert data["message_type"] == "text"
        assert data["sender_id"] == ALICE_USER_ID
        assert data["chat_id"] == EXISTING_CHAT_ID
        assert "message_id" in data

    def test_send_sound_message(self, auth_token):
        """POST /api/chats/{chat_id}/messages - Send sound message"""
        response = requests.post(
            f"{BASE_URL}/api/chats/{EXISTING_CHAT_ID}/messages",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "content": "Laugh sound",
                "message_type": "sound",
                "sound_id": "laugh_1"
            }
        )
        assert response.status_code == 200, f"Send sound failed: {response.text}"
        
        data = response.json()
        assert data["message_type"] == "sound"
        assert data["sound_id"] == "laugh_1"

    def test_get_messages_in_chat(self, auth_token):
        """GET /api/chats/{chat_id}/messages - Get messages"""
        response = requests.get(
            f"{BASE_URL}/api/chats/{EXISTING_CHAT_ID}/messages",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get messages failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0, "Should have at least one message"
        
        # Verify message structure
        msg = data[-1]  # Get latest message
        assert "message_id" in msg
        assert "content" in msg
        assert "sender_id" in msg
        assert "message_type" in msg


class TestGroupEndpoints:
    """Test group chat endpoints"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_create_group(self, auth_token):
        """POST /api/groups - Create a group chat"""
        group_name = f"Test Group {uuid.uuid4().hex[:6]}"
        
        response = requests.post(
            f"{BASE_URL}/api/groups",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "name": group_name,
                "description": "A test group",
                "participants": [BOB_USER_ID]
            }
        )
        assert response.status_code == 200, f"Create group failed: {response.text}"
        
        data = response.json()
        assert data["name"] == group_name
        assert data["chat_type"] == "group"
        assert ALICE_USER_ID in data["participants"]
        assert BOB_USER_ID in data["participants"]
        assert ALICE_USER_ID in data["admins"]


class TestSoundPackEndpoints:
    """Test sound pack endpoints"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_get_sound_packs(self, auth_token):
        """GET /api/sound-packs - Returns sound packs with sounds"""
        response = requests.get(
            f"{BASE_URL}/api/sound-packs",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get sound packs failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0, "Should have at least one sound pack"
        
        # Verify structure of first pack
        pack = data[0]
        assert "pack_id" in pack
        assert "name" in pack
        assert "sounds" in pack
        assert isinstance(pack["sounds"], list)
        assert len(pack["sounds"]) > 0
        
        # Verify sound structure
        sound = pack["sounds"][0]
        assert "sound_id" in sound
        assert "name" in sound
        assert "emoji" in sound
        assert "duration" in sound

    def test_sound_packs_contain_expected_packs(self, auth_token):
        """Verify expected sound packs exist"""
        response = requests.get(
            f"{BASE_URL}/api/sound-packs",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        data = response.json()
        
        pack_ids = [p["pack_id"] for p in data]
        assert "emotions" in pack_ids, "Emotions pack should exist"
        assert "reactions" in pack_ids, "Reactions pack should exist"


class TestSettingsEndpoints:
    """Test user settings endpoints"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_get_settings(self, auth_token):
        """GET /api/settings - Returns user settings"""
        response = requests.get(
            f"{BASE_URL}/api/settings",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get settings failed: {response.text}"
        
        data = response.json()
        assert "master_volume" in data
        assert "night_mode_enabled" in data
        assert "read_receipts_enabled" in data

    def test_update_settings(self, auth_token):
        """PATCH /api/settings - Update settings"""
        new_volume = 0.8
        
        response = requests.patch(
            f"{BASE_URL}/api/settings",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"master_volume": new_volume}
        )
        assert response.status_code == 200, f"Update settings failed: {response.text}"
        
        data = response.json()
        assert data["master_volume"] == new_volume

        # Verify persistence with GET
        verify_resp = requests.get(
            f"{BASE_URL}/api/settings",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert verify_resp.json()["master_volume"] == new_volume


class TestUserEndpoints:
    """Test user-related endpoints"""

    @pytest.fixture
    def auth_token(self):
        """Get authentication token for alice"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
        return response.json()["token"]

    def test_get_users(self, auth_token):
        """GET /api/users - Returns list of users"""
        response = requests.get(
            f"{BASE_URL}/api/users",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get users failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list)
        
        # Should include Bob but not Alice (current user)
        user_ids = [u["user_id"] for u in data]
        assert ALICE_USER_ID not in user_ids, "Current user should not be in list"
        assert BOB_USER_ID in user_ids, "Bob should be in users list"

    def test_get_user_by_id(self, auth_token):
        """GET /api/users/{user_id} - Get specific user"""
        response = requests.get(
            f"{BASE_URL}/api/users/{BOB_USER_ID}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200, f"Get user failed: {response.text}"
        
        data = response.json()
        assert data["user_id"] == BOB_USER_ID
        assert data["name"] == "Bob Johnson"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
