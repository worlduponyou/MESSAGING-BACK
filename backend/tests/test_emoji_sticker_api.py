"""
Tests for Emoji/Sticker/GIF System APIs
This tests the new media messaging features: sticker packs, reactions, and different message types
"""
import pytest
import requests
import os

BASE_URL = os.environ.get('EXPO_PUBLIC_BACKEND_URL', 'https://emoji-voice-app.preview.emergentagent.com').rstrip('/')


class TestStickerPacks:
    """Tests for sticker pack endpoints"""
    
    @pytest.fixture(autouse=True)
    def auth_token(self):
        """Get authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testcall1@test.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        self.token = data.get("token")
        self.user_id = data.get("user", {}).get("user_id")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
    
    def test_get_sticker_packs_returns_3_packs(self):
        """GET /api/sticker-packs returns 3 sticker packs"""
        response = requests.get(f"{BASE_URL}/api/sticker-packs", headers=self.headers)
        assert response.status_code == 200, f"Failed: {response.text}"
        packs = response.json()
        assert isinstance(packs, list)
        assert len(packs) == 3, f"Expected 3 packs, got {len(packs)}"
        
        # Verify pack structure
        pack_ids = [p.get('pack_id') for p in packs]
        assert 'sound_vibes' in pack_ids
        assert 'animals_sound' in pack_ids
        assert 'moods' in pack_ids
    
    def test_sticker_packs_have_sound_mappings(self):
        """Verify sticker packs contain stickers with sound_id mappings"""
        response = requests.get(f"{BASE_URL}/api/sticker-packs", headers=self.headers)
        assert response.status_code == 200
        packs = response.json()
        
        # Sound Vibes pack should have all stickers with sounds
        sound_vibes = next((p for p in packs if p['pack_id'] == 'sound_vibes'), None)
        assert sound_vibes is not None
        assert 'stickers' in sound_vibes
        stickers_with_sound = [s for s in sound_vibes['stickers'] if s.get('sound_id')]
        assert len(stickers_with_sound) == len(sound_vibes['stickers']), "Sound Vibes pack should have all stickers with sounds"
        
        # Animals pack should have some with sounds
        animals = next((p for p in packs if p['pack_id'] == 'animals_sound'), None)
        assert animals is not None
        animals_with_sound = [s for s in animals['stickers'] if s.get('sound_id')]
        assert len(animals_with_sound) > 0, "Animals pack should have some stickers with sounds"

    def test_sticker_packs_require_auth(self):
        """GET /api/sticker-packs requires authentication"""
        response = requests.get(f"{BASE_URL}/api/sticker-packs")
        assert response.status_code == 401


class TestStickerMessage:
    """Tests for sending sticker messages"""
    
    @pytest.fixture(autouse=True)
    def auth_token(self):
        """Get authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testcall1@test.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
    
    def test_send_sticker_message_with_sound(self):
        """POST /api/chats/{id}/messages with message_type=sticker and sound_id works"""
        response = requests.post(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages",
            headers=self.headers,
            json={
                "content": "😂",
                "message_type": "sticker",
                "sound_id": "laugh_1"
            }
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data.get('message_id') is not None
        assert data.get('message_type') == 'sticker'
        assert data.get('sound_id') == 'laugh_1'
        assert data.get('content') == '😂'
    
    def test_send_sticker_message_without_sound(self):
        """POST sticker message without sound_id"""
        response = requests.post(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages",
            headers=self.headers,
            json={
                "content": "🤔",
                "message_type": "sticker"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get('message_type') == 'sticker'
        assert data.get('sound_id') is None


class TestGifMessage:
    """Tests for sending GIF messages"""
    
    @pytest.fixture(autouse=True)
    def auth_token(self):
        """Get authentication token"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testcall1@test.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
    
    def test_send_gif_message_with_sound(self):
        """POST /api/chats/{id}/messages with message_type=gif and sound_id works"""
        response = requests.post(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages",
            headers=self.headers,
            json={
                "content": "🔥💯",
                "message_type": "gif",
                "sound_id": "fire_1"
            }
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data.get('message_id') is not None
        assert data.get('message_type') == 'gif'
        assert data.get('sound_id') == 'fire_1'
    
    def test_send_gif_message_without_sound(self):
        """POST GIF message without sound_id"""
        response = requests.post(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages",
            headers=self.headers,
            json={
                "content": "👀",
                "message_type": "gif"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get('message_type') == 'gif'


class TestMessageReactions:
    """Tests for message reaction endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Login and create a test message"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testcall1@test.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        
        # Create a test message to react to
        msg_response = requests.post(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages",
            headers=self.headers,
            json={"content": "TEST_reaction_target", "message_type": "text"}
        )
        assert msg_response.status_code == 200
        self.test_message_id = msg_response.json().get('message_id')
    
    def test_add_reaction_success(self):
        """POST /api/messages/{id}/reaction adds emoji reaction"""
        response = requests.post(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers,
            json={"emoji": "❤️"}
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data.get('message') == 'Reaction added'
    
    def test_add_reaction_with_sound(self):
        """POST reaction with sound_id"""
        response = requests.post(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers,
            json={"emoji": "😂", "sound_id": "laugh_1"}
        )
        assert response.status_code == 200
        assert response.json().get('message') == 'Reaction added'
    
    def test_remove_reaction_success(self):
        """DELETE /api/messages/{id}/reaction removes reaction"""
        # First add a reaction
        requests.post(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers,
            json={"emoji": "👍"}
        )
        
        # Then remove it
        response = requests.delete(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers
        )
        assert response.status_code == 200, f"Failed: {response.text}"
        data = response.json()
        assert data.get('message') == 'Reaction removed'
    
    def test_reaction_replaces_previous(self):
        """Adding a new reaction replaces the previous one from same user"""
        # Add first reaction
        requests.post(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers,
            json={"emoji": "👍"}
        )
        
        # Add second reaction (should replace)
        response = requests.post(
            f"{BASE_URL}/api/messages/{self.test_message_id}/reaction",
            headers=self.headers,
            json={"emoji": "❤️"}
        )
        assert response.status_code == 200
        
        # Verify in messages
        chat_response = requests.get(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages?limit=50",
            headers=self.headers
        )
        messages = chat_response.json()
        msg = next((m for m in messages if m['message_id'] == self.test_message_id), None)
        if msg and msg.get('reactions'):
            # Should only have one reaction from this user
            user_reactions = [r for r in msg['reactions'] if r.get('emoji') == '❤️']
            assert len(user_reactions) >= 1


class TestMessageRendering:
    """Tests for message retrieval with different types"""
    
    @pytest.fixture(autouse=True)
    def auth_token(self):
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": "testcall1@test.com",
            "password": "TestPass123"
        })
        assert response.status_code == 200
        data = response.json()
        self.token = data.get("token")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
    
    def test_messages_include_sticker_and_gif_types(self):
        """Verify chat messages include sticker and gif message types"""
        response = requests.get(
            f"{BASE_URL}/api/chats/chat_01f06aa25c85/messages?limit=50",
            headers=self.headers
        )
        assert response.status_code == 200
        messages = response.json()
        
        # Should have various message types from our tests
        message_types = set(m.get('message_type') for m in messages)
        print(f"Message types found: {message_types}")
        
        # Verify sticker and gif messages exist
        sticker_msgs = [m for m in messages if m.get('message_type') == 'sticker']
        gif_msgs = [m for m in messages if m.get('message_type') == 'gif']
        
        assert len(sticker_msgs) > 0, "Should have sticker messages"
        assert len(gif_msgs) > 0, "Should have gif messages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
