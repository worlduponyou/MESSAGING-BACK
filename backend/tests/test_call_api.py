"""
SoundChat Call API Tests
Tests call-related endpoints: initiate, answer, decline, end, history
"""
import pytest
import requests
import os
import uuid
import time

# Get base URL from environment
BASE_URL = os.environ.get('EXPO_PUBLIC_BACKEND_URL', 'https://emoji-voice-app.preview.emergentagent.com').rstrip('/')

# Test credentials
TEST_USER1 = {"email": "alice@test.com", "password": "TestPass123"}
TEST_USER2 = {"email": "bob@test.com", "password": "TestPass123"}
ALICE_USER_ID = "user_28bcca91dfae"
BOB_USER_ID = "user_fd078634be09"


@pytest.fixture
def alice_token():
    """Get authentication token for alice"""
    response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER1)
    assert response.status_code == 200, f"Alice login failed: {response.text}"
    return response.json()["token"]


@pytest.fixture
def bob_token():
    """Get authentication token for bob"""
    response = requests.post(f"{BASE_URL}/api/auth/login", json=TEST_USER2)
    assert response.status_code == 200, f"Bob login failed: {response.text}"
    return response.json()["token"]


class TestCallInitiate:
    """Test call initiation endpoint"""

    def test_initiate_call_success(self, alice_token):
        """POST /api/calls/initiate - Successfully initiate a voice call"""
        response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        assert response.status_code == 200, f"Initiate call failed: {response.text}"
        
        data = response.json()
        assert "call_id" in data, "Response should contain call_id"
        assert data["initiator_id"] == ALICE_USER_ID
        assert BOB_USER_ID in data["participant_ids"]
        assert data["call_type"] == "voice"
        assert data["status"] == "initiated"
        assert "started_at" in data
        
        # Store for cleanup
        return data["call_id"]

    def test_initiate_call_without_auth(self):
        """POST /api/calls/initiate - Should fail without authentication"""
        response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        assert response.status_code == 401


class TestCallAnswer:
    """Test call answer endpoint"""

    def test_answer_call_success(self, alice_token, bob_token):
        """POST /api/calls/{call_id}/answer - Successfully answer an incoming call"""
        # First, alice initiates a call to bob
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        assert init_response.status_code == 200
        call_id = init_response.json()["call_id"]
        
        # Bob answers the call
        response = requests.post(
            f"{BASE_URL}/api/calls/{call_id}/answer",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 200, f"Answer call failed: {response.text}"
        
        data = response.json()
        assert data["message"] == "Call answered"

    def test_answer_nonexistent_call(self, bob_token):
        """POST /api/calls/{call_id}/answer - Should fail for non-existent call"""
        response = requests.post(
            f"{BASE_URL}/api/calls/nonexistent_call_id/answer",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 404

    def test_answer_call_unauthorized_user(self, alice_token, bob_token):
        """POST /api/calls/{call_id}/answer - Should fail for unauthorized participant"""
        # Create a third user scenario - alice calls bob, but we try to answer as alice (the initiator)
        # The initiator is not in participant_ids, only bob is
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        call_id = init_response.json()["call_id"]
        
        # Alice (initiator) tries to answer - should fail since she's not a participant
        response = requests.post(
            f"{BASE_URL}/api/calls/{call_id}/answer",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert response.status_code == 403


class TestCallDecline:
    """Test call decline endpoint"""

    def test_decline_call_success(self, alice_token, bob_token):
        """POST /api/calls/{call_id}/decline - Successfully decline a call"""
        # Alice initiates a call to bob
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        call_id = init_response.json()["call_id"]
        
        # Bob declines the call
        response = requests.post(
            f"{BASE_URL}/api/calls/{call_id}/decline",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 200, f"Decline call failed: {response.text}"
        
        data = response.json()
        assert data["message"] == "Call declined"

    def test_decline_nonexistent_call(self, bob_token):
        """POST /api/calls/{call_id}/decline - Should fail for non-existent call"""
        response = requests.post(
            f"{BASE_URL}/api/calls/nonexistent_call_id/decline",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        assert response.status_code == 404


class TestCallEnd:
    """Test call end endpoint"""

    def test_end_call_success(self, alice_token, bob_token):
        """POST /api/calls/{call_id}/end - Successfully end a call"""
        # Alice initiates a call
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        call_id = init_response.json()["call_id"]
        
        # Bob answers
        requests.post(
            f"{BASE_URL}/api/calls/{call_id}/answer",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        
        # Wait a moment to accumulate duration
        time.sleep(1)
        
        # Alice ends the call
        response = requests.post(
            f"{BASE_URL}/api/calls/{call_id}/end",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert response.status_code == 200, f"End call failed: {response.text}"
        
        data = response.json()
        assert data["message"] == "Call ended"
        assert "duration" in data
        assert data["duration"] >= 0  # Should have some duration

    def test_end_nonexistent_call(self, alice_token):
        """POST /api/calls/{call_id}/end - Should fail for non-existent call"""
        response = requests.post(
            f"{BASE_URL}/api/calls/nonexistent_call_id/end",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert response.status_code == 404


class TestCallHistory:
    """Test call history endpoint"""

    def test_get_call_history(self, alice_token):
        """GET /api/calls/history - Get user's call history"""
        response = requests.get(
            f"{BASE_URL}/api/calls/history",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        assert response.status_code == 200, f"Get call history failed: {response.text}"
        
        data = response.json()
        assert isinstance(data, list)
        
        # If there are calls, verify structure
        if len(data) > 0:
            call = data[0]
            assert "call_id" in call
            assert "initiator_id" in call
            assert "participant_ids" in call
            assert "call_type" in call
            assert "status" in call
            assert "started_at" in call
            assert "participants" in call  # Should have participant info
            assert "is_outgoing" in call  # Direction indicator

    def test_call_history_shows_participant_info(self, alice_token, bob_token):
        """GET /api/calls/history - Verify participant details are included"""
        # Create a call first
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        call_id = init_response.json()["call_id"]
        
        # End it
        requests.post(
            f"{BASE_URL}/api/calls/{call_id}/end",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        
        # Get history for alice
        response = requests.get(
            f"{BASE_URL}/api/calls/history",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        data = response.json()
        
        # Find the call we just made
        our_call = next((c for c in data if c["call_id"] == call_id), None)
        assert our_call is not None, f"Our call not found in history"
        
        # Verify it shows as outgoing for alice
        assert our_call["is_outgoing"] == True
        
        # Verify participant info is included
        assert len(our_call["participants"]) > 0
        participant = our_call["participants"][0]
        assert "user_id" in participant
        assert "name" in participant

    def test_call_history_direction_for_receiver(self, alice_token, bob_token):
        """GET /api/calls/history - Verify incoming call direction for receiver"""
        # Alice calls bob
        init_response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "voice"
            }
        )
        call_id = init_response.json()["call_id"]
        
        # End call
        requests.post(
            f"{BASE_URL}/api/calls/{call_id}/end",
            headers={"Authorization": f"Bearer {alice_token}"}
        )
        
        # Get history for bob (receiver)
        response = requests.get(
            f"{BASE_URL}/api/calls/history",
            headers={"Authorization": f"Bearer {bob_token}"}
        )
        data = response.json()
        
        # Find the call
        our_call = next((c for c in data if c["call_id"] == call_id), None)
        assert our_call is not None, f"Call not found in Bob's history"
        
        # Verify it shows as incoming for bob
        assert our_call["is_outgoing"] == False

    def test_call_history_without_auth(self):
        """GET /api/calls/history - Should fail without authentication"""
        response = requests.get(f"{BASE_URL}/api/calls/history")
        assert response.status_code == 401


class TestCallTypeValidation:
    """Test call type validation"""

    def test_initiate_video_call(self, alice_token):
        """POST /api/calls/initiate - Initiate a video call"""
        response = requests.post(
            f"{BASE_URL}/api/calls/initiate",
            headers={"Authorization": f"Bearer {alice_token}"},
            json={
                "participant_ids": [BOB_USER_ID],
                "call_type": "video"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["call_type"] == "video"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
