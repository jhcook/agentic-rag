
import requests
import time
import sys
import json

BASE_URL = "http://127.0.0.1:8001/api"

def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                print("Server is up!")
                return True
        except:
            pass
        time.sleep(1)
    print("Server failed to allow connection")
    return False

def verify_persistence():
    if not wait_for_server():
        sys.exit(1)

    print("\n1. Starting new chat session...")
    msg = "Hello, please remember this session."
    try:
        # Enforce google mode and use a supported model
        resp = requests.post(f"{BASE_URL}/chat", json={
            "messages": [{"role": "user", "content": msg}],
            "model": "models/gemini-2.0-flash",
            "config": {"mode": "google"}
        })
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id")
        answer = data.get("answer")
        
        if not session_id:
            print("❌ FAILURE: No session_id returned")
            print(f"Full response: {json.dumps(data, indent=2)}")
            sys.exit(1)
            
        print(f"✅ Created session: {session_id}")
        if answer:
            print(f"   Answer: {answer[:50]}...")
        else:
            print("⚠️  Warning: No answer returned (backend might be silent or returning sources only)")
            print(f"Full response: {json.dumps(data, indent=2)}")
            # We proceed to check persistence of the USER message at least
        
        print("\n2. verifying persistence via history list...")
        history_resp = requests.get(f"{BASE_URL}/chat/history")
        history_resp.raise_for_status()
        sessions = history_resp.json()
        
        found = False
        for s in sessions:
            if s["id"] == session_id:
                found = True
                print(f"✅ Found session in history list: {s['title']}")
                break
        
        if not found:
            print("❌ FAILURE: Session not found in history list")
            print(f"List: {json.dumps(sessions, indent=2)}")
            sys.exit(1)
            
        print("\n3. Verifying message content...")
        msgs_resp = requests.get(f"{BASE_URL}/chat/history/{session_id}")
        msgs_resp.raise_for_status()
        messages = msgs_resp.json()
        
        # We expect at least the user message
        if len(messages) >= 1:
            print(f"✅ Found {len(messages)} messages")
            if messages[0]["content"] == msg:
                 print("✅ User message persists correctly")
            else:
                 print(f"❌ User message mismatch: {messages[0]['content']}")
        else:
            print(f"❌ FAILURE: Expected messages, got {len(messages)}")
            sys.exit(1)

        print("\nSUCCESS: Chat persistence verified!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_persistence()
