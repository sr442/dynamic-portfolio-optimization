
import requests

def test_api():
    print("Testing API Health...")
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        print(r.json())
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    print("\nTesting Optimization Endpoint...")
    payload = {
        "tickers": ["AAPL", "MSFT"], # Fewer tickers for speed
        "capital": 100000,
        "optimizer": "Mean-Variance", 
        "lookback_days": 100 # Smaller lookback
    }
    try:
        r = requests.post("http://127.0.0.1:8000/api/optimize", json=payload, timeout=30)
        if r.status_code == 200:
            print("Optimization Success!")
            print(r.json())
        else:
            print(f"Optimization Failed: {r.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
