import time
import math
from fedpro.api.dashboard_server import start_dashboard_server, broadcast_event

def simulate_dummy_backend():
    print("Starting Dummy Backend API...")
    start_dashboard_server(port=8000)
    print("Dashboard API started on http://127.0.0.1:8000")
    print("Make sure your React frontend is running!")
    
    time.sleep(3)
    
    num_hospitals = 5
    num_rounds = 30
    
    dp_clip_norm = 1.0
    dp_noise_multiplier = 0.5
    epsilon_budget = 0.0
    
    broadcast_event("privacy_config", {
        "clip_norm": dp_clip_norm, 
        "noise_multiplier": dp_noise_multiplier,
        "status": "Enabled (Simulated)"
    })
    
    print(f"Simulating {num_hospitals} hospitals over {num_rounds} rounds...")
    
    loss = 1.5
    accuracy = 0.4
    
    for round_num in range(1, num_rounds + 1):
        print(f"--- Round {round_num} ---")
        broadcast_event("round_start", {"round": round_num, "clients": num_hospitals})
        
        time.sleep(1.5)
        
        loss = loss * 0.85 + (0.05 * (1 - math.exp(-round_num/5)))
        accuracy = min(0.98, accuracy + (0.95 - accuracy) * 0.15 + 0.015 * math.sin(round_num))
        
        broadcast_event("round_end", {"round": round_num, "samples": 500 * num_hospitals})
        broadcast_event("evaluation", {"round": round_num, "loss": loss, "accuracy": accuracy})
        
        epsilon_budget += 0.12
        broadcast_event("privacy_update", {"epsilon": epsilon_budget})
        
        print(f"Evaluation: Loss={loss:.4f}, Acc={accuracy:.4f}, Epsilon={epsilon_budget:.2f}")
        time.sleep(0.5)

    print("Dummy simulation complete! Keeping server alive...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == "__main__":
    simulate_dummy_backend()
