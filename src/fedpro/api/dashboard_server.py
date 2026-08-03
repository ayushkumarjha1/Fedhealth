"""
Production-grade FastAPI and WebSocket Backend for FedHealth.
Provides RESTful APIs and real-time WebSocket telemetry for the React Dashboard.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
import torch
import numpy as np

from fedpro.configs.base_config import FedHealthConfig
from fedpro.core.server import FLServer
from fedpro.models.mlp import HealthcareMLP
from fedpro.data.medical_datasets import load_breast_cancer_data
from fedpro.data.partitioner import dirichlet_non_iid_partition, iid_partition
from fedpro.algorithms.registry import get_algorithm_client_class
from fedpro.utils.logger import get_logger

logger = get_logger("DashboardServer")

app = FastAPI(
    title="FedHealth Clinical AI API",
    version="1.0.0",
    description="Research-grade REST and WebSocket telemetry API for Federated Healthcare Learning."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class SimulationState:
    def __init__(self):
        self.config = FedHealthConfig()
        self.is_running = False
        self.is_paused = False
        self.current_round = 0
        self.server: Optional[FLServer] = None
        self.latest_telemetry: Dict[str, Any] = {
            "round": 0,
            "metrics": {},
            "hospitals": [],
            "privacy": {},
            "copilot": {},
            "xai": []
        }
        self.simulation_task: Optional[asyncio.Task] = None

sim_state = SimulationState()

class ConnectionManager:
    """Thread-safe WebSocket manager broadcasting live simulation telemetry."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket client: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

def sync_telemetry_callback(event: Dict[str, Any]):
    """Synchronous callback from FLServer bridged into asynchronous event loop."""
    data = event.get("data", {})
    event_type = event.get("type", "update")
    
    if event_type == "round_end":
        sim_state.latest_telemetry = data
        sim_state.current_round = data.get("round", sim_state.current_round)
        
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(event), loop)
    except Exception:
        pass

async def run_simulation_worker(config: FedHealthConfig):
    """Background coroutine running the federated training simulation loop."""
    sim_state.is_running = True
    sim_state.config = config
    logger.info(f"Starting simulation worker for experiment: {config.experiment_name}")
    
    try:
        # Load Dataset
        train_data, test_data, input_dim, num_classes, feature_names = load_breast_cancer_data(
            test_size=config.data.test_split_ratio, random_state=config.data.random_seed
        )
        
        # Partition
        if config.data.partition_type == "dirichlet":
            subsets = dirichlet_non_iid_partition(
                train_data, 
                num_clients=config.training.num_hospitals, 
                alpha=config.data.dirichlet_alpha,
                num_classes=num_classes
            )
        else:
            subsets = iid_partition(train_data, num_clients=config.training.num_hospitals)
            
        loaders = [
            torch.utils.data.DataLoader(s, batch_size=config.training.batch_size, shuffle=True)
            for s in subsets
        ]
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.training.batch_size, shuffle=False)
        
        # Model
        global_model = HealthcareMLP(
            input_dim=input_dim,
            hidden_dims=config.model.hidden_dims,
            num_classes=num_classes,
            dropout_rate=config.model.dropout_rate,
            use_batch_norm=config.model.use_batch_norm
        )
        
        # Server
        sim_state.server = FLServer(
            global_model=global_model,
            config=config,
            telemetry_callback=sync_telemetry_callback,
            feature_names=feature_names
        )
        
        # Instantiate Clients
        client_cls = get_algorithm_client_class(config.algorithm.name)
        clients = []
        for i, h in enumerate(sim_state.server.hospitals):
            c_model = HealthcareMLP(
                input_dim=input_dim,
                hidden_dims=config.model.hidden_dims,
                num_classes=num_classes,
                dropout_rate=config.model.dropout_rate,
                use_batch_norm=config.model.use_batch_norm
            )
            c = client_cls(
                client_id=h.id,
                model=c_model,
                device="cpu",
                dp_config=config.privacy
            )
            clients.append(c)
            
        # Simulation Loop
        for r in range(1, config.training.num_rounds + 1):
            if not sim_state.is_running:
                logger.info("Simulation halted by user.")
                break
                
            while sim_state.is_paused and sim_state.is_running:
                await asyncio.sleep(0.5)
                
            # Execute round
            sim_state.server.fit_round(
                round_num=r,
                clients=clients,
                dataloaders=loaders,
                config={
                    "epochs": config.training.local_epochs,
                    "lr": config.training.learning_rate,
                    "momentum": config.training.momentum,
                    "weight_decay": config.training.weight_decay
                }
            )
            
            # Evaluate global model
            sim_state.server.evaluate_round(
                round_num=r,
                evaluator_client=clients[0],
                test_loader=test_loader
            )
            
            # Brief async sleep to allow WebSocket dispatch
            await asyncio.sleep(0.1)
            
        logger.info("Simulation successfully completed all rounds.")
        await manager.broadcast({"type": "simulation_complete", "data": {"status": "Complete"}})
    except Exception as e:
        logger.error(f"Error during simulation execution: {e}", exc_info=True)
        await manager.broadcast({"type": "simulation_error", "data": {"error": str(e)}})
    finally:
        sim_state.is_running = False

# REST API Endpoints
@app.get("/api/health")
def get_health():
    """Health check & compute hardware inspection."""
    return {
        "status": "online",
        "service": "FedHealth Backend",
        "version": "1.0.0",
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "is_simulation_running": sim_state.is_running
    }

@app.get("/api/config")
def get_configuration():
    """Retrieve active simulation hyperparameters."""
    return sim_state.config.to_dict()

@app.post("/api/config")
def update_configuration(new_config: FedHealthConfig):
    """Update simulation configuration."""
    if sim_state.is_running:
        raise HTTPException(status_code=400, detail="Cannot update configuration while simulation is actively running.")
    sim_state.config = new_config
    return {"status": "success", "config": sim_state.config.to_dict()}

@app.get("/api/status")
def get_simulation_status():
    """Get live telemetry and hospital status."""
    return {
        "is_running": sim_state.is_running,
        "is_paused": sim_state.is_paused,
        "current_round": sim_state.current_round,
        "total_rounds": sim_state.config.training.num_rounds,
        "latest_telemetry": sim_state.latest_telemetry
    }

@app.post("/api/control/start")
async def start_simulation(background_tasks: BackgroundTasks, config_override: Optional[FedHealthConfig] = None):
    """Start federated learning simulation."""
    if sim_state.is_running:
        return {"status": "already_running", "message": "Simulation is already active."}
        
    cfg = config_override or sim_state.config
    sim_state.simulation_task = asyncio.create_task(run_simulation_worker(cfg))
    return {"status": "started", "experiment": cfg.experiment_name}

@app.post("/api/control/pause")
def toggle_pause():
    """Pause or resume training."""
    sim_state.is_paused = not sim_state.is_paused
    return {"status": "success", "is_paused": sim_state.is_paused}

@app.post("/api/control/stop")
def stop_simulation():
    """Stop active simulation."""
    sim_state.is_running = False
    return {"status": "stopped"}

@app.get("/api/copilot/latest")
def get_copilot_latest():
    """Retrieve latest AI Copilot diagnostic."""
    if sim_state.server and sim_state.server.copilot:
        return sim_state.server.copilot.get_latest_insight() or {}
    return {}

@app.get("/api/copilot/history")
def get_copilot_history():
    """Retrieve all chronological AI Copilot insights."""
    if sim_state.server and sim_state.server.copilot:
        return sim_state.server.copilot.insights_history
    return []

@app.get("/api/replay/snapshots")
def get_all_replay_snapshots():
    """Get all saved round snapshots for time travel playback."""
    if sim_state.server and sim_state.server.replay:
        return sim_state.server.replay.get_all_snapshots()
    return []

@app.get("/api/replay/round/{round_num}")
def get_round_replay_snapshot(round_num: int):
    """Retrieve specific round snapshot."""
    if sim_state.server and sim_state.server.replay:
        snap = sim_state.server.replay.get_snapshot(round_num)
        if snap:
            return snap
        raise HTTPException(status_code=404, detail="Round snapshot not found.")
    raise HTTPException(status_code=400, detail="Replay engine not initialized.")

@app.get("/api/xai/features")
def get_xai_feature_attributions():
    """Get global biomarker feature attributions."""
    if sim_state.server and sim_state.server.explainer:
        _, test_data, _, _, _ = load_breast_cancer_data()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
        return sim_state.server.explainer.explain_global_features(test_loader)
    return []

@app.post("/api/xai/diagnose")
def diagnose_single_patient(sample: List[float]):
    """Run clinical explainer inference on a sample patient vector."""
    if sim_state.server and sim_state.server.explainer:
        tensor = torch.FloatTensor(sample)
        return sim_state.server.explainer.explain_single_patient(tensor)
    raise HTTPException(status_code=400, detail="Explainer model not initialized.")

@app.get("/api/report/markdown", response_class=PlainTextResponse)
def get_markdown_research_report():
    """Generate Markdown clinical research report."""
    if sim_state.server:
        hospitals_dict = [h.to_dict() for h in sim_state.server.hospitals]
        priv_summary = sim_state.server.privacy_accountant.get_privacy_spent()
        xai_summary = sim_state.latest_telemetry.get("xai", [])
        return sim_state.server.report_generator.generate_markdown(
            experiment_config=sim_state.config.to_dict(),
            metrics_history=sim_state.server.tracker.rounds_history,
            hospitals_data=hospitals_dict,
            privacy_summary=priv_summary,
            xai_summary=xai_summary
        )
    return "# FedHealth Research Report\nNo simulation has run yet."

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send initial state immediately upon connection
    await websocket.send_json({
        "type": "initial_state",
        "data": {
            "is_running": sim_state.is_running,
            "is_paused": sim_state.is_paused,
            "current_round": sim_state.current_round,
            "config": sim_state.config.to_dict(),
            "latest": sim_state.latest_telemetry
        }
    })
    try:
        while True:
            data = await websocket.receive_text()
            # Handle client heartbeats or inbound messages
    except WebSocketDisconnect:
        manager.disconnect(websocket)
