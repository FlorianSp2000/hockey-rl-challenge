# src/utils/model_pool.py
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class ModelRecord:
    path: str
    win_rates: Dict[str, float]  # opponent_id -> win_rate
    timesteps: int
    relative_strength: float = 0.0 # elo-like value for rating models

@dataclass
class OpponentLog:
    timestep: int
    opponent_id: str
    win_rate: float # from current model perspective
    relative_strength: float # from current model perspective
    relative_strength_opponent: float # from opponent perspective

class ModelPool:
    def __init__(self, pool_size: int = 5, save_dir: str = None, 
                 start_relative_strength: float = 1000.0,
                 basic_opponents: Dict[str, str] = {}, 
                 curriculum_thresholds: Dict[str, float] = {},
                 game_decision_margins: Dict[str, float] = {},
                 k_factors: Dict[str, int] = {},
                 ):
        self.pool_size = pool_size
        self.save_dir = save_dir
        self.models: Dict[str, ModelRecord] = {}
        self.current_relative_strength = start_relative_strength # default value for elo-like rating
        self.curriculum_phase = "assessment"
        self.CURRICULUM_THRESHOLDS = curriculum_thresholds
        self.GAME_DECISION_MARGINS = game_decision_margins
        self.K_FACTOR_PER_PHASE = k_factors
        print(f"CURRICULUM_THRESHOLDS: {self.CURRICULUM_THRESHOLDS}")
        print(f"GAME_DECISION_MARGINS: {self.GAME_DECISION_MARGINS}")
        print(f"K_FACTOR_PER_PHASE: {self.K_FACTOR_PER_PHASE}")

        for id, path in basic_opponents.items():
            self.models[id] = ModelRecord(
                path=path,
                win_rates={},
                relative_strength=1350.0, # default value for external opponents
                timesteps=0
            )
        # Initialize basic opponents with empty ModelRecord objects
        self.models["basic_weak"] = ModelRecord(
            path="",  # Empty path since basic opponents don't need model files
            win_rates={},
            relative_strength=1000.0,
            timesteps=0
        )
        self.models["basic_strong"] = ModelRecord(
            path="",
            win_rates={},
            relative_strength=1200.0,  # Set to 1.2k to represent stronger opponent
            timesteps=0
        )

        
    def add_model(self, model_path: str, timesteps: int, win_rates: Dict[str, float]):
        """Add a new model to the pool"""
        model_id = f"model_{timesteps}"
        print(f"win_rates.values() in add_model: {win_rates.values()}")
        current_relative_strength = self.current_relative_strength
        print(f"SELECT OPP: current_relative_strength: {current_relative_strength}")
        # If pool is full, remove worst performing model (excluding basic opponents)
        if len(self.models) >= self.pool_size + 2: # + 2 for basic opponents
            worst_model = min(
                [(mid, m) for mid, m in self.models.items() if not mid.startswith("basic_")],
                key=lambda x: x[1].relative_strength
            )
            print(f"worst_model: {worst_model}")
            print(f"worst_model[1].relative_strength {worst_model[1].relative_strength}")
            print(f"current_relative_strength {current_relative_strength}")
            if worst_model[1].relative_strength < current_relative_strength:
                print(f"number of models before deletion: {len(self.models)}")
                # if os.path.exists(worst_model[1].path): CAREFUL THIS DELETES PREVIOUS CHEKPOINTS!
                #     os.remove(worst_model[1].path)
                del self.models[worst_model[0]]
                print(f"number of models before deletion: {len(self.models)}")
            else:
                print("Current Model is worse than worst model in pool")
                return False
        
        self.models[model_id] = ModelRecord(
            path=model_path,
            win_rates=win_rates,
            relative_strength=current_relative_strength,
            timesteps=timesteps
        )
        print(f"Added model {model_id} to pool")
        self._save_pool_state()
        return True

    def select_opponent(self) -> str:
        """Select opponent based on current performance"""
        if len(self.models) <= 2: # number of basic opponents
            self.curriculum_phase = "assessment"
            # During warmup/assessment, alternate between weak and strong basic opponents
            return np.random.choice(["basic_weak", "basic_strong"])
        print(f"SELECT OPP: self.current_relative_strength: {self.current_relative_strength}")
        # Calculate selection probabilities based on win rates
        candidates = list(self.models.keys())
        strengths = np.array([self.models[mid].relative_strength for mid in candidates])
        
        if self.current_relative_strength < self.CURRICULUM_THRESHOLDS['assessment']:
            # draw from basic opponents with higher probability
            probs = np.array([1.0 if mid.startswith("basic_") else 0.2 for mid in candidates]) 
        elif self.current_relative_strength < self.CURRICULUM_THRESHOLDS['learning']:
            # Learning phase: prefer slightly stronger opponents
            target_strength = self.current_relative_strength + 100
            probs = 1 / (abs(strengths - target_strength) + 50)
            self.curriculum_phase = "learning"
        else:
            self.curriculum_phase = "competitive"
            # Competitive phase: prefer strong opponents
            probs = np.array([
                1.0 if strength >= self.current_relative_strength - 100 else 0.2 
                for strength in strengths
            ])

        print(f"self.curriculum_phase: {self.curriculum_phase}")
        probs = probs / probs.sum()
        print(f"select_opponent method candidates: {candidates}")
        print(f"select_opponent method strengths: {strengths}")
        print(f"select_opponent method probs: {probs}")
        return str(np.random.choice(candidates, p=probs))

    def _update_relative_strengths(self, win_rates: Dict[str, float]):
        """Update relative strength ratings based on win rates against known opponents"""
        K = self.K_FACTOR_PER_PHASE[self.curriculum_phase]
        
        new_relative_strength = self.current_relative_strength
        # Iterate multiple times to stabilize ratings
        # for _ in range(3):
        # TODO: order independence
        for opponent_id, win_rate in win_rates.items():
            if opponent_id in self.models:
                # each win rate corresponds to multiple games
                outcome = None
                if win_rate < self.GAME_DECISION_MARGINS['loss']:
                    outcome = 0
                elif win_rate < self.GAME_DECISION_MARGINS['draw']:
                    outcome = 0.5
                else:
                    outcome = 1
                
                opponent = self.models[opponent_id]
                expected_score = 1 / (1 + 10**((opponent.relative_strength - new_relative_strength) / 400)) # typical elo formula, E_a
                actual_score = outcome # S_a
                
                # Update both model and opponent ratings
                rating_change = K * (actual_score - expected_score)
                print(f"rating_change for current model against {opponent_id}: {rating_change}")
                new_relative_strength += rating_change
                opponent.relative_strength -= rating_change
            else:
                raise ValueError(f"Unknown opponent ID in win_rates during elo-update: {opponent_id}")
        
        self.current_relative_strength = new_relative_strength
        return self.current_relative_strength

    
    def get_model_path(self, model_id: str) -> str:
        model_path = self.models[model_id].path if model_id in self.models else None
        
        if model_path is None:
            raise ValueError(f"Model {model_id} not found in pool")
        return model_path

    def _save_pool_state(self):
        """Save pool state to disk"""
        if self.save_dir:
            state = {
                model_id: {
                    "path": record.path,
                    "win_rates": record.win_rates,
                    "relative_strength": record.relative_strength,
                    "timesteps": record.timesteps
                }
                for model_id, record in self.models.items()
            }
            with open(os.path.join(self.save_dir, "model_pool.json"), "w") as f:
                json.dump(state, f, indent=2)

    def load_pool_state(self):
        """Load pool state from disk"""
        if self.save_dir:
            state_path = os.path.join(self.save_dir, "model_pool.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                self.models = {
                    model_id: ModelRecord(**record)
                    for model_id, record in state.items()
                }
                # Ensure basic opponents are always present
                if "basic_weak" not in self.models:
                    self.models["basic_weak"] = ModelRecord("", {}, 0.0, 0)
                if "basic_strong" not in self.models:
                    self.models["basic_strong"] = ModelRecord("", {}, 1.0, 0)
