from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
load_dotenv()

import hockey.hockey_env as h_env
import numpy as np

from comprl.client import Agent, launch_client
from src.custom_sb3.SAC_ERE import SACERE
from pathlib import Path

from client.replay_game import set_env_state_from_observation
from hockey.hockey_env import HockeyEnv

class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # print(f"Observation: {observation}")
        action = self.hockey_agent.act(observation).tolist()
        # print(f"Action: {action}")
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class MinimalSACAgent(HockeyAgent):
    def __init__(self, weak: bool) -> None:
        super().__init__(weak)
        self.model = SACERE.load(Path('').resolve() / Path('models\\run_2025-02-22_18-01-57\\model_6700134'))
        print("Model loaded")

    def get_step(self, observation: list[float]) -> list[float]:
        observation = np.array(observation, dtype=np.float32)
        
        action, _ = self.model.predict(observation, deterministic=True)
        action = action[:4].tolist()


        return action


class SACAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.env = HockeyEnv() # Environment to simulate environment state and get reward
        
        self.model = SACERE.load(Path('').resolve() / Path('models\\run_2025-02-22_18-01-57\\model_3100062'))
        print("Model loaded")
        # Initialize replay buffer if it doesn't exist
        if not hasattr(self.model, "replay_buffer") or self.model.replay_buffer is None:
            raise ValueError("Replay buffer is not initialized in the model.")
        print("Replay buffer initialized")
        
        self.current_obs = None
        self.current_action = None

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

    def get_step(self, observation: list[float]) -> list[float]:
        observation = np.array(observation, dtype=np.float32)
        
        set_env_state_from_observation(self.env, observation)
        info = self.env._get_info()
        reconstructed_reward = self.env.get_reward(info)

        if self.current_obs is not None:
            # The previous step led to this observation, so we store the transition
            reward = reconstructed_reward
            done = False  # Update this based on game termination logic
            info = {}  # TODO: when on_end_game is called with result=True and stats[0] > stats[1] then -> info={"winner": 1}  
            # if stats[0]== stats[1] info={"winner": 0} else if stats[0] < stats[1] info={"winner": -1}
            # and set done True

            # Store transition in replay buffer
            self.model.replay_buffer.add(
                obs=self.current_obs,
                next_obs=observation,
                action=np.array(self.current_action, dtype=np.float32),
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=np.float32),
                infos=[info],
            )

        action, _ = self.model.predict(observation, deterministic=True)
        action = action[:4].tolist()

        # Store current observation and action for the next step
        self.current_obs = observation
        self.current_action = np.array(action, dtype=np.float32)

        return action
    

# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "sac", "sac-record"],
        default="sac",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "sac":
        agent = MinimalSACAgent(weak=False)
    elif args.agent == "sac-record":
        agent = SACAgent(weak=False)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()