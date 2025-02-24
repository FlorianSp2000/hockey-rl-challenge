"""
Skript for replaying games from the game_data folder.
"""
import os
import time
from glob import glob

import imageio
import numpy as np
import pickle


from hockey.hockey_env import CENTER_X, CENTER_Y, FPS, HockeyEnv, ContactDetector


def set_env_state_from_observation(env, observation):
    env.player1.position = (observation[[0, 1]] + [CENTER_X, CENTER_Y]).tolist()
    env.player1.angle = observation[2]
    env.player1.linearVelocity = [observation[3], observation[4]]
    env.player1.angularVelocity = observation[5]
    env.player2.position = (observation[[6, 7]] + [CENTER_X, CENTER_Y]).tolist()
    env.player2.angle = observation[8]
    env.player2.linearVelocity = [observation[9], observation[10]]
    env.player2.angularVelocity = observation[11]
    env.puck.position = (observation[[12, 13]] + [CENTER_X, CENTER_Y]).tolist()
    env.puck.linearVelocity = [observation[14], observation[15]]


def setup_video(output_path, id, fps):
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{id}.mp4")
    print("Record video in {}".format(file_path))
    # noinspection SpellCheckingInspection
    return (
        imageio.get_writer(
            file_path, fps=fps, codec="mjpeg", quality=10, pixelformat="yuvj444p"
        ),
        file_path,
    )


def main(games_path, id, record, render, output_path, verbose):
    env = HockeyEnv()
    _, _ = env.reset()
    if env.world.contactListener is not None:
        print("Contact listener is active")
    if isinstance(env.world.contactListener, ContactDetector):
        print("ContactDetector is active")

    # print absolute path of games_path
    games_path = os.path.abspath(games_path)
    print(f"games_path: {games_path}")
    
    matches = []
    for match_path in glob(os.path.join(games_path, "**", "*.pkl"), recursive=True):
        with open(match_path, "rb") as f:
            match_data = pickle.load(f)

        # Extract identifier from filename (everything before ".pkl")
        identifier = os.path.splitext(os.path.basename(match_path))[0]
        match_data["identifier"] = identifier  # Add identifier to the dictionary

        matches.append(match_data)

    if id is not None:
        matches = [match for match in matches if match["identifier"] == id]

    for match in matches:
        if "links" in match['identifier']:
            print(f"STARTING FROM LEFT SIDE")
            side = 0
        elif "rechts" in match['identifier']:
            print(f"STARTING FROM RIGHT SIDE")
            side = 1
        else:
            raise ValueError("Could not determine side from match data.")
        num_rounds = match["num_rounds"][0].item()
        print(f"Replaying {num_rounds} rounds from match: {match['identifier']}")
        
        for round_idx in range(num_rounds):
            obs_key = f"observations_round_{round_idx}"
            act_key = f"actions_round_{round_idx}"
    
            observations = match[obs_key]
            actions = match[act_key]
    
            print(f"Round {round_idx}: Observations shape: {len(observations)}, Actions shape: {len(actions)}")

            for step_idx in range(len(observations)):
                if step_idx >= len(actions):
                    print(f"Warning: Skipping action step {step_idx} in round {round_idx} due to missing action.")
                    # break
                else:
                    action = np.array(actions[step_idx])
                observation = np.array(observations[step_idx])
                # observation[12:14] = np.array([4, 0])  # Set puck position to 0, 0
                if step_idx == 0:
                    print("RESET STATE")
                set_env_state_from_observation(env, observation)
                
                done = False
                info = {}
                obs, reward, done, truncated, info = env.step(action)

                if done:
                    print(f"reward: {reward}")
                    # print(f"len(observations))={len(observations)}")
                    print(f"Game ended with winner: {info['winner']}")
                    time.sleep(2)
                    if len(observations) - step_idx > 5:
                        print(f"We are at STEP: {step_idx}, {step_idx - len(observations)} steps before the game would end!!!")
                    break
    
                if render:
                    env.render()
                    time.sleep(2 / FPS)
            _, _ = env.reset()


if __name__ == "__main__":
    games_path = 'game_data'
    id = None
    record=False
    render = True
    output_path = 'game_data/replays'
    verbose= True
    main(
        games_path,
        id,
        record,
        render,
        output_path,
        verbose,
    )