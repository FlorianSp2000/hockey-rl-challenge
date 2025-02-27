"""
Skript for replaying games from the game_data folder.
"""
import os
import time
from glob import glob

import imageio
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio

from hockey.hockey_env import CENTER_X, CENTER_Y, FPS, HockeyEnv

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


def main(games_path, id, record, render, output_path, verbose):
    env = HockeyEnv()
    _, _ = env.reset()

    # print absolute path of games_path
    games_path = os.path.abspath(games_path)
    print(f"games_path: {games_path}")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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
        match_dir = os.path.join(output_path, match["identifier"])
        if not os.path.exists(match_dir):
            os.makedirs(match_dir)
        
        if "links" in match['identifier']:
            print(f"STARTING FROM LEFT SIDE")
            side = 0
        elif "rechts" in match['identifier']:
            print(f"STARTING FROM RIGHT SIDE")
            side = 1
        # else:
        #     raise ValueError("Could not determine side from match data.")
        num_rounds = match["num_rounds"][0].item()
        print(f"Replaying {num_rounds} rounds from match: {match['identifier']}")
        
        for round_idx in range(num_rounds):
            round_dir = os.path.join(match_dir, str(round_idx))
            if not os.path.exists(round_dir):
                os.makedirs(round_dir)

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

                if record:
                    # Get the frame from env.render(mode='rgb_array')
                    frame = env.render(mode='rgb_array')
                    print(f"frame.shape: {frame.shape}")
                    # Save the frame as a PNG file
                    frame_filename = os.path.join(round_dir, f"step_{step_idx:04d}.png")
                    plt.imsave(frame_filename, frame)

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
                    time.sleep(1 / FPS)
            _, _ = env.reset()


def plot_game_frames(game_id, round_id=0, step_size=5, base_path="game_data/replays", border_size=5, save_path=None):
    round_path = os.path.join(base_path, game_id, str(round_id))
    if not os.path.exists(round_path):
        print(f"Error: Path {round_path} does not exist.")
        return
    
    # Get all step images and sort them
    frame_files = sorted(
        [f for f in os.listdir(round_path) if f.startswith("step_") and f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # Extract step number and sort
    )
    
    # Apply step size
    frame_files = frame_files[::step_size]
    
    if not frame_files:
        print("No frames found.")
        return
    
    # Load images and add borders
    frames = []
    for f in frame_files:
        img = imageio.imread(os.path.join(round_path, f))
        
        # Add a black border around the image
        bordered_img = np.pad(img, pad_width=((border_size, border_size), (border_size, border_size), (0, 0)), 
                              mode='constant', constant_values=0)
        frames.append(bordered_img)

    # Plot frames side by side
    fig, axes = plt.subplots(1, len(frames), figsize=(len(frames) * 1.5, 2), constrained_layout=True)
    
    if len(frames) == 1:
        axes = [axes]  # Ensure iteration works for a single frame
    
    for ax, frame, fname in zip(axes, frames, frame_files):
        ax.imshow(frame)
        ax.set_title(f"Frame {int(fname.split('.')[0].split('_')[1])}", fontsize=8)  # Decrease font size
        ax.axis("off")

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


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