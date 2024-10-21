import numpy as np
import torch
import os
import random
import threading
import queue
import gymnasium as gym
from gymnasium.wrappers import *
import time
import cv2

class PoleEnv:
    def __init__(self):
        self.env = GrayscaleObservation(AddRenderObservation(gym.make("CartPole-v1", render_mode="rgb_array", max_episode_steps=200), render_only=True), keep_dim=False)

        print("Setting seeds.")
        seed = 42
        np.random.seed(seed)

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256, 256), dtype=np.uint8)
        self.action_size = self.action_space.n

        self.data_buffer = queue.Queue(maxsize=1000)
        self.stop_generation = False
        self.pause_generation = threading.Event()
        self.simulation_thread = None

        self.current_obs_added_per_s = 1
        self.all_rewards = []

        self.image_side_length = 128

    def simulation_worker(self):
        while not self.stop_generation:
            self.pause_generation.wait()  # Wait if paused

            start_time = time.time()

            obs, _ = self.env.reset()
            obs = self.resize_observation(obs)
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self.env.action_space.sample()  # Random action
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs = self.resize_observation(next_obs)

                episode_obs.append(obs)
                episode_actions.append(action)
                episode_rewards.append(reward)

                obs = next_obs
                self.all_rewards.append(reward)

            if self.data_buffer.full():
                self.data_buffer.get()
            self.data_buffer.put((episode_obs, episode_actions, episode_rewards))

            end_time = time.time()
            time_taken = end_time - start_time
            self.current_obs_added_per_s = len(episode_obs) / time_taken

    def resize_observation(self, obs):
        return cv2.resize(obs, (self.image_side_length, self.image_side_length), interpolation=cv2.INTER_NEAREST)

    def start_data_generation(self):
        self.stop_generation = False
        self.pause_generation.set()  # Ensure it starts unpaused
        self.simulation_thread = threading.Thread(target=self.simulation_worker)
        self.simulation_thread.start()

    def stop_data_generation(self):
        self.stop_generation = True
        self.pause_generation.set()  # Ensure it's not paused when stopping
        if self.simulation_thread:
            self.simulation_thread.join()

    def sample_buffer(self, batch_size):
        self.pause_generation.clear()  # Pause the simulator
        list_data_buffer = list(self.data_buffer.queue)
        self.pause_generation.set()

        if self.data_buffer.qsize() < 1:
            print("Not enough data!")
            return None

        obs_batch, action_batch, reward_batch = [], [], []

        for _ in range(batch_size):
            episode = random.choice(list_data_buffer)
            episode_obs, episode_actions, episode_rewards = episode

            if len(episode_obs) < 1:
                continue  # Skip episodes that are too short

            idx = random.randint(0, len(episode_obs) - 1)

            obs_batch.append(episode_obs[idx])
            action_batch.append(episode_actions[idx])
            reward_batch.append(episode_rewards[idx])

        return np.array(obs_batch), np.array(action_batch), np.array(reward_batch)

    def get_latest_episode(self):
        return self.get_episode_with_index(self.data_buffer.qsize() - 1)

    def get_episode_with_index(self, index):
        self.pause_generation.clear()  # Pause the simulator

        if index < 0 or index >= self.data_buffer.qsize():
            self.pause_generation.set()  # Resume if index is invalid
            return None
        
        episode = list(self.data_buffer.queue)[index]

        self.pause_generation.set()

        obs_batch, action_batch, reward_batch = episode
        return np.array(obs_batch), np.array(action_batch), np.array(reward_batch)

    def gen_vid_from_obs(self, obs, filename="simulation.mp4", fps=10.0):
        if not os.path.dirname(filename):
            filename = os.path.join(os.getcwd(), filename)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Assuming obs is a list of 2D grayscale arrays (256x256)
        height, width = self.image_side_length, self.image_side_length
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in obs:
            # Convert grayscale to RGB
            rgb_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            out.write(rgb_frame)

        out.release()


import unittest
import numpy as np
import time
import os

class TestPoleEnv(unittest.TestCase):
    def setUp(self):
        self.env = PoleEnv()
        self.env.start_data_generation()

    def tearDown(self):
        self.env.stop_data_generation()
        self.env.close()

    def test_initialization(self):
        self.assertIsNotNone(self.env.action_space)
        self.assertIsNotNone(self.env.observation_space)
        self.assertEqual(self.env.action_size, 2)  # CartPole has 2 actions

    def test_data_generation(self):
        time.sleep(2)  # Wait for some data to be generated
        self.assertGreater(self.env.data_buffer.qsize(), 0)

    def test_sample_buffer(self):
        time.sleep(2)  # Wait for some data to be generated
        batch_size = 5
        sample = self.env.sample_buffer(batch_size)
        self.assertIsNotNone(sample)
        obs, actions, rewards = sample
        self.assertEqual(len(obs), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)

    def test_get_latest_episode(self):
        time.sleep(2)  # Wait for some data to be generated
        episode = self.env.get_latest_episode()
        self.assertIsNotNone(episode)
        obs, actions, rewards = episode
        self.assertGreater(len(obs), 0)
        self.assertEqual(len(obs), len(actions))
        self.assertEqual(len(obs), len(rewards))

    def test_gen_vid_from_obs(self):
        time.sleep(2)  # Wait for some data to be generated
        episode = self.env.get_latest_episode()
        self.assertIsNotNone(episode)
        obs, _, _ = episode
        
        video_filename = "test_episode.mp4"
        self.env.gen_vid_from_obs(obs, filename=video_filename)
        
        self.assertTrue(os.path.exists(video_filename))
        self.assertGreater(os.path.getsize(video_filename), 0)
        
        # Clean up the video file
        os.remove(video_filename)