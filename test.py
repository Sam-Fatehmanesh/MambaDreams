import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from MambaDreams.models.world import WorldModel
from MambaDreams.testenv.poleenv import PoleEnv
from MambaDreams.custom_functions.laprop import LaProp
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Check if CUDA is available
device = "cuda:0"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize environment and world model
env = PoleEnv()
env.start_data_generation()
print("Warming up datagen...")
time.sleep(4)
print("Done.")

world_model = WorldModel(
    action_dims=(env.action_size,),
    image_side_size=env.image_side_length,
    image_latent_category_size=16
).to(device)

# Training parameters
num_epochs = 2000
batch_size = 16
seq_length = 20
learning_rate = 1e-3

optimizer = optim.Adam(world_model.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()
# bce_loss = nn.BCELoss()

# Lists to store loss values for plotting
kl_div_losses = []
obs_losses = []
latent_losses = []
reward_losses = []

# Training loop
for epoch in range(num_epochs):
    # Sample data from the environment
    obs_batch, action_batch, reward_batch = env.sample_buffer(batch_size, seq_length)
    
    if obs_batch.size == 0:
        print("Not enough data. Waiting...")
        continue

    # Convert to tensors and move to device
    obs_tensor = torch.FloatTensor(obs_batch).to(device) / 255.0
    action_tensor = torch.LongTensor(action_batch).to(device)
    reward_tensor = torch.FloatTensor(reward_batch).unsqueeze(-1).to(device)

    # Encode observations
    decoded_obs, obs_lats, obs_lat_dists = world_model.encode_obs(obs_tensor)

    # Forward pass
    pred_obs, pred_lat_dists, pred_rewards, _ = world_model(obs_lats[:, :-1].detach(), action_tensor[:, :-1], reward_tensor[:, :-1])

    # Calculate losses
    # Reconstruction loss
    obs_loss = F.binary_cross_entropy(decoded_obs, obs_tensor, reduction='sum') 

    # KL loss
    obs_lat_dists = obs_lat_dists[:, 1:].reshape(-1, world_model.image_latent_category_size) 
    pred_lat_dists = pred_lat_dists.reshape(-1, world_model.image_latent_category_size)

    kl_divergence =  torch.sum(obs_lat_dists * torch.log(obs_lat_dists  / pred_lat_dists)) #+ torch.sum(pred_lat_dists.detach() * torch.log( pred_lat_dists.detach() / obs_lat_dists))

    vae_loss = obs_loss + kl_divergence

    reward_loss = mse_loss(pred_rewards, reward_tensor[:, 1:])

    latent_loss = 10*ce_loss(pred_lat_dists, obs_lat_dists.detach())

    total_loss = vae_loss + reward_loss + latent_loss

    # Backward pass and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Store loss values for plotting
    kl_div_losses.append(kl_divergence.item())
    obs_losses.append(obs_loss.item())
    latent_losses.append(latent_loss.item())
    reward_losses.append(reward_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, Env steps: {env.total_steps}")

# Generate videos
def generate_video(obs_sequence, filename):
    env.gen_vid_from_obs(obs_sequence, filename=filename)

# Get a sample episode
obs, actions, rewards = env.get_latest_episode()

# Convert to tensors and move to device
obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
action_tensor = torch.LongTensor(actions).unsqueeze(0).to(device)
reward_tensor = torch.FloatTensor(rewards).unsqueeze(0).unsqueeze(-1).to(device)

# Generate predictions
with torch.no_grad():
    _, obs_lats, _ = world_model.encode_obs(obs_tensor)
    pred_obs, _, _, _ = world_model(obs_lats, action_tensor, reward_tensor)

# Convert predictions back to numpy arrays
pred_obs_np = (pred_obs.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

# Generate videos
os.makedirs("output_videos", exist_ok=True)
generate_video(obs, "output_videos/true_observations.mp4")
generate_video(pred_obs_np, "output_videos/predicted_observations.mp4")

print("Videos generated in the 'output_videos' directory.")

# Stop data generation
env.stop_data_generation()

# Plot and save loss graphs
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(kl_div_losses)
plt.title('KL Divergence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(obs_losses)
plt.title('Observation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 3)
plt.plot(latent_losses)
plt.title('Latent Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(reward_losses)
plt.title('Reward Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('loss_graphs.png')
plt.close()

print("Loss graphs saved as 'loss_graphs.png'.")

torch.save(world_model.state_dict(), 'model_state_dict.pth')
