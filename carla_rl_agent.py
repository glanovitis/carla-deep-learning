#!/usr/bin/env python

import glob
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import cv2
import random
import time
import psutil

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from environment_setup import spawn_traffic
from ego_vehicle_setup import EgoVehicle
from data_collector import calculate_reward

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class CNNFeatureExtractor(nn.Module):
    """
    CNN to extract features from image input
    """

    def __init__(self, in_channels=3):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # For 84x84 input, after the convolutions we have 64 * 7 * 7 = 3136 features
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Print shapes for debugging
        # print(f"Input shape: {x.shape}")

        x = x / 255.0  # Normalize image
        x = self.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")

        x = self.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")

        x = self.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten
        # print(f"After flatten: {x.shape}")

        x = self.relu(self.fc(x))
        return x


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    """

    def __init__(self):
        super(ActorCritic, self).__init__()

        # Feature extractor
        self.cnn = CNNFeatureExtractor(in_channels=3)  # RGB image

        # Actor (Policy) network
        self.actor_mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 actions: throttle, steer, brake
        )

        # Standard deviations for exploration
        self.actor_std = nn.Parameter(torch.ones(3) * 0.1)

        # Critic (Value) network
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        features = self.cnn(state)

        # Actor output
        action_mean = self.actor_mean(features)

        # We'll use tanh to constrain actions between -1 and 1
        action_mean = torch.tanh(action_mean)

        # Create normal distribution with mean from actor network
        # and standard deviation as a learned parameter
        action_std = self.actor_std.expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        # Value estimate
        value = self.critic(features)

        return dist, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.06, gamma=0.99, epsilon=0.4, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = {
            'states': [],
            'actions': [],
            'action_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def act(self, state):
        """
        Select an action based on the current state
        """
        # Convert to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            dist, value = self.policy(state)
            action = dist.sample()
            action_prob = dist.log_prob(action).sum(dim=-1)

        # Convert to numpy array
        action = action.cpu().numpy()[0]
        action_prob = action_prob.cpu().item()
        value = value.cpu().item()

        return action, action_prob, value

    def remember(self, state, action, action_prob, reward, done, value):
        """
        Store experience in memory
        """
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['action_probs'].append(action_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['values'].append(value)

    def update(self):
        """
        Update the policy using the PPO algorithm
        """
        states = torch.FloatTensor(np.array(self.memory['states'])).to(device)
        actions = torch.FloatTensor(np.array(self.memory['actions'])).to(device)
        old_action_probs = torch.FloatTensor(np.array(self.memory['action_probs'])).to(device)
        rewards = torch.FloatTensor(np.array(self.memory['rewards'])).to(device)
        dones = torch.FloatTensor(np.array(self.memory['dones'])).to(device)
        values = torch.FloatTensor(np.array(self.memory['values'])).to(device)

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0

        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            R = reward + self.gamma * R * (1 - done)
            returns.insert(0, R)
            advantages.insert(0, R - value)

        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(5):  # Multiple epochs of mini-batch updates
            # Get current policy distribution
            dist, values = self.policy(states)

            # Get log probabilities of actions
            new_action_probs = dist.log_prob(actions).sum(dim=-1)

            # Calculate ratio
            ratio = torch.exp(new_action_probs - old_action_probs)

            # Clipped objective function
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages

            # Value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)

            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy loss for exploration
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear memory
        self.memory = {
            'states': [],
            'actions': [],
            'action_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def save(self, path):
        """
        Save the model
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """
        Load a saved model
        """
        try:
            # Use weights_only=False to fix PyTorch 2.6 compatibility issue
            self.policy.load_state_dict(torch.load(path, weights_only=False))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using default weights instead")


def attach_spectator_to_vehicle(world, vehicle, view_mode='third_person_rear'):
    """
        Set up the simulator view to follow the vehicle
        """
    try:
        spectator = world.get_spectator()

        # Define the camera offsets for different views
        views = {
            'third_person_rear': {
                'offset': carla.Location(x=-30, z=15),
                'rotation_offset': carla.Rotation(pitch=-15)
            },
            'first_person': {
                'offset': carla.Location(x=0.4, z=1.4),
                'rotation_offset': carla.Rotation(pitch=0)
            },
            'bird_eye': {
                'offset': carla.Location(x=0, z=50.0),
                'rotation_offset': carla.Rotation(pitch=-90)
            }
        }

        # Default to third-person if specified view not found
        if view_mode not in views:
            print(f"Warning: View mode '{view_mode}' not found, using third_person_rear")
            view_mode = 'third_person_rear'

        view = views[view_mode]

        def update_spectator():
            try:
                if vehicle and vehicle.is_alive:
                    # Get vehicle transform
                    vehicle_transform = vehicle.get_transform()

                    if view_mode == 'first_person':
                        # First person view
                        forward_vec = vehicle_transform.get_forward_vector()
                        spectator_location = vehicle_transform.location + \
                                             forward_vec * view['offset'].x + \
                                             vehicle_transform.get_up_vector() * view['offset'].z

                        spectator_rotation = vehicle_transform.rotation
                        spectator_rotation.pitch += view['rotation_offset'].pitch

                    elif view_mode == 'bird_eye':
                        # Bird's eye view
                        spectator_location = vehicle_transform.location + carla.Location(z=view['offset'].z)
                        spectator_rotation = carla.Rotation(
                            pitch=view['rotation_offset'].pitch,
                            yaw=vehicle_transform.rotation.yaw,
                            roll=0
                        )

                    else:
                        # Third-person view
                        forward_vec = vehicle_transform.get_forward_vector()
                        spectator_location = vehicle_transform.location - \
                                             forward_vec * abs(view['offset'].x) + \
                                             carla.Location(z=view['offset'].z)

                        # Look at vehicle
                        direction = vehicle_transform.location - spectator_location
                        spectator_rotation = carla.Rotation(
                            pitch=view['rotation_offset'].pitch,
                            yaw=math.degrees(math.atan2(direction.y, direction.x)),
                            roll=0
                        )

                    # Set spectator transform
                    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            except Exception as e:
                print(f"Error updating spectator: {e}")
                # Don't propagate the exception to prevent training interruption
                # Just log it and continue

        return update_spectator
    except Exception as e:
        print(f"Error setting up vehicle POV: {e}")
        # Return a dummy function that does nothing
        return lambda: None

def preprocess_image(image):
    """
    Preprocess image for CNN input:
    1. Resize to 84x84 (standard size for RL with images)
    2. Convert to PyTorch format (C, H, W)
    """
    if image is None:
        return np.zeros((3, 84, 84), dtype=np.float32)

    # Resize to smaller dimensions for faster processing
    image = cv2.resize(image, (84, 84))

    # Convert to float and normalize
    image = image.astype(np.float32)

    # Transpose from (H, W, C) to (C, H, W) for PyTorch
    image = image.transpose(2, 0, 1)

    return image


def destroy_all_actors(client, world):
    """Very aggressive actor cleanup that tries multiple approaches"""
    print("Performing aggressive actor cleanup...")

    # Save original settings
    original_settings = world.get_settings()

    # Step 1: Switch to synchronous mode to ensure commands complete
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Step 2: Stop all controllers first (they control walkers)
    controllers = world.get_actors().filter('controller.ai.*')
    for controller in controllers:
        try:
            if controller.is_alive:
                controller.stop()
        except:
            pass

    # Tick to process stop commands
    world.tick()
    time.sleep(0.5)

    # Step 3: Group actors by type for ordered destruction
    vehicles = world.get_actors().filter('vehicle.*')
    walkers = world.get_actors().filter('walker.*')
    sensors = world.get_actors().filter('sensor.*')

    # Count actors
    print(f"Found {len(vehicles)} vehicles, {len(walkers)} walkers, " 
          f"{len(controllers)} controllers, {len(sensors)} sensors")

    # Step 4: First attempt - destroy sensors first (they're attached to vehicles)
    destroy_count = 0

    for sensor in sensors:
        try:
            if sensor.is_alive:
                sensor.destroy()
                destroy_count += 1
        except:
            pass

    # Tick to process destroy commands
    world.tick()

    # Step 5: Use synchronous batch commands for remaining actors
    # This ensures CARLA processes each batch before moving to the next
    for actor_list in [controllers, walkers, vehicles]:
        if actor_list:
            # Try batch destroy with verification
            try:
                print(f"Destroying {len(actor_list)} actors...")
                batch = [carla.command.DestroyActor(x) for x in actor_list]
                responses = client.apply_batch_sync(batch, True)

                # Count successes/failures
                success = sum(1 for r in responses if not r.has_error())
                failure = len(responses) - success
                print(f"Batch destroy: {success} succeeded, {failure} failed")
                destroy_count += success
            except Exception as e:
                print(f"Batch destroy failed: {e}")

                # Fallback to individual destruction with ticks
                for actor in actor_list:
                    try:
                        if actor.is_alive:
                            actor.destroy()
                            destroy_count += 1
                            # Tick after each destroy to ensure it's processed
                            world.tick()
                    except:
                        pass

        # Tick after each batch
        world.tick()

    # Step 6: Final verification and cleanup
    time.sleep(1.0)

    # Count remaining actors
    remaining_actors = len(world.get_actors().filter('vehicle.*')) + \
                      len(world.get_actors().filter('walker.*')) + \
                      len(world.get_actors().filter('controller.ai.*')) + \
                      len(world.get_actors().filter('sensor.*'))

    print(f"Cleanup results: {destroy_count} actors destroyed, {remaining_actors} actors remain")

    # Restore original settings
    world.apply_settings(original_settings)

    return remaining_actors == 0


def train_agent(episodes=100, steps_per_episode=2000, update_frequency=200, start_episode=0):
    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world
        world = client.get_world()

        # Set synchronous mode
        settings = world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        checkpoint_interval = 2
        checkpoint_path = "carla_checkpoint.pth"

        # Create an agent
        agent = PPOAgent(state_dim=(3, 84, 84), action_dim=3)  # RGB image input

        # Load existing model if continuing training
        if start_episode > 0 and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint...")
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                agent.policy.load_state_dict(checkpoint['model_state_dict'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                best_reward = checkpoint.get('best_reward', -float('inf'))
                print(f"Checkpoint loaded, best reward: {best_reward}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                best_reward = -float('inf')
        else:
            best_reward = -float('inf')

        # Training variables
        total_steps = 0

        for episode in range(episodes):
            # Calculate absolute episode number for logging
            episode_number = start_episode + episode

            try:
                print(f"Starting episode {episode_number + 1}/{start_episode + episodes}")

                # Spawn traffic
                vehicles, walkers, controllers = spawn_traffic(client, 20, 10)

                # Choose a random spawn point
                spawn_points = world.get_map().get_spawn_points()

                # Initialize variables before creating the ego_vehicle
                episode_reward = 0
                done = False
                step = 0

                # Initialize reward tracking variables
                prev_location = None
                prev_speed = None
                distance_traveled = 0
                prev_steering = None

                # Initialize collision tracking - MOVE THIS HERE!
                last_collision_step = None

                # Creating the ego_vehicle
                ego_transform = random.choice(spawn_points) if spawn_points else carla.Transform()
                ego_vehicle = EgoVehicle(world)
                update_spectator = attach_spectator_to_vehicle(world, ego_vehicle.vehicle, 'first_person')

                prev_location = ego_vehicle.vehicle.get_transform().location

                # Main training loop
                while not done and step < steps_per_episode:
                    # Process world tick
                    # REMOVE THIS LINE: last_collision_step = None
                    world.tick()

                    # Update spectator position to follow the vehicle
                    update_spectator()

                    # Get current state (RGB image)
                    rgb_image = ego_vehicle.sensor_data['rgb_front']
                    state = preprocess_image(rgb_image)

                    # Select action
                    action, action_prob, value = agent.act(state)

                    if total_steps < 5000 and random.random() < 0.3:  # 30% chance in first 5000 steps
                        # Force high throttle, minimal brake, small random steering
                        action[0] = random.uniform(0.7, 1.0)  # High throttle
                        action[2] = random.uniform(0.0, 0.1)  # Minimal brake

                    # Convert action to CARLA control
                    control = carla.VehicleControl()
                    # Explicitly convert numpy.float32 to Python float
                    control.throttle = float(max(0, min(1.0, action[0])))  # Clamp between 0 and 1
                    control.steer = float(max(-1.0, min(1.0, action[1])))  # Clamp between -1 and 1
                    control.brake = float(max(0, min(1.0, action[2])))  # Clamp between 0 and 1

                    # Apply control to vehicle
                    ego_vehicle.apply_control(control)

                    reward, current_location, current_speed, distance_traveled, current_steering, last_collision_step, episode_done = calculate_reward(
                        ego_vehicle=ego_vehicle,
                        prev_location=prev_location,
                        prev_speed=prev_speed,
                        distance_traveled=distance_traveled,
                        total_steps=total_steps,
                        prev_steering=prev_steering,
                        target_speed=30,
                        last_collision_step=last_collision_step  # Pass and update this
                    )

                    # Only end episode for valid termination conditions
                    if episode_done:
                        done = True
                        print(f"Episode ending at step {step} - termination condition met")

                    # Update tracking variables for next iteration
                    prev_location = current_location
                    prev_speed = current_speed
                    prev_steering = current_steering

                    # Store experience
                    agent.remember(state, action, action_prob, reward, done, value)

                    # Update total rewards
                    episode_reward += reward

                    # Update policy if enough steps have been taken
                    total_steps += 1
                    if total_steps % update_frequency == 0:
                        print("Updating policy...")
                        agent.update()

                    step += 1

                    # Add debugging info - print if reward is significant
                    if reward > 1.0 or reward < -1.0:
                        print(
                            f"Step {step}, Reward: {reward:.2f}, Speed: {current_speed:.2f} km/h, Distance: {distance_traveled:.2f}m")

                # Episode summary with distance traveled
                print(
                    f"Episode {episode_number + 1} completed with reward: {episode_reward:.2f}, Distance: {distance_traveled:.2f}m")

                # Update policy at the end of each episode if there's data
                if len(agent.memory['states']) > 0:
                    agent.update()

                # Save model if it's the best so far
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    agent.save("carla_ppo_best.pth")
                    print(f"New best model saved with reward: {best_reward}")

                # Save checkpoint every 10 episodes
                if episode % 10 == 0:
                    agent.save(f"carla_ppo_checkpoint_ep{episode}.pth")

                # Save training checkpoint with more details
                if episode % checkpoint_interval == 0:
                    torch.save({
                        'episode': episode,
                        'model_state_dict': agent.policy.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'best_reward': best_reward,
                    }, checkpoint_path)
                    print(f"Checkpoint saved at episode {episode}")

                    process = psutil.Process(os.getpid())
                    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

            except Exception as e:
                print(f"Error in episode {episode + 1}: {e}")
            finally:
                # Cleanup ego vehicle first
                if 'ego_vehicle' in locals() and ego_vehicle:
                    try:
                        ego_vehicle.destroy()
                        print("Ego vehicle destroyed")
                    except Exception as e:
                        print(f"Error destroying ego vehicle: {e}")

                # Now clean up all other actors
                try:
                    destroy_all_actors(client, world)
                    print("Actors destroyed")
                except Exception as e:
                    print(f"Error during actor cleanup: {e}")

                # Restore original settings
                if 'original_settings' in locals() and original_settings:
                    try:
                        world.apply_settings(original_settings)
                        print("Original settings restored")
                    except Exception as e:
                        print(f"Error restoring original settings: {e}")

        # Save final model
        agent.save("carla_ppo_final.pth")
        print("Training completed!")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Clean up
        if 'ego_vehicle' in locals():
            try:
                ego_vehicle.destroy()
            except:
                pass

        # Clean up traffic if exists
        if 'controllers' in locals() and 'walkers' in locals() and 'vehicles' in locals():
            try:
                client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
                client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
            except:
                pass

        # Restore original settings
        if 'original_settings' in locals():
            world.apply_settings(original_settings)

def test_agent(model_path, num_episodes=5, steps_per_episode=1000):
    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world
        world = client.get_world()

        # Set synchronous mode
        settings = world.get_settings()
        original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Create an agent
        agent = PPOAgent(state_dim=(3, 150, 200), action_dim=3)
        agent.load(model_path)
        print(f"Model loaded from {model_path}")

        for episode in range(num_episodes):
            print(f"Starting test episode {episode + 1}/{num_episodes}")

            # Spawn traffic
            vehicles, walkers, controllers = spawn_traffic(client, 20, 10)

            # Choose a random spawn point
            spawn_points = world.get_map().get_spawn_points()
            ego_transform = random.choice(spawn_points) if spawn_points else carla.Transform()

            # Create the ego vehicle
            ego_vehicle = EgoVehicle(world, ego_transform)

            # Episode variables
            episode_reward = 0
            done = False
            step = 0
            prev_location = ego_transform.location

            while not done and step < steps_per_episode:
                # Process world tick
                world.tick()

                # Get current state (RGB image)
                rgb_image = ego_vehicle.sensor_data['rgb_front']
                state = preprocess_image(rgb_image)

                # Select action
                action, _, _ = agent.act(state)

                # Convert action to CARLA control
                control = carla.VehicleControl()
                control.throttle = max(0, min(1.0, action[0]))
                control.steer = max(-1.0, min(1.0, action[1]))
                control.brake = max(0, min(1.0, action[2]))

                # Apply control to vehicle
                ego_vehicle.apply_control(control)

                # Calculate reward
                reward, current_location = calculate_reward(ego_vehicle, prev_location)
                prev_location = current_location

                # Display state and action
                cv2.imshow('RGB Camera', cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

                # Update total rewards
                episode_reward += reward

                # Check for termination conditions
                if ego_vehicle.sensor_data['collision']:
                    print("Collision detected! Ending episode.")
                    done = True

                step += 1

            print(f"Test episode {episode + 1} completed with reward: {episode_reward}")

            # Clean up ego vehicle
            ego_vehicle.destroy()

            # Clean up traffic
            client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
            client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])

    except KeyboardInterrupt:
        print("Testing interrupted")
    finally:
        # Clean up
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()

        # Clean up traffic if exists
        if 'controllers' in locals() and 'walkers' in locals() and 'vehicles' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
            client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])

        # Restore original settings
        if 'original_settings' in locals():
            world.apply_settings(original_settings)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CARLA Reinforcement Learning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--model', type=str, default='carla_ppo_best.pth',
                        help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for training')

    args = parser.parse_args()

    if args.mode == 'train':
        train_agent(episodes=args.episodes)
    else:
        test_agent(model_path=args.model)