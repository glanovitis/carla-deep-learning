#!/usr/bin/env python

import glob
import os
import sys
import time
import numpy as np
import torch
import random
import subprocess
import signal
import math
from pathlib import Path
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla_rl_agent import PPOAgent, preprocess_image
from environment_setup import spawn_traffic
from ego_vehicle_setup import EgoVehicle
from data_collector import calculate_reward


class CarlaServerManager:
    """Manages the CARLA server process with auto-restart capability"""

    def __init__(self, carla_path=None, port=2000, low_quality=True, carla_server=True):
        # Try to find CARLA path if not provided
        if carla_path is None:
            # Default Windows path
            if os.name == 'nt':
                possible_paths = [
                    r"C:\Program Files\Epic Games\UE_4.26\Engine\Binaries\Win64\CarlaUE4.exe",
                    r"C:\Program Files\Epic Games\CARLA\CarlaUE4.exe",
                    r"C:\CARLA\CarlaUE4.exe",
                    r"E:\AIstuff\DataScience\Main\ClaraDeep\WindowsNoEditor\CarlaUE4.exe",
                    # Check current working directory
                    os.path.join(os.getcwd(), "CarlaUE4.exe")
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        carla_path = path
                        break

            # Default Linux path
            else:
                possible_paths = [
                    "/opt/carla-simulator/CarlaUE4.sh",
                    os.path.join(os.path.expanduser("~"), "carla/CarlaUE4.sh"),
                    os.path.join(os.getcwd(), "CarlaUE4.sh")
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        carla_path = path
                        break

        if carla_path is None:
            print("Warning: CARLA executable not found. Please start CARLA manually.")

        self.carla_path = carla_path
        self.port = port
        self.low_quality = low_quality
        self.carla_server = carla_server
        self.process = None

    def start_server(self):
        """Start the CARLA server"""
        if self.carla_path is None:
            print("No CARLA path specified. Please start CARLA manually.")
            return False

        if self.is_server_running():
            print("CARLA is already running")
            return True

        try:
            # Build command
            cmd = [self.carla_path]

            # Add quality settings
            if self.low_quality:
                cmd.append("-quality-level=Low")

            # Start as server
            if self.carla_server:
                cmd.append("-carla-server")

            # Add port
            cmd.append(f"-carla-port={self.port}")

            # Add no rendering if needed
            # cmd.append("-RenderOffScreen")  # Uncomment for headless mode

            # Start process
            print(f"Starting CARLA server: {' '.join(cmd)}")

            if os.name == 'nt':  # Windows
                self.process = subprocess.Popen(
                    cmd,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:  # Linux/Mac
                self.process = subprocess.Popen(
                    cmd,
                    preexec_fn=os.setsid
                )

            # Wait for server to start
            time.sleep(10)  # Adjust as needed

            return self.is_server_running()

        except Exception as e:
            print(f"Error starting CARLA server: {e}")
            return False

    def stop_server(self):
        """Stop the CARLA server"""
        if self.process is not None:
            try:
                print("Stopping CARLA server...")

                if os.name == 'nt':  # Windows
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:  # Linux/Mac
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

                # Wait for process to terminate
                for _ in range(10):  # Wait up to 10 seconds
                    if not self.is_server_running():
                        break
                    time.sleep(1)

                # Force kill if still running
                if self.is_server_running():
                    if os.name == 'nt':  # Windows
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                    else:  # Linux/Mac
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

                self.process = None

                # Additional checks to ensure no carla processes are left
                self.kill_all_carla_processes()

                # Wait after stopping
                time.sleep(5)  # Allow time for full cleanup

                return True

            except Exception as e:
                print(f"Error stopping CARLA server: {e}")
                return False
        return True

    def restart_server(self):
        """Restart the CARLA server"""
        print("Restarting CARLA server...")
        self.stop_server()
        time.sleep(5)  # Wait between stop and start
        return self.start_server()

    def is_server_running(self):
        """Check if the CARLA server is running"""
        if self.process is None:
            return False

        return self.process.poll() is None

    def kill_all_carla_processes(self):
        """Kill all CARLA processes"""
        try:
            if os.name == 'nt':  # Windows
                subprocess.call('taskkill /f /im CarlaUE4.exe', shell=True)
            else:  # Linux/Mac
                os.system('pkill -f CarlaUE4')
        except Exception as e:
            print(f"Error killing CARLA processes: {e}")


def robust_training(episodes=100, episodes_per_restart=10, max_server_restarts=10,
                    steps_per_episode=1000, update_frequency=2000):
    """
    Training function with automatic CARLA server restart capability
    """
    # Initialize the CARLA server manager
    server_manager = CarlaServerManager()

    # For manual CARLA start (comment out if using auto-start)
    # server_manager.carla_path = None

    # Create agent
    agent = PPOAgent(state_dim=(3, 84, 84), action_dim=3)

    # Training variables
    total_steps = 0
    best_reward = -float('inf')
    current_episode = 0
    server_restarts = 0

    # Check for existing checkpoint
    checkpoint_path = Path('carla_checkpoint.pth')
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            current_episode = checkpoint['episode']
            best_reward = checkpoint['best_reward']
            print(f"Resuming training from episode {current_episode} with best reward {best_reward}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    try:
        # Main training loop with server restart capability
        while current_episode < episodes and server_restarts < max_server_restarts:
            # Start the CARLA server if needed
            if not server_manager.is_server_running():
                if not server_manager.start_server():
                    print("Failed to start CARLA server. Exiting.")
                    break
                # Wait for server to initialize properly
                time.sleep(10)
                server_restarts += 1
                print(f"CARLA server started (restart {server_restarts}/{max_server_restarts})")

            # Connect to the CARLA server
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(20.0)  # Increased timeout

                # Get the world
                world = client.get_world()

                # Load a specific map if desired
                # world = client.load_world('Town02')  # Use smaller town for better performance

                # Set synchronous mode
                settings = world.get_settings()
                original_settings = settings
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)

                # Episode loop - run a batch of episodes before restarting server
                episodes_in_batch = min(episodes_per_restart, episodes - current_episode)
                for batch_episode in range(episodes_in_batch):
                    if not server_manager.is_server_running():
                        print("CARLA server crashed. Restarting...")
                        break

                    current_episode_number = current_episode + batch_episode
                    print(f"Starting episode {current_episode_number + 1}/{episodes}")

                    try:
                        # Use fewer vehicles to reduce load
                        vehicles, walkers, controllers = spawn_traffic(client, 10, 5)  # Reduced traffic

                        # Get spawn points safely
                        spawn_points = world.get_map().get_spawn_points()
                        if not spawn_points:
                            print("Warning: No spawn points found, using default transform")
                            ego_transform = carla.Transform(carla.Location(x=0, y=0, z=1), carla.Rotation())
                        else:
                            ego_transform = random.choice(spawn_points)

                        # Initialize variables
                        episode_reward = 0
                        done = False
                        step = 0

                        # Create ego vehicle
                        ego_vehicle = EgoVehicle(world, transform=ego_transform)
                        prev_location = ego_vehicle.vehicle.get_transform().location

                        # Set up spectator to follow vehicle
                        update_spectator = attach_spectator_to_vehicle(world, ego_vehicle.vehicle, 'third_person_rear')

                        # Training loop for this episode
                        while not done and step < steps_per_episode:
                            if not server_manager.is_server_running():
                                print("CARLA server crashed during episode. Will restart.")
                                raise Exception("CARLA server crashed")

                            try:
                                # Process world tick with timeout handling
                                try:
                                    world.tick(timeout=2.0)  # Short timeout for quicker detection of issues
                                except Exception as e:
                                    print(f"Error during world tick: {e}")
                                    raise Exception("Tick failed")

                                # Update spectator
                                update_spectator()

                                # Get current state
                                rgb_image = ego_vehicle.sensor_data['rgb_front']
                                if rgb_image is None:
                                    print("Warning: No RGB image received")
                                    continue

                                state = preprocess_image(rgb_image)

                                # Select action
                                action, action_prob, value = agent.act(state)

                                # Apply control
                                control = carla.VehicleControl()
                                control.throttle = float(max(0, min(1.0, action[0])))
                                control.steer = float(max(-1.0, min(1.0, action[1])))
                                control.brake = float(max(0, min(1.0, action[2])))

                                ego_vehicle.apply_control(control)

                                # Calculate reward
                                reward, current_location = calculate_reward(ego_vehicle, prev_location)
                                prev_location = current_location

                                # Check for termination
                                if ego_vehicle.sensor_data['collision']:
                                    print("Collision detected! Ending episode.")
                                    done = True

                                # Store experience
                                agent.remember(state, action, action_prob, reward, done, value)

                                # Update rewards
                                episode_reward += reward

                                # Update policy if needed
                                total_steps += 1
                                if total_steps % update_frequency == 0:
                                    print("Updating policy...")
                                    agent.update()

                                step += 1

                            except Exception as e:
                                print(f"Error in step {step}: {e}")
                                done = True  # End episode on error

                        # End of episode
                        print(f"Episode {current_episode_number + 1} completed with reward: {episode_reward}")

                        # Update policy at episode end
                        if len(agent.memory['states']) > 0:
                            agent.update()

                        # Save model if best
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            agent.save("carla_ppo_best.pth")
                            print(f"New best model saved with reward: {best_reward}")

                        # Save checkpoint every few episodes
                        if (current_episode_number + 1) % 5 == 0:
                            agent.save(f"carla_ppo_checkpoint_ep{current_episode_number + 1}.pth")

                            # Save training state for resuming
                            torch.save({
                                'episode': current_episode_number + 1,
                                'best_reward': best_reward,
                                'model_state_dict': agent.policy.state_dict(),
                                'optimizer_state_dict': agent.optimizer.state_dict(),
                            }, 'carla_checkpoint.pth')

                    except Exception as e:
                        print(f"Error in episode {current_episode_number + 1}: {e}")

                    finally:
                        # Clean up
                        if 'ego_vehicle' in locals() and ego_vehicle is not None:
                            try:
                                ego_vehicle.destroy()
                            except Exception as e:
                                print(f"Error destroying ego vehicle: {e}")

                        # Safely destroy other actors
                        try:
                            for list_name, actor_list in [
                                ('controllers', controllers if 'controllers' in locals() else []),
                                ('walkers', walkers if 'walkers' in locals() else []),
                                ('vehicles', vehicles if 'vehicles' in locals() else [])
                            ]:
                                if actor_list:
                                    for actor in actor_list:
                                        try:
                                            if hasattr(actor, 'is_alive') and actor.is_alive:
                                                if list_name == 'controllers':
                                                    actor.stop()
                                                actor.destroy()
                                        except Exception as e:
                                            pass  # Ignore errors during cleanup
                        except Exception as e:
                            print(f"Error in cleanup: {e}")

                # Increment current episode count
                current_episode += episodes_in_batch

                # Restore original settings
                try:
                    world.apply_settings(original_settings)
                except Exception as e:
                    print(f"Error restoring settings: {e}")

                # Restart CARLA after batch of episodes
                server_manager.restart_server()

            except Exception as e:
                print(f"Error connecting to CARLA: {e}")
                server_manager.restart_server()

        # Training complete
        print(f"Training completed after {current_episode} episodes and {server_restarts} server restarts.")
        agent.save("carla_ppo_final.pth")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Final cleanup
        server_manager.stop_server()


# Function for camera view
def attach_spectator_to_vehicle(world, vehicle, view_mode='third_person_rear'):
    """
    Set up the simulator view to follow the vehicle
    """
    try:
        spectator = world.get_spectator()

        # Define camera offsets
        views = {
            'third_person_rear': {
                'offset': carla.Location(x=-5.5, z=2.8),
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

        if view_mode not in views:
            view_mode = 'third_person_rear'

        view = views[view_mode]

        def update_spectator():
            try:
                if vehicle and hasattr(vehicle, 'is_alive') and vehicle.is_alive:
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

                        direction = vehicle_transform.location - spectator_location
                        spectator_rotation = carla.Rotation(
                            pitch=view['rotation_offset'].pitch,
                            yaw=math.degrees(math.atan2(direction.y, direction.x)),
                            roll=0
                        )

                    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))
            except Exception as e:
                pass  # Silently ignore errors to avoid interrupting training

        return update_spectator
    except Exception as e:
        return lambda: None  # Return dummy function on error


# Main entry point
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CARLA Robust Training')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Total number of episodes')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Episodes per CARLA restart')
    parser.add_argument('--max-restarts', type=int, default=20,
                        help='Maximum number of CARLA server restarts')

    args = parser.parse_args()

    robust_training(
        episodes=args.episodes,
        episodes_per_restart=args.batch_size,
        max_server_restarts=args.max_restarts
    )