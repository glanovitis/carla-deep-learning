#!/usr/bin/env python

import os
import sys
import argparse
import glob

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
from data_collector import DataCollector, calculate_reward
from carla_rl_agent import train_agent, test_agent


def setup_carla_environment():
    """
    Setup CARLA environment and return client and world
    """
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

    return client, world, original_settings


def run_data_collection(client, world, num_samples=2000):
    """
    Run the data collection process
    """
    from data_collector import main
    print("Running data collection...")
    main()


def run_training(client, world, episodes=100):
    """
    Run the training process
    """
    from carla_rl_agent import train_agent
    print("Running training...")
    train_agent(episodes=episodes)


def run_testing(client, world, model_path):
    """
    Run the testing process
    """
    from carla_rl_agent import test_agent
    print(f"Testing model from {model_path}...")
    test_agent(model_path=model_path)


def main():
    parser = argparse.ArgumentParser(description='CARLA Deep Learning Navigation')
    parser.add_argument('--mode', type=str, default='train', choices=['collect', 'train', 'test'],
                        help='Mode: collect data, train model, or test model')
    parser.add_argument('--model', type=str, default='carla_ppo_best.pth',
                        help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for training')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples for data collection')

    args = parser.parse_args()

    client = None
    world = None
    original_settings = None

    try:
        # Setup CARLA environment
        client, world, original_settings = setup_carla_environment()

        if args.mode == 'collect':
            run_data_collection(client, world, num_samples=args.samples)
        elif args.mode == 'train':
            run_training(client, world, episodes=args.episodes)
        elif args.mode == 'test':
            run_testing(client, world, model_path=args.model)

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Restore original settings
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
                print("Original settings restored")
            except Exception as e:
                print(f"Error restoring original settings: {e}")


if __name__ == '__main__':
    main()