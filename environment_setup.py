#!/usr/bin/env python

import glob
import os
import sys
import time
import numpy as np
import random
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def spawn_traffic(client, num_vehicles=30, num_pedestrians=20):
    """
    Spawns vehicles and pedestrians in the CARLA world
    """
    world = client.get_world()

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Get the spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn vehicles
    vehicle_bps = blueprint_library.filter('vehicle.*')

    vehicles_list = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)

        # Set autopilot mode
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Try to spawn the vehicle
        if spawn_points:
            transform = random.choice(spawn_points)
            vehicle = world.try_spawn_actor(bp, transform)
            if vehicle is not None:
                vehicle.set_autopilot(True)
                vehicles_list.append(vehicle)

    print(f'Spawned {len(vehicles_list)} vehicles')

    # Spawn pedestrians
    pedestrian_bps = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')

    walkers_list = []
    controllers_list = []

    # Spawn pedestrians with controllers
    for i in range(num_pedestrians):
        bp = random.choice(pedestrian_bps)

        # Randomize pedestrian attributes
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')

        # Spawn the pedestrian
        spawn_point = carla.Transform()
        spawn_point.location = world.get_random_location_from_navigation()

        if spawn_point.location:
            walker = world.try_spawn_actor(bp, spawn_point)
            if walker is not None:
                # Spawn controller for the walker
                controller = world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1.4)  # Average human walking speed

                walkers_list.append(walker)
                controllers_list.append(controller)

    print(f'Spawned {len(walkers_list)} pedestrians')

    return vehicles_list, walkers_list, controllers_list


def safe_destroy_actors(client, actor_list):
    """
    Safely destroy a list of actors with error handling
    """
    commands = []
    for actor in actor_list:
        try:
            if actor.is_alive:
                commands.append(carla.command.DestroyActor(actor))
        except Exception as e:
            print(f"Error preparing actor for destruction: {e}")

    # Use batch apply for efficiency if there are any valid actors to destroy
    if commands:
        try:
            client.apply_batch_sync(commands, True)
            print(f"Destroyed {len(commands)} actors successfully")
        except Exception as e:
            print(f"Error in batch destroy: {e}")


def main():
    try:
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # Get the world
        world = client.get_world()

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn traffic
        vehicles, walkers, controllers = spawn_traffic(client, 30, 20)

        print("Press Ctrl+C to exit")
        while True:
            world.tick()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping simulation")
    finally:
        # Clean up
        print("Cleaning up spawned actors")

        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        if 'controllers' in locals() and controllers:
            safe_destroy_actors(client, controllers)
        if 'walkers' in locals() and walkers:
            safe_destroy_actors(client, walkers)
        if 'vehicles' in locals() and vehicles:
            safe_destroy_actors(client, vehicles)

        print("Actors destroyed")


if __name__ == '__main__':
    main()