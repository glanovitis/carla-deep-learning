#!/usr/bin/env python

import glob
import os
import sys
import time
import numpy as np
import cv2
import queue
import h5py
import datetime

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


class DataCollector:
    def __init__(self, ego_vehicle, buffer_size=1000):
        """
        Collects training data from the ego vehicle's sensors
        """
        self.ego_vehicle = ego_vehicle
        self.buffer_size = buffer_size

        # Data buffers
        self.rgb_buffer = []
        self.depth_buffer = []
        self.lidar_buffer = []
        self.collision_buffer = []
        self.lane_invasion_buffer = []
        self.gnss_buffer = []
        self.control_buffer = []
        self.reward_buffer = []

        self.current_control = None

    def collect_step(self, control, reward):
        """
        Collect a single step of data
        """
        # Store control
        self.current_control = control

        # Get sensor data
        sensor_data = self.ego_vehicle.get_sensor_data()

        # Store data
        self.rgb_buffer.append(sensor_data['rgb_front'].copy() if sensor_data['rgb_front'] is not None else None)
        self.depth_buffer.append(sensor_data['depth_front'].copy() if sensor_data['depth_front'] is not None else None)
        self.lidar_buffer.append(sensor_data['lidar'].copy() if sensor_data['lidar'] is not None else None)
        self.collision_buffer.append(sensor_data['collision'])
        self.lane_invasion_buffer.append(sensor_data['lane_invasion'])
        self.gnss_buffer.append(sensor_data['gnss'])
        self.control_buffer.append([control.throttle, control.steer, control.brake])
        self.reward_buffer.append(reward)

        # Trim buffers if they exceed buffer size
        if len(self.rgb_buffer) > self.buffer_size:
            self.rgb_buffer = self.rgb_buffer[-self.buffer_size:]
            self.depth_buffer = self.depth_buffer[-self.buffer_size:]
            self.lidar_buffer = self.lidar_buffer[-self.buffer_size:]
            self.collision_buffer = self.collision_buffer[-self.buffer_size:]
            self.lane_invasion_buffer = self.lane_invasion_buffer[-self.buffer_size:]
            self.gnss_buffer = self.gnss_buffer[-self.buffer_size:]
            self.control_buffer = self.control_buffer[-self.buffer_size:]
            self.reward_buffer = self.reward_buffer[-self.buffer_size:]

    def save_data(self, filename=None):
        """
        Save collected data to a HDF5 file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"carla_data_{timestamp}.h5"

        print(f"Saving data to {filename}")

        with h5py.File(filename, 'w') as hf:
            # Save RGB images
            rgb_data = np.array([img for img in self.rgb_buffer if img is not None])
            if len(rgb_data) > 0:
                hf.create_dataset('rgb', data=rgb_data, compression="gzip")

            # Save depth images
            depth_data = np.array([img for img in self.depth_buffer if img is not None])
            if len(depth_data) > 0:
                hf.create_dataset('depth', data=depth_data, compression="gzip")

            # Save LIDAR data - this requires special handling due to variable size
            lidar_data = [pc for pc in self.lidar_buffer if pc is not None]
            if len(lidar_data) > 0:
                lidar_group = hf.create_group('lidar')
                for i, pc in enumerate(lidar_data):
                    lidar_group.create_dataset(f'scan_{i}', data=pc, compression="gzip")

            # Save other data
            hf.create_dataset('collision', data=np.array(self.collision_buffer))
            hf.create_dataset('lane_invasion', data=np.array(self.lane_invasion_buffer))

            # Filter out None values for GNSS
            gnss_data = np.array([g if g is not None else [0, 0, 0] for g in self.gnss_buffer])
            hf.create_dataset('gnss', data=gnss_data)

            hf.create_dataset('control', data=np.array(self.control_buffer))
            hf.create_dataset('reward', data=np.array(self.reward_buffer))

        print(f"Data saved successfully with {len(self.rgb_buffer)} timesteps")


def calculate_reward(ego_vehicle, prev_location=None, target_speed=30):
    """
    Calculate reward based on:
    1. Forward progress toward destination
    2. Speed maintenance (penalty for too slow or too fast)
    3. Collision penalty
    4. Lane invasion penalty
    """
    reward = 0

    # Get current state
    transform = ego_vehicle.vehicle.get_transform()
    velocity = ego_vehicle.vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h
    location = transform.location

    # Forward progress reward
    if prev_location is not None:
        # Calculate distance traveled in the forward direction
        forward_vector = transform.get_forward_vector()
        movement_vector = carla.Vector3D(
            location.x - prev_location.x,
            location.y - prev_location.y,
            location.z - prev_location.z
        )

        # Project movement onto forward direction
        forward_progress = (forward_vector.x * movement_vector.x +
                            forward_vector.y * movement_vector.y +
                            forward_vector.z * movement_vector.z)

        # Reward forward progress
        reward += forward_progress * 10.0

    # Speed maintenance reward/penalty
    speed_diff = abs(speed - target_speed)
    speed_reward = -0.05 * speed_diff
    reward += speed_reward

    # Collision penalty
    if ego_vehicle.sensor_data['collision']:
        reward -= 100

    # Lane invasion penalty
    if ego_vehicle.sensor_data['lane_invasion']:
        reward -= 5

    return reward, location


def main():
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

        # Spawn traffic
        vehicles, walkers, controllers = spawn_traffic(client, 20, 10)

        # Choose a good spawn point
        spawn_points = world.get_map().get_spawn_points()
        ego_transform = spawn_points[0] if spawn_points else carla.Transform()

        # Create the ego vehicle
        ego_vehicle = EgoVehicle(world, ego_transform)

        # Create data collector
        data_collector = DataCollector(ego_vehicle)

        # Previous location for reward calculation
        prev_location = None

        # Simple control loop to collect data
        print("Starting data collection. Press Ctrl+C to exit.")
        for i in range(1000):  # Collect 1000 steps of data
            # Process world tick
            world.tick()

            # Apply a varied control to explore the environment
            control = carla.VehicleControl()

            # Varied control pattern for exploration
            if i % 300 < 100:
                control.throttle = 0.5
                control.steer = 0
            elif i % 300 < 200:
                control.throttle = 0.4
                control.steer = 0.3
            else:
                control.throttle = 0.4
                control.steer = -0.3

            # Random brake occasionally
            if i % 50 == 0:
                control.brake = 0.5
            else:
                control.brake = 0

            ego_vehicle.apply_control(control)

            # Calculate reward
            reward, current_location = calculate_reward(ego_vehicle, prev_location)
            prev_location = current_location

            # Collect data
            data_collector.collect_step(control, reward)

            # Display progress
            if i % 100 == 0:
                print(f"Collected {i} steps of data")

            time.sleep(0.01)  # Small sleep to not block

        # Save the collected data
        data_collector.save_data()

    except KeyboardInterrupt:
        print("Data collection interrupted")
    finally:
        # Clean up
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()

        # Destroy traffic
        if 'controllers' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
        if 'walkers' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in walkers])
        if 'vehicles' in locals():
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])

        # Restore original settings
        if 'original_settings' in locals():
            world.apply_settings(original_settings)


if __name__ == '__main__':
    main()