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


def calculate_reward_1(ego_vehicle, prev_location=None, prev_speed=None,
                     distance_traveled=0, total_steps=0, prev_steering=None, target_speed=30):
    """
    Optimized reward function for CARLA deep learning
    """
    reward = 0

    # Get current state
    transform = ego_vehicle.vehicle.get_transform()
    velocity = ego_vehicle.vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h
    location = transform.location
    current_steering = ego_vehicle.vehicle.get_control().steer

    # --- INITIAL ACCELERATION INCENTIVE ---
    # Strong reward for any initial acceleration from standstill
    if prev_speed is not None and prev_speed < 5.0:
        # More reward for acceleration from low speeds
        acceleration = max(0, speed - prev_speed)
        # Higher multiplier for starting from near-zero speed
        accel_multiplier = max(1.0, 5.0 - prev_speed)
        accel_reward = acceleration * accel_multiplier * 0.5
        reward += accel_reward

    # --- FORWARD PROGRESS REWARD (ENHANCED) ---
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

        # Enhanced forward progress reward (increased multiplier)
        # Award more in early training and for first movements
        if total_steps < 10000:  # Early training phase
            forward_multiplier = 30.0  # Increased from 20.0
        else:
            forward_multiplier = 20.0

        reward += forward_progress * forward_multiplier

        # Extra bonus for consistent forward movement
        if forward_progress > 0.01:  # If making meaningful progress
            reward += 0.2  # Small constant bonus for any forward progress

    # --- MILESTONE REWARDS ---
    # Distance milestone rewards (every 5 meters)
    new_distance = distance_traveled
    if prev_location is not None:
        # Add distance traveled since last step
        step_distance = location.distance(prev_location)
        new_distance += step_distance

    milestone = int(new_distance / 5)
    prev_milestone = int(distance_traveled / 5)
    if milestone > prev_milestone:
        # Significant bonus for each 5m milestone
        reward += 2.0 * (milestone - prev_milestone)

    # --- SPEED MAINTENANCE (SMOOTHER) ---
    # More lenient speed penalty early in training
    if total_steps < 10000:  # Early training
        # Just reward any speed, less concern about target
        speed_reward = min(speed * 0.05, 1.0)  # Cap at 1.0
        reward += speed_reward
    else:
        # Original speed maintenance reward/penalty, but gentler slope
        speed_diff = abs(speed - target_speed)
        speed_reward = -0.03 * speed_diff  # Reduced from -0.05
        reward += speed_reward

    # --- ANTI-WIGGLING PENALTIES ---
    # Penalize steering changes when not moving (prevents wheel wiggling)
    if speed < 2.0 and abs(current_steering) > 0.1:
        # Stronger penalty for steering while nearly stationary
        wiggle_penalty = abs(current_steering) * 0.5
        reward -= wiggle_penalty

    # Penalize rapid steering direction changes (back and forth)
    if prev_steering is not None:
        steering_change = abs(current_steering - prev_steering)
        if steering_change > 0.2 and speed < 5.0:
            # Penalize quick steering reversals at low speed
            reward -= steering_change * 0.6

    # --- REGULAR PENALTIES (ADJUSTED) ---
    # Collision penalty - adaptive based on training progress
    if ego_vehicle.sensor_data['collision']:
        if total_steps < 20000:  # Early training
            collision_penalty = 50  # Reduced from 200
        else:
            collision_penalty = 200  # Original value
        reward -= collision_penalty

    # Lane invasion penalty - slightly increased
    if ego_vehicle.sensor_data['lane_invasion']:
        reward -= 5  # Increased from 2

    return reward, location, speed, new_distance, current_steering


def calculate_reward(ego_vehicle, prev_location=None, prev_speed=None, distance_traveled=0, total_steps=0,
                     prev_steering=None, target_speed=30, last_collision_step=None):
    """Optimized reward function with collision event tracking"""
    reward = 0

    # Get current state
    transform = ego_vehicle.vehicle.get_transform()
    velocity = ego_vehicle.vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)  # km/h
    location = transform.location
    current_steering = ego_vehicle.vehicle.get_control().steer

    # Set collision tracking variables
    collision_detected = ego_vehicle.sensor_data['collision']
    new_collision = False
    this_collision_step = last_collision_step

    # Check if this is a new collision event
    if collision_detected:
        if last_collision_step is None or (total_steps - last_collision_step) > 20:
            # This is a new collision if we haven't had one before,
            # or if it's been at least 20 steps since the last one
            new_collision = True
            this_collision_step = total_steps
            print(f"Collision detected with {ego_vehicle.sensor_data.get('collision_type', 'unknown')}")

    # === MOVEMENT REWARDS (Same as before) ===
    if speed > 0.1:  # Any movement at all
        speed_reward = min(speed * 0.3, 10.0)  # Capped at 10.0
        reward += speed_reward

        # Extra bonus for getting faster than previous speed
        if prev_speed is not None and speed > prev_speed:
            acceleration_bonus = (speed - prev_speed) * 2.0
            if prev_speed < 2.0:
                acceleration_bonus *= 3.0
            reward += acceleration_bonus

    # === ANTI-WIGGLING (Same as before) ===
    if total_steps > 20:
        if speed < 0.5 and abs(current_steering) > 0.2:
            wiggle_penalty = min(abs(current_steering) * 0.3, 0.5)
            reward -= wiggle_penalty

        if prev_steering is not None and total_steps > 50:
            steering_change = abs(current_steering - prev_steering)
            if steering_change > 0.3 and speed < 1.0:
                reward -= steering_change * 0.3

    # === MOVEMENT TRACKING (Same as before) ===
    step_distance = 0
    new_distance = distance_traveled

    if prev_location is not None:
        step_distance = location.distance(prev_location)
        forward_vector = transform.get_forward_vector()
        movement_vector = carla.Vector3D(
            location.x - prev_location.x,
            location.y - prev_location.y,
            location.z - prev_location.z
        )
        forward_progress = (forward_vector.x * movement_vector.x +
                            forward_vector.y * movement_vector.y +
                            forward_vector.z * movement_vector.z)

        if forward_progress > 0:
            forward_reward = forward_progress * 25.0
            reward += forward_reward
            new_distance += step_distance

    # === MILESTONE REWARDS (Same as before) ===
    milestone = int(new_distance / 5)
    prev_milestone = int(distance_traveled / 5)
    if milestone > prev_milestone:
        milestone_reward = 5.0 * (milestone - prev_milestone)
        reward += milestone_reward
        print(f"Milestone reached! {milestone * 5}m - Bonus: +{milestone_reward}")

    # === EMERGENCY ACCELERATION BOOST (Same as before) ===
    if total_steps > 20 and speed < 0.1 and ego_vehicle.vehicle.get_control().throttle > 0.5:
        reward += 0.5

    # === PENALTIES - ONLY FOR NEW COLLISIONS ===
    # Apply collision penalty ONLY when a new collision is detected
    if new_collision:
        collision_penalty = 50 if total_steps < 20000 else 200
        reward -= collision_penalty
        print(f"Applied collision penalty: -{collision_penalty}")

    # Lane invasion - minor penalty
    if ego_vehicle.sensor_data['lane_invasion']:
        reward -= 2

    # Determine if episode should end
    done = False
    if new_collision:
        # End episode only on significant new collisions
        collision_impulse = ego_vehicle.sensor_data.get('collision_impulse', 0)
        if collision_impulse > 10:
            done = True
            print("Significant collision detected - ending episode")

    # Return the updated collision step along with other values
    return reward, location, speed, new_distance, current_steering, this_collision_step, done

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