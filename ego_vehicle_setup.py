#!/usr/bin/env python

import glob
import os
import sys
import time
import numpy as np
import cv2
import queue
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class EgoVehicle:
    def __init__(self, world, vehicle_type='lincoln.mkz_2017', transform=None):
        """
        Creates an ego vehicle with various sensors
        """
        self.world = world
        self.blueprint_library = world.get_blueprint_library()

        # Create vehicle - either use specified type or choose random vehicle
        if vehicle_type:
            try:
                # Use specified vehicle if provided
                vehicle_filters = self.blueprint_library.filter(f'vehicle.{vehicle_type}')
                if len(vehicle_filters) > 0:
                    self.vehicle_bp = vehicle_filters[0]
                else:
                    print(f"Warning: Vehicle type {vehicle_type} not found, using random vehicle")
                    self.vehicle_bp = self._get_random_vehicle_bp()
            except:
                print("Error using specified vehicle type, falling back to tesla.model3")
                self.vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        else:
            # Use random vehicle
            self.vehicle_bp = self._get_random_vehicle_bp()

        # Get spawn points
        spawn_points = world.get_map().get_spawn_points()

        # If no transform provided, use random spawn point
        if transform is None and spawn_points:
            transform = random.choice(spawn_points)

        # Try to spawn the vehicle
        self.vehicle = None
        max_attempts = 20  # Try up to 20 different spawn points

        for attempt in range(max_attempts):
            try:
                print(f"Attempting to spawn ego vehicle (attempt {attempt + 1}/{max_attempts})...")
                self.vehicle = world.spawn_actor(self.vehicle_bp, transform)
                print(f"Ego vehicle spawned successfully: {self.vehicle.type_id}")
                break
            except RuntimeError as e:
                print(f"Spawn attempt {attempt + 1} failed: {e}")
                if "collision" in str(e).lower() and spawn_points and attempt < max_attempts - 1:
                    print("Trying a different spawn point...")
                    transform = random.choice(spawn_points)
                    # Wait a bit before next attempt
                    time.sleep(0.1)
                else:
                    # If we've tried all options or got a different error
                    raise RuntimeError(f"Failed to spawn ego vehicle after {attempt + 1} attempts: {e}")

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle after maximum attempts")

        # Initialize sensor data storage
        self.sensor_data = {
            'rgb_front': None,
            'depth_front': None,
            'lidar': None,
            'collision': False,
            'lane_invasion': False,
            'gnss': None
        }

        self.sensor_queues = {
            'rgb_front': queue.Queue(),
            'depth_front': queue.Queue(),
            'lidar': queue.Queue(),
            'collision': queue.Queue(),
            'lane_invasion': queue.Queue(),
            'gnss': queue.Queue()
        }

        self.sensors = []
        self._setup_sensors()

    def _setup_sensors(self):
        """
        Sets up all sensors on the ego vehicle
        """
        # RGB Camera
        rgb_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', '800')
        rgb_bp.set_attribute('image_size_y', '600')
        rgb_bp.set_attribute('fov', '90')
        rgb_transform = carla.Transform(carla.Location(x=2.0, z=1.7))
        self.rgb_cam = self.world.spawn_actor(rgb_bp, rgb_transform, attach_to=self.vehicle)
        self.sensors.append(self.rgb_cam)

        # Set callback for RGB camera
        self.rgb_cam.listen(lambda image: self._process_rgb_image(image))

        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '800')
        depth_bp.set_attribute('image_size_y', '600')
        depth_bp.set_attribute('fov', '90')
        depth_transform = carla.Transform(carla.Location(x=2.0, z=1.7))
        self.depth_cam = self.world.spawn_actor(depth_bp, depth_transform, attach_to=self.vehicle)
        self.sensors.append(self.depth_cam)

        # Set callback for depth camera
        self.depth_cam.listen(lambda image: self._process_depth_image(image))

        # LIDAR
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('range', '100')
        lidar_transform = carla.Transform(carla.Location(z=2.0))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.sensors.append(self.lidar)

        # Set callback for LIDAR
        self.lidar.listen(lambda point_cloud: self._process_lidar_data(point_cloud))

        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(self.collision_sensor)

        # Set callback for collision sensor
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Lane Invasion Sensor
        lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(self.lane_invasion_sensor)

        # Set callback for lane invasion
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

        # GNSS Sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        self.gnss_sensor = self.world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(self.gnss_sensor)

        # Set callback for GNSS
        self.gnss_sensor.listen(lambda event: self._on_gnss_update(event))

    @staticmethod
    def efficient_image_processing(image, is_rgb=True):
        """
        Process CARLA image data without memory leaks
        Returns numpy array without saving to disk
        """
        # Convert raw data to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))

        if is_rgb:
            # RGB image processing - keep 3 channels
            array = array[:, :, :3]  # Keep RGB channels only
        else:
            # Semantic segmentation - keep only first channel
            array = array[:, :, 0]

        return array.copy()  # Return a copy to ensure memory is properly managed

    def _process_rgb_image(self, image):
        """Memory-optimized RGB image processing"""
        array = self.efficient_image_processing(image, is_rgb=True)
        self.sensor_data['rgb_front'] = array
        self.sensor_queues['rgb_front'].put(array)

    def _process_depth_image(self, image):
        """Memory-optimized depth image processing"""
        # Get raw data as uint8 array with RGB channels
        array = self.efficient_image_processing(image, is_rgb=True)

        # Convert to float32 before depth calculation to avoid overflow
        # Use astype(np.float32) instead of float which would use float64 and double memory usage
        array = array.astype(np.float32)

        # Calculate depth - using vectorized operations for better performance
        # Depth is encoded in the RGB values
        normalized = (array[:, :, 0] + array[:, :, 1] * 256 + array[:, :, 2] * 256 * 256) / (256 * 256 * 256 - 1)
        depth_meters = normalized * 1000  # Convert to meters

        # Store only the depth map (single channel) to reduce memory usage
        self.sensor_data['depth_front'] = depth_meters
        self.sensor_queues['depth_front'].put(depth_meters)

    def _process_lidar_data(self, point_cloud):
        """Memory-optimized LIDAR processing"""
        # Direct buffer access for better memory management
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # Store a copy to ensure the original buffer can be released
        self.sensor_data['lidar'] = data.copy()
        self.sensor_queues['lidar'].put(data)

    def _on_collision(self, event):
        self.sensor_data['collision'] = True
        self.sensor_queues['collision'].put(True)
        print(f"Collision detected with {event.other_actor.type_id}")

    def _on_lane_invasion(self, event):
        self.sensor_data['lane_invasion'] = True
        self.sensor_queues['lane_invasion'].put(True)
        print("Lane invasion detected")

    def _on_gnss_update(self, event):
        self.sensor_data['gnss'] = (event.latitude, event.longitude, event.altitude)
        self.sensor_queues['gnss'].put((event.latitude, event.longitude, event.altitude))

    def apply_control(self, control):
        """
        Apply vehicle control (throttle, steer, brake)
        """
        self.vehicle.apply_control(control)

    def _get_random_vehicle_bp(self):
        """Get a random vehicle blueprint that's suitable for an ego vehicle"""
        try:
            # Get all available car blueprints
            all_vehicles = self.blueprint_library.filter('vehicle.*')

            # Make sure we have vehicles
            if not all_vehicles:
                print("No vehicles found in blueprint library, using default")
                return self.blueprint_library.find('vehicle.tesla.model3')

            # Filter out bicycles, motorcycles, and other problematic vehicles
            car_blueprints = []
            for bp in all_vehicles:
                if (not bp.id.startswith('vehicle.bh.') and
                        not bp.id.startswith('vehicle.bicycle') and
                        not bp.id.startswith('vehicle.motorcycle')):
                    car_blueprints.append(bp)

            # Make sure we have cars after filtering
            if not car_blueprints:
                print("No suitable cars after filtering, using default")
                return self.blueprint_library.find('vehicle.tesla.model3')

            # Choose a random vehicle blueprint
            vehicle_bp = random.choice(car_blueprints)
            print(f"Selected random vehicle: {vehicle_bp.id}")

            # Set random color
            if vehicle_bp.has_attribute('color'):
                colors = vehicle_bp.get_attribute('color').recommended_values
                if colors:
                    color = random.choice(colors)
                    vehicle_bp.set_attribute('color', color)

            return vehicle_bp
        except Exception as e:
            print(f"Error selecting random vehicle: {e}")
            # Fall back to a known vehicle
            return self.blueprint_library.find('vehicle.tesla.model3')

    def get_sensor_data(self):
        """
        Returns the current sensor data
        """
        return self.sensor_data

    def destroy(self):
        """
        Destroy all actors created
        """
        print("Destroying ego vehicle and sensors...")

        # Safely destroy sensors first
        for sensor in self.sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception as e:
                print(f"Error destroying sensor: {e}")

        # Then destroy the vehicle
        try:
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
        except Exception as e:
            print(f"Error destroying vehicle: {e}")