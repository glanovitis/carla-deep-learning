import carla
import random
import numpy as np
import cv2
import time
import os


def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set up the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Get a vehicle blueprint
    vehicle_bp = blueprint_library.filter('model3')[0]

    # Get a random spawn point
    spawn_point = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    try:
        # Set up the camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')

        # Camera relative position to the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        # Spawn the camera and attach to vehicle
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Create output folder
        output_dir = 'output/training_data'
        os.makedirs(output_dir, exist_ok=True)

        # Data collection function
        def process_image(image):
            # Convert image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel

            # Save image
            img_path = os.path.join(output_dir, f"{image.frame:06d}.png")
            cv2.imwrite(img_path, array)

            # Record steering angle (you'd need to track this separately)
            steering = vehicle.get_control().steer
            with open(os.path.join(output_dir, "steering_data.txt"), "a") as f:
                f.write(f"{image.frame:06d}.png,{steering}\n")

        # Register callback function
        camera.listen(process_image)

        # Drive the vehicle manually or with autopilot
        vehicle.set_autopilot(True)

        # Collect data for 60 seconds
        print("Collecting data...")
        time.sleep(60)

    finally:
        # Clean up actors
        if 'camera' in locals():
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        print("Data collection complete")


if __name__ == '__main__':
    main()