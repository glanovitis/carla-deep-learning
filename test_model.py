import carla
import random
import numpy as np
import cv2
import tensorflow as tf
import time


def process_image(image, model):
    # Convert image to numpy array
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha channel

    # Preprocess the image for the model
    img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))  # Same size as training
    img = img / 255.0  # Normalize

    # Predict steering angle
    steering_angle = float(model.predict(np.array([img]))[0])

    # Apply the predicted steering to the vehicle
    control = carla.VehicleControl()
    control.steer = steering_angle
    control.throttle = 0.5  # Constant speed for simplicity
    control.brake = 0.0

    return control


def main():
    # Load trained model
    model = tf.keras.models.load_model('models/self_driving_model.h5')

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

        # Control the vehicle using the model
        def image_callback(image):
            control = process_image(image, model)
            vehicle.apply_control(control)

        # Register callback function
        camera.listen(image_callback)

        # Keep simulation running for 60 seconds
        print("Running autonomous mode...")
        time.sleep(60)

    finally:
        # Clean up actors
        if 'camera' in locals():
            camera.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        print("Autonomous test complete")


if __name__ == '__main__':
    main()