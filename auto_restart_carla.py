#!/usr/bin/env python

import os
import sys
import argparse
import glob
import time
import subprocess
import signal
import platform
import psutil
import torch
import logging
import socket
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler("carla_training.log"),
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarlaServerManager:
    def __init__(self, carla_path=None, port=2000, low_quality=True):
        self.port = port
        self.low_quality = low_quality
        self.process = None

        # Find CARLA path
        if carla_path is None:
            if platform.system() == 'Windows':
                possible_paths = [
                    r".\WindowsNoEditor\CarlaUE4.exe",
                    r"C:\Program Files\Epic Games\CARLA\WindowsNoEditor\CarlaUE4.exe",
                    r"C:\CARLA\WindowsNoEditor\CarlaUE4.exe"
                ]
            else:
                possible_paths = [
                    "/opt/carla-simulator/CarlaUE4.sh",
                    os.path.expanduser("~/carla/CarlaUE4.sh")
                ]

            for path in possible_paths:
                if os.path.exists(path):
                    carla_path = path
                    break

        self.carla_path = carla_path
        if not self.carla_path or not os.path.exists(self.carla_path):
            logger.warning(f"WARNING: CARLA executable not found at {self.carla_path}")

    def is_port_in_use(self, port):
        """Check if a port is in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except Exception as e:
            logger.warning(f"Error checking port {port}: {e}")
            # If there's an error checking, assume it's not in use
            return False

    def kill_all_carla_processes(self):
        """Kill all running CARLA processes"""
        logger.info("Killing all CARLA processes...")

        # Track whether we succeeded
        success = False

        try:
            if platform.system() == 'Windows':
                try:
                    # Force kill any CarlaUE4 processes
                    subprocess.run(['taskkill', '/F', '/IM', 'CarlaUE4.exe'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(5)  # Give more time for processes to terminate

                    # Double check - kill any remaining processes by name
                    for proc in psutil.process_iter(['pid', 'name']):
                        try:
                            name = proc.info['name'].lower()
                            if 'carla' in name or 'ue4' in name:
                                logger.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                                psutil.Process(proc.info['pid']).kill()
                        except Exception as inner_e:
                            logger.warning(f"Failed to kill process: {inner_e}")
                            pass
                except Exception as e:
                    logger.error(f"Error killing CARLA processes: {e}")
            else:
                try:
                    # Find and kill all CARLA processes on Linux/Mac
                    for proc in psutil.process_iter(['pid', 'name']):
                        if 'carla' in proc.info['name'].lower():
                            logger.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                            try:
                                os.kill(proc.info['pid'], signal.SIGKILL)
                            except Exception as inner_e:
                                logger.warning(f"Failed to kill process: {inner_e}")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Error killing CARLA processes: {e}")

            # Wait for port to be released
            max_tries = 15  # Increased number of retries
            for i in range(max_tries):
                if not self.is_port_in_use(self.port):
                    logger.info(f"Port {self.port} is now free")
                    success = True
                    break
                logger.warning(f"Port {self.port} still in use, waiting... ({i + 1}/{max_tries})")
                time.sleep(3)  # Longer wait between checks

            # Even if port still seems to be in use, wait extra time
            # Sometimes the port reports as in use but isn't actually bound
            if not success:
                logger.warning("Port still appears to be in use, but proceeding after delay")
                time.sleep(10)  # Wait extra time before continuing anyway

            return True
        except Exception as e:
            logger.error(f"Error in kill_all_carla_processes: {e}")
            time.sleep(15)  # Long wait as a fallback
            return False

    def start_carla_server(self):
        """Start the CARLA server with retry mechanism"""
        if not self.carla_path:
            logger.error("CARLA path not found. Cannot start server.")
            return False

        # First kill any existing processes
        logger.info("Preparing to start CARLA - killing existing processes first")
        self.kill_all_carla_processes()

        # Add an extra sleep before checking port again
        time.sleep(5)

        # Check if port is free
        if self.is_port_in_use(self.port):
            logger.error(f"Port {self.port} is still in use after cleanup. Will try to proceed anyway.")
            # We'll try to start CARLA anyway - sometimes the port check is wrong

        # Construct command with appropriate flags
        cmd = [self.carla_path]

        # Use low quality settings to reduce memory usage
        if self.low_quality:
            cmd.append("-quality-level=Low")

        # Set specific port
        cmd.append(f"-carla-port={self.port}")

        # Use OpenGL for more stability (Linux only)
        if platform.system() != 'Windows':
            cmd.append("-opengl")

        # Add the -nosound flag to reduce resource usage
        cmd.append("-nosound")

        logger.info(f"Starting CARLA server: {' '.join(cmd)}")

        # Maximum number of start attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Start the process with different error handling
                logger.info(f"Starting CARLA - attempt {attempt + 1}/{max_attempts}")

                try:
                    if platform.system() == 'Windows':
                        self.process = subprocess.Popen(cmd,
                                                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                    else:
                        self.process = subprocess.Popen(cmd,
                                                        preexec_fn=os.setsid)
                except Exception as e:
                    logger.error(f"Failed to start CARLA process: {e}")
                    time.sleep(10)
                    continue

                # Wait for server to start
                logger.info(
                    f"CARLA process started, waiting for initialization (attempt {attempt + 1}/{max_attempts})...")

                # Give more time for startup and initialization
                startup_time = 25  # Longer startup time
                conn_attempts = 12  # More connection attempts

                # Initial wait before trying to connect
                time.sleep(startup_time)

                # Try connecting multiple times
                for i in range(conn_attempts):
                    try:
                        client = carla.Client('localhost', self.port)
                        client.set_timeout(10.0)  # Longer timeout for connection checks
                        version = client.get_server_version()
                        logger.info(f"Connected to CARLA server version {version}")

                        # Add additional verification that world is accessible
                        try:
                            world = client.get_world()
                            logger.info(f"Successfully connected to CARLA world: {world.get_map().name}")
                            return True
                        except Exception as world_e:
                            logger.warning(f"Connected to CARLA but couldn't access world: {world_e}")
                            if i < conn_attempts - 1:
                                time.sleep(5)
                                continue
                            else:
                                break
                    except Exception as e:
                        if i < conn_attempts - 1:  # Not the last attempt
                            logger.warning(f"Connection attempt {i + 1}/{conn_attempts} failed: {e}")
                            time.sleep(5)  # Longer wait between attempts
                        else:
                            logger.error(f"Failed to connect after {conn_attempts} attempts")
                            # If all connection attempts failed, kill and retry
                            self.stop_carla_server()
                            time.sleep(10)  # Longer wait before retry
                            break

            except Exception as e:
                logger.error(f"Error during CARLA server start: {e}")
                logger.error(traceback.format_exc())  # Add stack trace
                if self.process:
                    client.set_timeout(20.0)
                    self.stop_carla_server()
                time.sleep(10)  # Longer wait before retry

        logger.error(f"Failed to start CARLA server after {max_attempts} attempts")
        return False

    def stop_carla_server(self):
        """Stop the CARLA server"""
        if self.process:
            logger.info("Stopping CARLA server...")
            try:
                if platform.system() == 'Windows':
                    try:
                        subprocess.call(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                    except Exception as e:
                        logger.error(f"Error using taskkill: {e}")
                        # Fallback - try to terminate directly
                        self.process.terminate()
                else:
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                    except Exception as e:
                        logger.error(f"Error killing process group: {e}")
                        # Fallback - try to terminate directly
                        self.process.terminate()
            except Exception as e:
                logger.error(f"Error stopping CARLA server: {e}")

            self.process = None
            time.sleep(5)  # Wait longer after stopping

        # Make sure all processes are gone
        self.kill_all_carla_processes()
        time.sleep(5)  # Additional wait after cleanup


def run_training_with_recovery(episodes=100, checkpoint_path="carla_checkpoint.pth", restart_frequency=5):
    """Run training with automatic CARLA server restart and crash recovery"""

    # Import the training module - do this inside the function to ensure fresh module state
    logger.info("Importing training module...")
    try:
        from carla_rl_agent import train_agent
    except Exception as e:
        logger.error(f"Failed to import training module: {e}")
        return

    # Create server manager
    server_manager = CarlaServerManager()

    # Load checkpoint if it exists
    start_episode = 0
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Use weights_only=False to fix the PyTorch 2.6 compatibility issue
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            start_episode = checkpoint.get('episode', 0) + 1
            logger.info(f"Resuming from episode {start_episode}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting from episode 0")

    # Initialize episode counter
    current_episode = start_episode

    while current_episode < episodes:
        # Add info about memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Current memory usage: {memory_mb:.2f} MB")

        # Calculate number of episodes to run before restart
        episodes_to_run = min(restart_frequency, episodes - current_episode)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting CARLA for episodes {current_episode + 1}-{current_episode + episodes_to_run}/{episodes}")
        logger.info(f"{'=' * 50}\n")

        # Start CARLA server - add retry logic here too
        max_start_retries = 3
        carla_started = False

        for retry in range(max_start_retries):
            try:
                logger.info(f"Starting CARLA server (attempt {retry + 1}/{max_start_retries})...")
                if server_manager.start_carla_server():
                    carla_started = True
                    break
                else:
                    logger.error(f"Failed to start CARLA (attempt {retry + 1}/{max_start_retries})")
                    time.sleep(15)  # Wait longer before retry
            except Exception as e:
                logger.error(f"Error starting CARLA: {e}")
                logger.error(traceback.format_exc())  # Add stack trace
                time.sleep(15)  # Wait longer before retry

        if not carla_started:
            logger.error("All attempts to start CARLA failed. Waiting before next overall attempt.")
            time.sleep(30)  # Long wait before trying again
            continue

        try:
            # Connect to CARLA with a higher timeout
            client = carla.Client('localhost', 2000)
            client.set_timeout(20.0)

            # Get the world
            world = client.get_world()

            logger.info(f"Running training for episodes {current_episode + 1}-{current_episode + episodes_to_run}")

            # Run training for a batch of episodes
            try:
                train_agent(episodes=episodes_to_run, start_episode=current_episode)
                # Update progress if training completed successfully
                current_episode += episodes_to_run
                logger.info(f"Successfully completed batch. Progress: {current_episode}/{episodes} episodes")
            except Exception as e:
                logger.error(f"Error in training function: {e}")
                logger.error(traceback.format_exc())  # Add stack trace

                # Check if we made any progress by looking at the checkpoint
                if os.path.exists(checkpoint_path):
                    try:
                        new_checkpoint = torch.load(checkpoint_path, weights_only=False)
                        new_episode = new_checkpoint.get('episode', 0) + 1
                        if new_episode > current_episode:
                            logger.info(f"Checkpoint shows progress to episode {new_episode}")
                            current_episode = new_episode
                    except Exception as ce:
                        logger.error(f"Error checking checkpoint: {ce}")

                logger.info("Proceeding to next batch after error")

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            break
        except Exception as e:
            logger.error(f"\nError during CARLA setup: {e}")
            logger.error(traceback.format_exc())  # Add stack trace
        finally:
            logger.info("Batch finished or interrupted - cleaning up...")
            try:
                # Stop CARLA server
                del client
                server_manager.stop_carla_server()
                logger.info("CARLA server stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping CARLA server: {e}")

            # Wait longer before next start
            wait_time = 20
            logger.info(f"Waiting {wait_time} seconds before next batch...")
            time.sleep(wait_time)
            logger.info("Continuing to next batch")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CARLA Training with Automatic Restart')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Total number of episodes to train')
    parser.add_argument('--restart-freq', type=int, default=5,
                        help='Restart CARLA server every N episodes')
    parser.add_argument('--checkpoint', type=str, default="carla_checkpoint.pth",
                        help='Path to checkpoint file')

    args = parser.parse_args()

    try:
        run_training_with_recovery(
            episodes=args.episodes,
            checkpoint_path=args.checkpoint,
            restart_frequency=args.restart_freq
        )
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}", exc_info=True)