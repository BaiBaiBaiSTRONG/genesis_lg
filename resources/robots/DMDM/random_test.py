import mujoco
import mujoco.viewer
import numpy as np
import time

class Go2JointTester:
    def __init__(self, scene_path):
        """Initialize MuJoCo model and data"""
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        # Define IMU sensor positions and initialize sensor data structure
        self.imu_positions = {
            'base': [-0.02557, 0, 0.04232],
            'FL': [0.1934, 0.0465, 0.04232],    # Front Left IMU
            'FR': [0.1934, -0.0465, 0.04232],   # Front Right IMU
            'RL': [-0.1934, 0.0465, 0.04232],   # Rear Left IMU
            'RR': [-0.1934, -0.0465, 0.04232],  # Rear Right IMU
            'ball': [0, 0, 0]                    # Ball IMU
        }
        
        # Initialize IMU sensor data structure
        self.imu_data = {name: {
            'accelerometer': np.zeros(3),  # Linear acceleration (x, y, z)
            'gyroscope': np.zeros(3),      # Angular velocity (roll, pitch, yaw)
            'position': np.zeros(3),       # Position (x, y, z)
            'orientation': np.zeros(3)      # Orientation (roll, pitch, yaw)
        } for name in self.imu_positions.keys()}
        
        # Create site objects and sensors for IMUs
        self._setup_imu_sites()
        
        # Print detailed model information
        print(f"\nModel Information:")
        print(f"qpos dimension: {self.data.qpos.shape}")
        print(f"qpos initial value: {self.data.qpos}")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Degrees of freedom: {self.model.nv}")
        
        # Analyze qpos dimension and set correct home position
        self.home_pos = np.zeros_like(self.data.qpos)
        
        # Parse initial position from keyframe string
        keyframe_str = "0 0 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0 0 0 0.45 0 0 0 0"
        keyframe_values = [float(x) for x in keyframe_str.split()]
        print(f"\nNumber of keyframe values: {len(keyframe_values)}")
        print(f"Keyframe values: {keyframe_values}")
        
        # Copy keyframe values to home_pos
        self.home_pos[:len(keyframe_values)] = keyframe_values
        
        print(f"\nHome position dimension: {self.home_pos.shape}")
        print(f"Home position values: {self.home_pos}")
        
        # Store joint information
        self.joint_info = self._get_joint_info()
        print(f"\nFound {len(self.joint_info)} joints")
        
        # Initialize control parameters
        self.paused = False
        self.show_wireframe = False
        self.random_range = 0.5
        self.smooth_factor = 0.02
        
        # Add state variables for generating smooth signals
        self.time = 0
        self.last_ctrl = np.zeros(self.model.nu)
        self.phase_offsets = np.random.uniform(0, 2 * np.pi, self.model.nu)
        self.frequencies = np.random.uniform(0.1, 0.3, self.model.nu)
        self.amplitudes = np.random.uniform(3, 6, self.model.nu)
        
        # Add timestep control
        self.physics_timestep = 0.002  # Physics simulation timestep
        self.render_fps = 60  # Desired render framerate
        
        # Add touch sensor data structure
        self.touch_sensors = {
            'ball_contact': None  # Will store ball contact force data
        }
        
        # Initialize sensors
        self._setup_sensors()
    
    def _get_joint_info(self):
        """Get information for all joints"""
        joint_info = []
        
        for i in range(self.model.njnt):
            range_min = self.model.jnt_range[i][0]
            range_max = self.model.jnt_range[i][1]
            
            try:
                leg_prefixes = ['FL', 'FR', 'RL', 'RR']
                joint_types = ['hip', 'thigh', 'calf']
                
                if i < 12:  # Leg joints
                    leg_idx = i // 3
                    joint_idx = i % 3
                    name = f"{leg_prefixes[leg_idx]}_{joint_types[joint_idx]}_joint"
                elif i == 12:
                    name = "motor_rotation"
                elif i == 13:
                    name = "rod_joint"
                else:
                    name = f"joint_{i}"
                
                joint_info.append({
                    'name': name,
                    'id': i,
                    'range_min': range_min,
                    'range_max': range_max
                })
                print(f"Joint {name}: range [{range_min:.2f}, {range_max:.2f}]")
                
            except Exception as e:
                print(f"Warning: Error processing joint {i}: {str(e)}")
                continue
        
        return joint_info
    
    def key_callback(self, keycode):
        """Handle keyboard input"""
        if chr(keycode) == ' ':
            self.paused = not self.paused
            print("Simulation {}".format("paused" if self.paused else "resumed"))
        elif chr(keycode) == 'w':
            self.show_wireframe = not self.show_wireframe
            print("Wireframe mode {}".format("enabled" if self.show_wireframe else "disabled"))
        elif chr(keycode) == 'r':
            self.reset_state()
            print("Reset to home position")
        elif chr(keycode) == 'q':
            return True
        return False

    def reset_state(self):
        """Reset state to initial position"""
        # Set positions one by one to avoid dimension mismatch
        for i in range(min(len(self.home_pos), len(self.data.qpos))):
            self.data.qpos[i] = self.home_pos[i]
            
        # Reset all velocities
        self.data.qvel[:] = 0
        
        # Update physics
        mujoco.mj_forward(self.model, self.data)
        
        # Reset signal generator state
        self.time = 0
        self.last_ctrl = np.zeros(self.model.nu)
        self.phase_offsets = np.random.uniform(0, 2 * np.pi, self.model.nu)
    
    def generate_random_control(self):
        """Generate smooth periodic control signals"""
        ctrl = np.zeros(self.model.nu)
        
        # Generate sine signals for each control dimension
        for i in range(self.model.nu):
            # Base sine signal
            base_signal = self.amplitudes[i] * np.sin(
                self.frequencies[i] * self.time + self.phase_offsets[i]
            )
            
            # Add small random perturbation
            noise = np.random.uniform(-0.5, 0.5)
            
            # Combine signals
            ctrl[i] = base_signal + noise
        
        # Use smooth factor for transition
        ctrl = self.last_ctrl + self.smooth_factor * (ctrl - self.last_ctrl)
        self.last_ctrl = ctrl.copy()
        
        # Update time
        self.time += 0.01
        
        return ctrl

    def run_test(self, duration=30, interval=1.0):
        """Run test"""
        print("\nKey Instructions:")
        print("Space: Pause/Resume")
        print("W: Toggle wireframe mode")
        print("R: Reset position")
        print("Q: Quit")
        print("Ctrl+C: Force quit")
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback) as viewer:
                # Set camera view
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 45.0
                viewer.cam.elevation = -20.0
                
                # Reset to initial state first
                self.reset_state()
                
                # Pause to let user observe initial state
                print("\nInitial state set, press space to start test...")
                self.paused = True
                
                start_time = time.time()
                last_print_time = start_time
                last_update_time = start_time
                
                while viewer.is_running():
                    current_time = time.time()
                    
                    if not self.paused:
                        # Calculate physics steps to catch up
                        steps_to_catch_up = int((current_time - last_update_time) / self.physics_timestep)
                        
                        for _ in range(steps_to_catch_up):
                            # Generate and apply random control signals
                            ctrl = self.generate_random_control()
                            self.data.ctrl[:] = ctrl
                            
                            # Step physics simulation
                            mujoco.mj_step(self.model, self.data)
                        
                        last_update_time = current_time
                        
                        # Print current state and control signals
                        if current_time - last_print_time >= interval:
                            print(f"\nBase height: {self.data.qpos[2]:.3f}")
                            print(f"Control signals: {ctrl}")
                            
                            # Get and print IMU data
                            imu_data = self.get_imu_data()
                            print("\nIMU Sensor Data:")
                            for imu_name, data in imu_data.items():
                                print(f"\nIMU {imu_name}:")
                                print(f"  Position: [{data['position'][0]:.3f}, {data['position'][1]:.3f}, {data['position'][2]:.3f}]")
                                print(f"  Orientation (rad): [{data['orientation'][0]:.3f}, {data['orientation'][1]:.3f}, {data['orientation'][2]:.3f}]")
                                print(f"  Acceleration: [{data['accelerometer'][0]:.3f}, {data['accelerometer'][1]:.3f}, {data['accelerometer'][2]:.3f}]")
                                print(f"  Angular Velocity: [{data['gyroscope'][0]:.3f}, {data['gyroscope'][1]:.3f}, {data['gyroscope'][2]:.3f}]")
                            
                            # Get and print touch sensor data
                            touch_data = self.get_touch_sensor_data()
                            print("\nTouch Sensor Data:")
                            for sensor_name, data in touch_data.items():
                                print(f"{sensor_name}:")
                                print(f"  Raw Force: {data['raw_force']:.6f} N")
                                print(f"  Magnitude: {data['magnitude']:.6f} N")
                                
                                # Print contact state
                                if data['contact_states']['no_contact']:
                                    contact_state = "No Contact"
                                elif data['contact_states']['light_contact']:
                                    contact_state = "Light Contact"
                                else:
                                    contact_state = "Firm Contact"
                                    
                                print(f"  Contact State: {contact_state}")
                            
                            last_print_time = current_time
                    
                    # Update view
                    if self.show_wireframe:
                        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
                    else:
                        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
                    
                    viewer.sync()
                    
                    # Control render framerate
                    time_to_next_frame = 1.0/self.render_fps - (time.time() - current_time)
                    if time_to_next_frame > 0:
                        time.sleep(time_to_next_frame)
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            print("Test ended")

    def _setup_imu_sites(self):
        """Setup IMU site objects and sensors in the model"""
        for imu_name, pos in self.imu_positions.items():
            site_name = f"imu_{imu_name}"
            try:
                # Get or create site
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id >= 0:
                    self.model.site_pos[site_id] = pos
                    
                    # Get accelerometer sensor
                    accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{site_name}_accel")
                    if accel_id < 0:
                        print(f"Warning: Accelerometer sensor for {site_name} not found in model")
                        
                    # Get gyroscope sensor
                    gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{site_name}_gyro")
                    if gyro_id < 0:
                        print(f"Warning: Gyroscope sensor for {site_name} not found in model")
                        
            except Exception as e:
                print(f"Warning: Could not setup IMU site and sensors {site_name}: {str(e)}")
    
    def get_imu_positions(self):
        """Get current world positions of all IMU sensors"""
        imu_world_pos = {}
        for imu_name in self.imu_positions.keys():
            site_name = f"imu_{imu_name}"
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id >= 0:
                    # Get the world position of the IMU site
                    pos = self.data.site_xpos[site_id]
                    imu_world_pos[imu_name] = pos
            except Exception as e:
                print(f"Warning: Could not get position for IMU {site_name}: {str(e)}")
        return imu_world_pos

    def get_imu_data(self):
        """Get comprehensive IMU sensor data including position, orientation, acceleration, and angular velocity"""
        for imu_name in self.imu_positions.keys():
            site_name = f"imu_{imu_name}"
            try:
                # Get site ID
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                if site_id >= 0:
                    # Get position
                    self.imu_data[imu_name]['position'] = self.data.site_xpos[site_id].copy()
                    
                    # Get orientation (from rotation matrix to euler angles)
                    rot_matrix = self.data.site_xmat[site_id].reshape(3, 3)
                    orientation = np.zeros(3)
                    orientation[0] = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])  # roll
                    orientation[1] = np.arcsin(-rot_matrix[2, 0])                    # pitch
                    orientation[2] = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])  # yaw
                    self.imu_data[imu_name]['orientation'] = orientation
                    
                    # Get accelerometer data
                    accel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{site_name}_accel")
                    if accel_id >= 0:
                        self.imu_data[imu_name]['accelerometer'] = self.data.sensordata[accel_id:accel_id+3]
                    
                    # Get gyroscope data
                    gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{site_name}_gyro")
                    if gyro_id >= 0:
                        self.imu_data[imu_name]['gyroscope'] = self.data.sensordata[gyro_id:gyro_id+3]
                    
            except Exception as e:
                print(f"Warning: Could not get IMU data for {site_name}: {str(e)}")
                
        return self.imu_data

    def _setup_sensors(self):
        """Setup all sensors including IMUs and touch sensors"""
        self._setup_imu_sites()
        
        # Setup touch sensors
        try:
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_contact_sensor")
            if sensor_id < 0:
                print("Warning: Ball contact sensor not found in model")
        except Exception as e:
            print(f"Warning: Could not setup touch sensor: {str(e)}")
    
    def get_touch_sensor_data(self):
        """Get touch sensor data"""
        try:
            # Get ball contact sensor data
            ball_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ball_contact_sensor")
            if ball_sensor_id >= 0:
                force = self.data.sensordata[ball_sensor_id]
                force_magnitude = abs(force)
                self.touch_sensors['ball_contact'] = {
                    'raw_force': force,
                    'magnitude': force_magnitude,
                    'contact_states': {
                        'no_contact': force_magnitude < 0.001,
                        'light_contact': 0.001 <= force_magnitude < 0.01,
                        'firm_contact': force_magnitude >= 0.01
                    },
                    'in_contact': force_magnitude >= 0.001
                }
                
        except Exception as e:
            print(f"Warning: Could not get touch sensor data: {str(e)}")
        
        return self.touch_sensors

def main():
    scene_path = "scene.xml"
    
    try:
        tester = Go2JointTester(scene_path)
        tester.run_test(duration=30, interval=1.0)
        
    except Exception as e:
        print(f"Error occurred during test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()