import time
from lane_line_detection import *
from traffic_sign_detection import detect_traffic_signs

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.previous_error = 0
        self.integral = 0

    def update(self, error):
        """Calculate the PID output."""
        p = self.kp * error
        self.integral += error * self.dt
        i = self.ki * self.integral
        derivative = (error - self.previous_error) / self.dt
        d = self.kd * derivative
        self.previous_error = error
        return p + i + d

class CarLogic:
    def __init__(self):
        self.last_sign_time = 0
        self.last_sign_detection = ''
        self.last_sign = ''
        self.turning_time = 0
        self.min_throttle = 0.2
        self.default_throttle = 0.35
        self.turn_duration = 5
        self.throttle = self.default_throttle
        self.steering_angle = 0
        self.max_turn_time = 10
        self.min_turn_time = 0.05
        self.stop_time = 4
        self.see_sign_time = 0
        self.max_see_sign_time = 1000
        self.stop = False

        self.pid_controller = PIDController(kp=0.015, ki=0.0002, kd=0.0001, dt=1/10)

    def calculate_control_signal(self, draw=None):
        """Calculate the control signal based on the detected lane lines."""
        self.left_point, self.right_point, self.have_left, self.have_right, self.len_line = calculate_left_right_point(self.img, draw=draw)

        im_center = self.img.shape[1] // 2
        if self.left_point != -1 and self.right_point != -1:
            middle_point = (self.right_point + self.left_point) // 2
            x_offset = im_center - middle_point
            self.steering_angle = -self.pid_controller.update(x_offset)

            # Adjust throttle based on steering angle
            self.throttle = self.min_throttle if abs(self.steering_angle) > 0.58 else self.default_throttle
        else:
            self.steering_angle = 0

    def detect_sign(self, signs):
        """Detect traffic signs and update the last detected sign."""
        for sign in signs:
            class_name, _, _, _, _ = sign
            if class_name in ['left', 'right', 'no_left', 'no_right', 'straight', 'stop']:
                self.last_sign_detection = class_name
                self.see_sign_time = time.time()

    def handle_sign_detection(self):
        """Handle behavior based on the last detected sign."""
        if self.steering_angle != 0 and self.last_sign_detection and not self.turning_time:
            self.throttle = self.min_throttle

        if self.last_sign_detection and not self.turning_time and (not self.have_left or not self.have_right):
            self.steering_angle = 0

        self.set_turning_time()


    def set_turning_time(self):
        """Set the turning time based on the last detected sign."""
        if not self.steering_angle and not self.turning_time:
            if self.last_sign_detection == 'right' and not self.have_right:
                self.start_turning('right')
            elif self.last_sign_detection == 'left' and not self.have_left:
                self.start_turning('left')
            elif self.last_sign_detection == 'no_right' and not self.have_left:
                self.start_turning('no_right')
            elif self.last_sign_detection == 'no_left' and not self.have_right:
                self.start_turning('no_left')
            elif self.last_sign_detection == 'straight':
                self.start_turning('straight')
            elif self.last_sign_detection == 'stop':
                print('stop')
                self.stop = True
                self.start_turning('stop', self.stop_time)

    def start_turning(self, direction, duration=None):
        """Start the turning process."""
        if duration is None: 
            self.turning_time = self.max_turn_time
        else:
            self.throttle = 0
            self.turning_time = duration
        self.last_sign_time = time.time()
        print(f"turn {direction}")

    def execute_turning(self):
        """Execute the turning process based on the sign and lane detection."""
        if self.turning_time and (0 <= (time.time() - self.last_sign_time) <= self.turning_time):
            self.perform_turn()
            if self.len_line == 2 and (time.time() - self.last_sign_time) >= self.min_turn_time:
                self.reset_turning_state()
        elif self.turning_time and (self.len_line == 2):
            self.reset_turning_state()


    def perform_turn(self):
        """Adjust throttle and steering angle during a turn."""
        if self.last_sign_detection == 'left':
            self.steering_angle = -1
        elif self.last_sign_detection == 'right':
            self.steering_angle = 1
        elif self.last_sign_detection == 'no_left':
            self.steering_angle = abs(self.steering_angle)
        elif self.last_sign_detection == 'no_right':
            self.steering_angle = -abs(self.steering_angle)
        elif self.last_sign_detection == 'straight':
            self.steering_angle = 0
        elif self.last_sign_detection == 'stop':
            self.throttle = 0
            self.steering_angle = 0

    def reset_turning_state(self):
        """Reset the state after completing a turn or stopping."""
        self.turning_time = 0
        self.last_sign = self.last_sign_detection
        self.last_sign_detection = ''
        self.see_sign_time = 0
        self.last_sign_time = 0
        print("turning state reseted")

    def reset_if_no_turn(self):
        """Reset the state if no turn was made after seeing a sign."""
        if self.last_sign_detection and not self.turning_time and (time.time() - self.see_sign_time) >= self.max_see_sign_time:
            self.reset_turning_state()
            print("turning state reseted")

    def decision_control(self, image, signs, draw=None):
        """Main control loop to process the image and signs."""
        self.img = image
        self.im_height, self.im_width = self.img.shape[:2]

        self.calculate_control_signal(draw=draw)
        self.detect_sign(signs)
        self.handle_sign_detection()
        self.execute_turning()
        self.reset_if_no_turn()

        if not self.have_left and not self.have_right and not self.last_sign_detection:
            self.steering_angle = 1
            self.throttle = 0.3

        if self.stop == True:
            self.throttle = 0
            print('car stopped')

        return self.throttle, self.steering_angle
