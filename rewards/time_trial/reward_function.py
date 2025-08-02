# Place import statement outside of function (supported libraries: math, random, numpy, scipy, and shapely)
import math


def reward_function(params):
    """
    Available parameters from AWS DeepRacer:

    Position Parameters:
    - x (float) (meters): Agent's x-coordinate in meters
    - y (float) (meters): Agent's y-coordinate in meters
    - distance_from_center (float) (meters) [0 to track_width/2]: Distance in meters from the track center
    - is_left_of_center (boolean): Flag to indicate if the agent is on the left side of track center

    Movement Parameters:
    - speed (float) (m/s) [0 to max_speed]: Agent's speed in meters per second
    - steering_angle (float) (degrees) [-30 to 30]: Agent's steering angle in degrees
    - heading (float) (degrees) [-180 to 180]: Agent's yaw in degrees

    Track Parameters:
    - track_width (float) (meters): Width of the track
    - track_length (float) (meters): Track length in meters
    - waypoints (list_tuple): List of (x,y) coordinates as milestones along the track center
    - closest_waypoints (list_int): Indices of the two nearest waypoints

    Progress Parameters:
    - progress (float) (percentage) [0 to 100]: Percentage of track completed
    - steps (int): Number of steps completed

    Status Parameters:
    - all_wheels_on_track (boolean): Flag to indicate if the agent is on the track
    - is_crashed (boolean): Boolean flag to indicate whether the agent has crashed
    - is_offtrack (boolean): Boolean flag to indicate whether the agent has gone off track
    - is_reversed (boolean): Flag to indicate if agent is driving clockwise (True) or counter-clockwise (False)
    """

    all_wheels_on_track = params["all_wheels_on_track"]
    distance_from_center = params["distance_from_center"]
    params["is_left_of_center"]
    heading = params["heading"]
    progress = params["progress"]
    speed = params["speed"]
    steering_angle = params["steering_angle"]
    steps = params["steps"]
    params["track_length"]
    track_width = params["track_width"]
    params["x"]
    params["y"]

    closest_waypoints = params["closest_waypoints"]
    waypoints = params["waypoints"]

    is_crashed = params["is_crashed"]
    is_offtrack = params["is_offtrack"]
    is_reversed = params["is_reversed"]

    # Safety checks
    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(1e-3)

    if is_reversed:
        return float(1e-3)

    # Initialize reward components for detailed tracking
    reward_components = {
        "centerline": 0.0,
        "speed": 0.0,
        "racing_line": 0.0,
        "steering": 0.0,
        "direction": 0.0,
        "progress": 0.0,
        "efficiency": 0.0,
    }

    # Calculate centerline precision markers
    marker_excellent = 0.08 * track_width
    marker_good = 0.18 * track_width
    marker_acceptable = 0.3 * track_width
    marker_poor = 0.5 * track_width

    # Smooth centerline following with granular precision rewards
    if distance_from_center <= marker_excellent:
        reward_components["centerline"] = 3.5
    elif distance_from_center <= marker_good:
        position_factor = 1.0 - (distance_from_center - marker_excellent) / (marker_good - marker_excellent)
        reward_components["centerline"] = 2.5 + (3.5 - 2.5) * position_factor
    elif distance_from_center <= marker_acceptable:
        position_factor = 1.0 - (distance_from_center - marker_good) / (marker_acceptable - marker_good)
        reward_components["centerline"] = 1.5 + (2.5 - 1.5) * position_factor
    elif distance_from_center <= marker_poor:
        position_factor = 1.0 - (distance_from_center - marker_acceptable) / (marker_poor - marker_acceptable)
        reward_components["centerline"] = 0.5 + (1.5 - 0.5) * position_factor
    else:
        # Significant penalty for poor positioning in time trials
        reward_components["centerline"] = -1.0

    # Enhanced speed rewards with exponential scaling for time trials
    if speed >= 3.8:
        speed_factor = min(speed / 4.0, 1.0)
        # Exponential reward for high speeds (prevents slow driving exploitation)
        reward_components["speed"] = 5.0 * (speed_factor**1.8)
    elif speed >= 2.0:
        speed_factor = (speed - 2.0) / (3.8 - 2.0)
        reward_components["speed"] = 3.5 * (speed_factor**1.2)
    else:
        # Strong penalty for being too slow in time trials
        reward_components["speed"] = -2.0

    optimal_line_distance = 0.12 * track_width
    centerline_distance = 0.25 * track_width

    if distance_from_center <= optimal_line_distance:
        # Maximum reward for perfect racing line
        line_factor = 1.0 - (distance_from_center / optimal_line_distance)
        reward_components["racing_line"] = 4.0 * (line_factor**0.8)
    elif distance_from_center <= centerline_distance:
        # Moderate reward for reasonable positioning
        line_factor = 1.0 - (distance_from_center / centerline_distance)
        reward_components["racing_line"] = 4.0 * line_factor * 0.6
    else:
        # Penalty for poor positioning
        reward_components["racing_line"] = -0.8

    if len(waypoints) > closest_waypoints[1]:
        try:
            # Advanced lookahead for better racing line prediction
            current_point = waypoints[closest_waypoints[0]]
            next_point = waypoints[closest_waypoints[1]]

            # Calculate track direction with improved precision
            track_direction = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
            track_direction = math.degrees(track_direction)

            # Direction alignment calculation
            direction_diff = abs(heading - track_direction)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            if direction_diff <= 8.0:
                alignment_factor = 1.0 - (direction_diff / 8.0)
                reward_components["direction"] = 2.0 * (alignment_factor**1.1)
            else:
                # Penalty for poor direction alignment in time trials
                reward_components["direction"] = -0.3

            # Enhanced curvature-based speed optimization
            if len(waypoints) > closest_waypoints[1] + 4:
                future_point = waypoints[closest_waypoints[1] + 4]

                future_direction = math.atan2(future_point[1] - next_point[1], future_point[0] - next_point[0])
                future_direction = math.degrees(future_direction)

                direction_change = abs(future_direction - track_direction)
                if direction_change > 180:
                    direction_change = 360 - direction_change

                # Smart speed adjustment for upcoming track features
                if direction_change > 50:  # Very sharp turn ahead
                    optimal_corner_speed = 2.0 + 0.8
                elif direction_change > 30:  # Sharp turn ahead
                    optimal_corner_speed = 2.0 + 1.2
                elif direction_change > 15:  # Medium turn
                    optimal_corner_speed = 3.8 - 0.3
                else:  # Straight or slight turn
                    optimal_corner_speed = 4.0

                # Reward speed matching track curvature
                speed_diff = abs(speed - optimal_corner_speed)
                if speed_diff <= 0.3:
                    reward_components["direction"] += 1.5 * 1.2
                elif speed_diff <= 0.6:
                    reward_components["direction"] += 1.5 * 0.6

        except (IndexError, ZeroDivisionError):
            pass

    abs_steering = abs(steering_angle)

    # Graduated steering rewards optimized for time trials
    if abs_steering <= 8.0:
        reward_components["steering"] = 2.5
    elif abs_steering <= 15.0:
        steering_factor = 1.0 - (abs_steering - 8.0) / (15.0 - 8.0)
        reward_components["steering"] = 1.5 + (2.5 - 1.5) * steering_factor
    elif abs_steering <= 25.0:
        # Allow some aggressive steering for racing, but with reduced reward
        reward_components["steering"] = 1.5 * 0.3
    else:
        reward_components["steering"] = -0.5

    if progress >= 99.0:
        reward_components["progress"] = 50.0
        # Additional time efficiency bonus for quickly completion
        if steps > 0:
            time_efficiency = min(1200.0 / steps, 20.0)  # Bonus for quickly lap times
            reward_components["progress"] += time_efficiency
    elif progress >= 90.0:
        reward_components["progress"] = 10.0 * 1.2
    elif progress >= 75.0:
        reward_components["progress"] = 10.0 * 0.8
    elif progress >= 50.0:
        reward_components["progress"] = 10.0 * 0.4
    elif progress >= 25.0:
        reward_components["progress"] = 10.0 * 0.2

    if steps > 0:
        # Multi-factor efficiency: progress, speed, and step optimization
        progress_efficiency = (progress / 100.0) / (steps / 100.0)
        speed_efficiency = speed / 4.0
        position_efficiency = max(0, 1.0 - (distance_from_center / (track_width * 0.5)))

        cbd_efficiency = progress_efficiency * speed_efficiency * (0.7 + 0.3 * position_efficiency)

        reward_components["efficiency"] = cbd_efficiency * 2.5

        # Enhanced step penalty for time trial optimization
        expected_steps = progress * 1.15  # Tighter step tolerance for time trials
        if steps > expected_steps:
            step_penalty = 0.015 * (steps - expected_steps)
            reward_components["efficiency"] -= step_penalty

    # Advanced consistency bonus for maintaining high performance
    consistency_factor = 1.0
    if speed >= 3.8 and distance_from_center <= marker_good and abs_steering <= 15.0:
        consistency_factor = 1.3

    total_reward = (
        reward_components["centerline"] * 2.5  # High weight on centerline precision
        + reward_components["speed"] * 3.0
        + reward_components["racing_line"] * 3.0
        + reward_components["direction"] * 1.5
        + reward_components["steering"] * 1.5
        + reward_components["progress"]
        + reward_components["efficiency"] * 3.0
    ) * consistency_factor

    total_reward = max(total_reward, 0.01)

    return float(total_reward)
