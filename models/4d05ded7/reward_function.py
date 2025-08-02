# Place import statement outside of function (supported libraries: math, random, numpy, scipy, and shapely)
import math


def reward_function(params):
    """
    Reward function for centerline following

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

    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(1e-3)

    if is_reversed:
        return float(1e-3)

    reward_components = {"position": 0.0, "speed": 0.0, "steering": 0.0, "direction": 0.0, "progress": 0.0, "efficiency": 0.0}

    marker_excellent = 0.08 * track_width
    marker_good = 0.2 * track_width
    marker_acceptable = 0.4 * track_width
    marker_poor = 0.6 * track_width

    if distance_from_center <= marker_excellent:
        reward_components["position"] = 2.0
    elif distance_from_center <= marker_good:
        position_factor = 1.0 - (distance_from_center - marker_excellent) / (marker_good - marker_excellent)
        reward_components["position"] = 1.0 + (2.0 - 1.0) * position_factor
    elif distance_from_center <= marker_acceptable:
        position_factor = 1.0 - (distance_from_center - marker_good) / (marker_acceptable - marker_good)
        reward_components["position"] = 0.3 + (1.0 - 0.3) * position_factor
    elif distance_from_center <= marker_poor:
        position_factor = 1.0 - (distance_from_center - marker_acceptable) / (marker_poor - marker_acceptable)
        reward_components["position"] = 0.1 + (0.3 - 0.1) * position_factor
    else:
        reward_components["position"] = 1e-3

    if speed >= 3.0:
        speed_factor = min(speed / 4.0, 1.0)
        reward_components["speed"] = 1.5 * (speed_factor**1.2)
    elif speed >= 1.2:
        speed_factor = (speed - 1.2) / (3.0 - 1.2)
        reward_components["speed"] = 1.5 * speed_factor * 0.7
    else:
        reward_components["speed"] = -0.5  # Penalty for being too slow

    # Enhanced steering smoothness with graduated rewards
    abs_steering = abs(steering_angle)
    if abs_steering <= 12.0:
        reward_components["steering"] = 1.0
    elif abs_steering <= 20.0:
        steering_factor = 1.0 - (abs_steering - 12.0) / (20.0 - 12.0)
        reward_components["steering"] = 0.5 + (1.0 - 0.5) * steering_factor
    elif abs_steering <= 28.0:
        reward_components["steering"] = 0.5 * 0.2
    else:
        reward_components["steering"] = -0.3

    # Waypoint direction alignment
    if len(waypoints) > closest_waypoints[1]:
        try:
            next_point = waypoints[closest_waypoints[1]]
            prev_point = waypoints[closest_waypoints[0]]

            track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
            track_direction = math.degrees(track_direction)

            direction_diff = abs(heading - track_direction)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            if direction_diff <= 10.0:
                alignment_factor = 1.0 - (direction_diff / 10.0)
                reward_components["direction"] = 0.8 * alignment_factor
        except (IndexError, ZeroDivisionError):
            pass

    # Progress rewards with major completion bonus
    if progress >= 99.0:
        reward_components["progress"] = 15.0
    elif progress >= 90.0:
        reward_components["progress"] = 15.0 * 0.3
    elif progress >= 75.0:
        reward_components["progress"] = 15.0 * 0.15

    # Efficiency rewards
    if steps > 0:
        # Reward speed and progress efficiency
        efficiency = (progress / 100.0) * (speed / 4.0) / (steps / 100.0)
        reward_components["efficiency"] = efficiency * 2.0

        # Slight penalty for taking too many steps
        if steps > progress * 1.5:  # More than 1.5 steps per progress percent
            reward_components["efficiency"] -= 0.02 * (steps - progress * 1.5)

    # Weighted final reward calculation
    final_reward = (
        reward_components["position"]
        + reward_components["speed"] * 2.0
        + reward_components["steering"]
        + reward_components["direction"]
        + reward_components["progress"]
        + reward_components["efficiency"]
    )

    reward = final_reward

    return float(reward)
