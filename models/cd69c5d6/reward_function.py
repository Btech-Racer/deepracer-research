import math

import numpy as np


def reward_function(params):
    """
    Head-to-head racing reward function

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

    Head-to-Head Parameters:
    - closest_objects (list_int): Zero-based indices of the two closest objects to agent's position
    - objects_distance (list_float) (meters) [0 to track_length]: List of objects' distances in meters from starting line
    - objects_heading (list_float) (degrees) [-180 to 180]: List of objects' headings in degrees
    - objects_left_of_center (list_boolean): List of Boolean flags indicating if objects are left of center
    - objects_location (list_tuple): List of object locations as (x,y) coordinates
    - objects_speed (list_float) (m/s): List of objects' speeds in meters per second
    """

    all_wheels_on_track = params["all_wheels_on_track"]
    distance_from_center = params["distance_from_center"]
    is_left_of_center = params["is_left_of_center"]
    heading = params["heading"]
    progress = params["progress"]
    speed = params["speed"]
    steering_angle = params["steering_angle"]
    steps = params["steps"]
    track_length = params["track_length"]
    track_width = params["track_width"]
    x = params["x"]
    y = params["y"]

    closest_waypoints = params["closest_waypoints"]
    waypoints = params["waypoints"]

    is_crashed = params["is_crashed"]
    is_offtrack = params["is_offtrack"]
    is_reversed = params["is_reversed"]

    params["closest_objects"]
    objects_distance = params["objects_distance"]
    params["objects_heading"]
    objects_left_of_center = params["objects_left_of_center"]
    objects_location = params["objects_location"]
    objects_speed = params["objects_speed"]

    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(15.0 * -1)

    if is_reversed:
        return float(1e-3)

    reward = 3.5

    speed_reward = 0.0

    if speed >= 4.0:
        speed_factor = min(speed / 5.0, 1.0)
        speed_reward = 2.5 * (speed_factor**2.5)

        if speed >= 4.5:
            speed_reward += 3.5

    elif speed >= 2.0:
        # Moderate reward for acceptable racing speed
        speed_factor = (speed - 2.0) / (4.0 - 2.0)
        speed_reward = speed_factor * 2.0
    else:
        speed_reward = -2.0  # Harsher than centerline scenarios

    reward += speed_reward

    object_reward = 0.0
    closest_object_distance = float("inf")
    objects_ahead = []
    objects_behind = []
    bot_speeds = []

    if objects_location and len(objects_location) > 0:
        objects_array = np.array(objects_location)
        agent_pos = np.array([x, y])
        distances = np.linalg.norm(objects_array - agent_pos, axis=1)

        closest_obj_index = np.argmin(distances)
        closest_object_distance = distances[closest_obj_index]

        current_progress_distance = progress * track_length / 100

        for i, obj_location in enumerate(objects_location):
            obj_progress = objects_distance[i] if i < len(objects_distance) else 0
            obj_speed = objects_speed[i] if i < len(objects_speed) else 0
            bot_speeds.append(obj_speed)

            if obj_progress > current_progress_distance + 5:  # 5m ahead threshold
                objects_ahead.append(i)
            elif obj_progress < current_progress_distance - 5:  # 5m behind threshold
                objects_behind.append(i)

        if closest_object_distance >= 1.0:
            # RACING CLEAR TRACK: Maximum speed rewards
            object_reward = 3.5 * 2

            # Speed dominance bonus when clear track
            if speed >= 4.0:
                object_reward += 4.0

        elif closest_object_distance >= 0.6:
            object_reward = 3.5

            if objects_ahead and speed > 2.0:
                # Calculate speed advantage
                avg_bot_speed = np.mean([objects_speed[i] for i in objects_ahead if i < len(objects_speed)])
                if speed > avg_bot_speed + 0.8:
                    object_reward += 8.0

                    closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
                    if closest_ahead_idx < len(objects_left_of_center):
                        bot_left = objects_left_of_center[closest_ahead_idx]
                        # Reward being on opposite side for overtake
                        if bot_left != is_left_of_center:
                            object_reward += 3.0

            if objects_ahead and closest_object_distance > 0.6 * 0.8:
                object_reward += 2.0

        elif closest_object_distance >= 0.3:
            object_reward = 1.0

            if speed <= 4.0 * 0.8:
                object_reward += 0.5  # Reward appropriate speed control

        else:
            object_reward = -15.0

    reward += object_reward
    racing_line_reward = 0.0

    if len(waypoints) > closest_waypoints[1] + 4:
        try:
            current_point = waypoints[closest_waypoints[0]]
            next_point = waypoints[closest_waypoints[1]]
            future_point = waypoints[closest_waypoints[1] + 4]

            # Calculate track direction
            track_direction = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
            track_direction = math.degrees(track_direction)

            # Calculate future track direction for racing line
            future_direction = math.atan2(future_point[1] - next_point[1], future_point[0] - next_point[0])
            future_direction = math.degrees(future_direction)

            # Heading alignment with track direction - more flexible than centerline
            direction_diff = abs(heading - track_direction)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            if direction_diff <= 20.0:
                alignment_bonus = 1.0 - (direction_diff / 20.0)
                racing_line_reward += 2.5 * alignment_bonus

            # RACING STRATEGY: Adaptive speed based on curvature and competition
            direction_change = abs(future_direction - track_direction)
            if direction_change > 180:
                direction_change = 360 - direction_change

            # Racing speed optimization (more aggressive than centerline)
            if direction_change > 45:  # Sharp turn ahead
                optimal_speed = 2.0 + 1.0  # Higher than centerline
                # Reward late braking in racing
                if speed >= optimal_speed and closest_object_distance > 0.6:
                    racing_line_reward += 1.0
            elif direction_change > 20:  # Medium turn
                optimal_speed = 4.0 - 0.5
                if speed >= optimal_speed:
                    racing_line_reward += 1.5
            else:  # Straight or slight turn - FULL ATTACK MODE
                optimal_speed = 5.0
                if speed >= optimal_speed * 0.9:
                    racing_line_reward += 3.5

            # Tactical positioning bonus based on objects
            if objects_ahead and closest_object_distance < 1.0:
                # Reward positioning for overtake opportunities
                if distance_from_center > track_width * 0.3:  # Outside line setup
                    racing_line_reward += 1.5
                elif distance_from_center < track_width * 0.15:  # Inside line defense
                    racing_line_reward += 2.0

        except (IndexError, ZeroDivisionError):
            pass

    reward += racing_line_reward

    position_reward = 0.0

    centerline_threshold = 0.6 * track_width

    if distance_from_center <= centerline_threshold * 0.5:
        position_reward = 2.5
    elif distance_from_center <= centerline_threshold:
        position_factor = 1.0 - (distance_from_center / centerline_threshold)
        position_reward = position_factor * 2.5 * 0.7

        if closest_object_distance < 1.0:
            position_reward += 3.0 * 0.5
    else:
        if closest_object_distance < 0.6:
            position_reward = -0.1
        else:
            position_reward = -0.5

    reward += position_reward
    steering_reward = 0.0
    abs_steering = abs(steering_angle)

    if abs_steering <= 15.0:
        steering_reward = 1.0
    elif abs_steering <= 25.0:
        if closest_object_distance < 1.0:
            steering_reward = 1.0
        else:
            steering_reward = 0.5
    else:
        if closest_object_distance < 0.6:
            # Emergency/tactical steering allowed
            steering_reward = 0.2
        else:
            # Excessive steering with no reason
            steering_reward = -0.3

    reward += steering_reward
    if progress >= 99.0:
        reward += 100.0  # Double centerline completion bonus
    elif progress >= 90.0:
        reward += 25.0  # Major progress bonus
    elif progress >= 75.0:
        reward += 10.0  # Good progress bonus
    elif progress >= 50.0:
        reward += 5.0  # Moderate progress bonus

    if steps > 0:
        speed_efficiency = speed * (progress / 100.0) / (steps / 100.0)
        reward += speed_efficiency * 15.0  # Higher than centerline scenarios

    if objects_behind:
        position_advantage = len(objects_behind) / max(len(objects_location), 1)
        reward += 4.0 * position_advantage

    if bot_speeds and speed > max(bot_speeds) + 0.8:
        speed_dominance = (speed - max(bot_speeds)) / 5.0
        reward += 2.5 * speed_dominance
    if objects_ahead and closest_object_distance < 1.0:
        if speed >= 4.5:
            reward += 8.0

            if distance_from_center > track_width * 0.35:  # Outside overtake
                reward += 1.5
            elif distance_from_center < track_width * 0.15:  # Inside overtake
                reward += 2.0

    if objects_behind and bot_speeds:
        max_bot_speed = max([objects_speed[i] for i in objects_behind if i < len(objects_speed)])
        if max_bot_speed > speed * 0.9:
            if distance_from_center <= centerline_threshold * 0.6:
                reward += 2.0

    if closest_object_distance > 1.0 * 1.5:
        if speed >= 4.0:
            clear_track_bonus = (speed / 5.0) ** 2
            reward += 3.5 * clear_track_bonus

    if objects_location and len(objects_location) > 0:
        for i, obj_left in enumerate(objects_left_of_center):
            if i < len(objects_location):
                obj_distance = math.sqrt((x - objects_location[i][0]) ** 2 + (y - objects_location[i][1]) ** 2)

                if 0.6 <= obj_distance <= 1.0:
                    if obj_left != is_left_of_center:
                        if speed >= 4.0 * 0.8:
                            reward += 3.0

                        if i < len(objects_speed) and speed > objects_speed[i] + 0.8:
                            reward += 8.0 * 0.5

    return float(reward)
