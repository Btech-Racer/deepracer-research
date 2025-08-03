# Place import statement outside of function (supported libraries: math, random, numpy, scipy, and shapely)
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
    objects_heading = params["objects_heading"]
    objects_left_of_center = params["objects_left_of_center"]
    objects_location = params["objects_location"]
    objects_speed = params["objects_speed"]

    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(20.0 * -1)

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
        speed_factor = (speed - 2.0) / (4.0 - 2.0)
        speed_reward = speed_factor * 2.0
    else:
        speed_reward = -2.0  # Harsher than centerline scenarios

    reward += speed_reward

    collision_prevention_penalty = 0.0
    if objects_location and len(objects_location) > 0:
        for i, obj_location in enumerate(objects_location):
            obj_distance = math.sqrt((x - obj_location[0]) ** 2 + (y - obj_location[1]) ** 2)

            # If very close, perform detailed collision analysis
            if obj_distance < 0.4 * 3:
                if i < len(objects_heading) and i < len(objects_speed):
                    obj_heading = objects_heading[i]
                    obj_speed = objects_speed[i]

                    # Calculate relative heading
                    heading_diff = abs(heading - obj_heading)
                    if heading_diff > 180:
                        heading_diff = 360 - heading_diff

                    # Calculate if we're converging (dangerous) or diverging (safe)
                    # Vector from agent to object
                    dx = obj_location[0] - x
                    dy = obj_location[1] - y
                    angle_to_object = math.degrees(math.atan2(dy, dx))

                    # Normalize angle
                    if angle_to_object < 0:
                        angle_to_object += 360
                    if heading < 0:
                        heading_normalized = heading + 360
                    else:
                        heading_normalized = heading

                    angle_diff = abs(heading_normalized - angle_to_object)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    # If heading towards object (small angle difference), apply penalty
                    if angle_diff < 45:  # Heading towards object
                        collision_prevention_penalty -= 20.0 * 2
                    elif angle_diff < 90 and heading_diff < 30:  # Potential convergence
                        collision_prevention_penalty -= 20.0

    reward += collision_prevention_penalty

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

        if closest_object_distance >= 1.2:
            object_reward = 3.5 * 2

            if speed >= 4.0:
                object_reward += 4.0

        elif closest_object_distance >= 0.8:
            object_reward = 3.5

            if objects_ahead and speed > 2.0:
                avg_bot_speed = np.mean([objects_speed[i] for i in objects_ahead if i < len(objects_speed)])
                if speed > avg_bot_speed + 0.8:

                    closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
                    if closest_ahead_idx < len(objects_left_of_center) and closest_ahead_idx < len(objects_location):
                        bot_left = objects_left_of_center[closest_ahead_idx]
                        bot_location = objects_location[closest_ahead_idx]

                        lateral_separation = abs(y - bot_location[1]) if is_left_of_center == bot_left else track_width * 0.5

                        if (
                            bot_left != is_left_of_center
                            and lateral_separation > track_width * 0.2
                            and distance_from_center > track_width * 0.25
                        ):  # Ensure we're using track width

                            # Check if we're moving away from object (not towards it)
                            if closest_ahead_idx < len(objects_heading):
                                bot_heading = objects_heading[closest_ahead_idx]
                                heading_diff = abs(heading - bot_heading)
                                if heading_diff > 180:
                                    heading_diff = 360 - heading_diff

                                # Only reward if not heading directly towards the bot
                                if heading_diff > 15.0:
                                    object_reward += 8.0
                                    object_reward += 3.0

            if objects_ahead and closest_object_distance > 0.6 * 1.2:
                # Check if we're properly behind (not approaching from side)
                closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
                if closest_ahead_idx < len(objects_location):
                    bot_location = objects_location[closest_ahead_idx]
                    # Calculate relative position - only draft if truly behind
                    relative_progress = (progress * track_length / 100) - objects_distance[closest_ahead_idx]
                    if relative_progress < -2:
                        object_reward += 2.0

        elif closest_object_distance >= 0.4:

            object_reward = -2.0

            if objects_ahead and speed > 2.0:
                closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
                if closest_ahead_idx < len(objects_left_of_center) and closest_ahead_idx < len(objects_speed):
                    bot_left = objects_left_of_center[closest_ahead_idx]
                    bot_speed = objects_speed[closest_ahead_idx]

                    # Only reduce penalty if clearly executing overtake maneuver
                    if (
                        bot_left != is_left_of_center
                        and speed > bot_speed + 0.8
                        and heading_diff > 15.0
                        and distance_from_center > track_width * 0.2
                    ):
                        object_reward = 0.5  # Small positive reward for controlled overtaking
                    elif speed <= 4.0 * 0.7:
                        object_reward = -1.0  # Reduced penalty if slowing appropriately

        else:
            object_reward = -20.0

            if objects_ahead and speed > 2.0:
                closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
                if (
                    closest_ahead_idx < len(objects_left_of_center)
                    and closest_ahead_idx < len(objects_speed)
                    and closest_ahead_idx < len(objects_heading)
                ):

                    bot_left = objects_left_of_center[closest_ahead_idx]
                    bot_speed = objects_speed[closest_ahead_idx]
                    bot_heading = objects_heading[closest_ahead_idx]

                    heading_diff = abs(heading - bot_heading)
                    if heading_diff > 180:
                        heading_diff = 360 - heading_diff

                    if (
                        bot_left != is_left_of_center
                        and speed > bot_speed + 0.8 * 2
                        and heading_diff > 15.0 * 1.5  # Stricter angle for emergency
                        and distance_from_center > track_width * 0.2 * 1.5
                    ):  # More lateral space needed
                        object_reward = -5.0
                    else:
                        object_reward = -20.0 * 2

    reward += object_reward
    racing_line_reward = 0.0

    # Calculate optimal racing line based on tactical situation
    if len(waypoints) > closest_waypoints[1] + 4:
        try:
            # Get future waypoints for racing line calculation
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

            direction_change = abs(future_direction - track_direction)
            if direction_change > 180:
                direction_change = 360 - direction_change

            # Racing speed optimization (more aggressive than centerline)
            if direction_change > 45:  # Sharp turn ahead
                optimal_speed = 2.0 + 1.0  # Higher than centerline
                # Reward late braking in racing
                if speed >= optimal_speed and closest_object_distance > 0.8:
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
            if objects_ahead and closest_object_distance < 1.2:
                # Reward positioning for overtake opportunities
                if distance_from_center > track_width * 0.3:  # Outside line setup
                    racing_line_reward += 1.5
                elif distance_from_center < track_width * 0.15:  # Inside line defense
                    racing_line_reward += 2.0

        except (IndexError, ZeroDivisionError):
            pass

    reward += racing_line_reward
    position_reward = 0.0

    # Racing line flexibility (vs strict centerline following)
    centerline_threshold = 0.6 * track_width

    if distance_from_center <= centerline_threshold * 0.5:
        # Excellent racing line position
        position_reward = 2.5
    elif distance_from_center <= centerline_threshold:
        # Good racing line position - allow tactical deviation
        position_factor = 1.0 - (distance_from_center / centerline_threshold)
        position_reward = position_factor * 2.5 * 0.7

        # Bonus for tactical positioning near objects
        if closest_object_distance < 1.2:
            position_reward += 3.0 * 0.5
    else:
        # Outside racing line - minimize penalty if it's tactical
        if closest_object_distance < 0.8:
            # Tactical wide line for overtaking - reduced penalty
            position_reward = -0.1
        else:
            # No tactical reason - standard penalty
            position_reward = -0.5

    reward += position_reward

    steering_reward = 0.0
    abs_steering = abs(steering_angle)

    if abs_steering <= 15.0:
        # Smooth steering - good for most situations
        steering_reward = 1.0
    elif abs_steering <= 25.0:
        # Racing steering - allowed and rewarded in tactical situations
        if closest_object_distance < 1.2:
            # Tactical steering for overtakes/defense
            steering_reward = 1.0
        else:
            # Moderate steering on clear track
            steering_reward = 0.5
    else:
        if closest_object_distance < 0.8:
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
        # Racing efficiency emphasizes speed over step conservation
        speed_efficiency = speed * (progress / 100.0) / (steps / 100.0)
        reward += speed_efficiency * 15.0  # Higher than centerline scenarios

    # Position dominance bonus - reward being ahead of bots
    if objects_behind:
        position_advantage = len(objects_behind) / max(len(objects_location), 1)
        reward += 4.0 * position_advantage

    # Speed advantage bonus - reward outpacing competition
    if bot_speeds and speed > max(bot_speeds) + 0.8:
        speed_dominance = (speed - max(bot_speeds)) / 5.0
        reward += 2.5 * speed_dominance

    # Tactical overtaking bonus - reward successful overtake execution with safety checks
    if objects_ahead and closest_object_distance < 1.2:
        closest_ahead_idx = objects_ahead[0] if objects_ahead else closest_obj_index
        if (
            closest_ahead_idx < len(objects_location)
            and closest_ahead_idx < len(objects_left_of_center)
            and closest_ahead_idx < len(objects_heading)
        ):

            bot_left = objects_left_of_center[closest_ahead_idx]
            bot_location = objects_location[closest_ahead_idx]
            bot_heading = objects_heading[closest_ahead_idx]

            # Calculate trajectory divergence
            heading_diff = abs(heading - bot_heading)
            if heading_diff > 180:
                heading_diff = 360 - heading_diff

            # Calculate lateral separation
            lateral_distance = abs(y - bot_location[1])

            if speed >= 4.5:
                # Only reward high speed overtaking if properly executing safe maneuver
                if (
                    bot_left != is_left_of_center  # Different lanes
                    and lateral_distance > track_width * 0.2
                    and heading_diff > 15.0
                ):

                    reward += 8.0

                    # Lane positioning bonus for safe overtake execution
                    if distance_from_center > track_width * 0.35:  # Outside overtake
                        reward += 1.5
                    elif distance_from_center < track_width * 0.15:  # Inside overtake
                        reward += 2.0
                else:
                    reward -= 8.0 * 0.5

    if objects_behind and bot_speeds:
        max_bot_speed = max([objects_speed[i] for i in objects_behind if i < len(objects_speed)])
        if max_bot_speed > speed * 0.9:
            if distance_from_center <= centerline_threshold * 0.6:
                reward += 2.0

    if closest_object_distance > 1.2 * 1.5:
        if speed >= 4.0:
            clear_track_bonus = (speed / 5.0) ** 2
            reward += 3.5 * clear_track_bonus

    if objects_location and len(objects_location) > 0:
        for i, obj_left in enumerate(objects_left_of_center):
            if i < len(objects_location) and i < len(objects_speed) and i < len(objects_heading):

                obj_distance = math.sqrt((x - objects_location[i][0]) ** 2 + (y - objects_location[i][1]) ** 2)
                obj_speed = objects_speed[i]
                obj_heading = objects_heading[i]

                if 0.8 <= obj_distance <= 1.2:
                    if obj_left != is_left_of_center:

                        heading_diff = abs(heading - obj_heading)
                        if heading_diff > 180:
                            heading_diff = 360 - heading_diff

                        obj_location = objects_location[i]
                        lateral_separation = abs(y - obj_location[1])

                        if (
                            speed >= 4.0 * 0.8
                            and heading_diff > 15.0  # Diverging paths
                            and lateral_separation > track_width * 0.2
                        ):  # Safe lateral distance

                            reward += 3.0

                            if speed > obj_speed + 0.8:
                                reward += 8.0 * 0.3
                        else:
                            reward -= 3.0 * 0.5

    return float(reward)
