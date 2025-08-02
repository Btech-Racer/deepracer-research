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

    Objects Parameters:
    - closest_objects (list_int): Zero-based indices of the two closest objects to agent's position
    - objects_distance (list_float) (meters) [0 to track_length]: List of objects' distances in meters from starting line
    - objects_heading (list_float) (degrees) [-180 to 180]: List of objects' headings in degrees
    - objects_left_of_center (list_boolean): List of Boolean flags indicating if objects are left of center
    - objects_location (list_tuple): List of object locations as (x,y) coordinates
    - objects_speed (list_float) (m/s): List of objects' speeds in meters per second
    """

    # Get parameters from AWS DeepRacer (all standard params are always available)
    all_wheels_on_track = params["all_wheels_on_track"]
    distance_from_center = params["distance_from_center"]
    is_left_of_center = params["is_left_of_center"]
    heading = params["heading"]
    progress = params["progress"]
    speed = params["speed"]
    steering_angle = params["steering_angle"]
    steps = params["steps"]
    params["track_length"]
    track_width = params["track_width"]
    x = params["x"]
    y = params["y"]

    closest_waypoints = params["closest_waypoints"]
    waypoints = params["waypoints"]

    is_crashed = params["is_crashed"]
    is_offtrack = params["is_offtrack"]
    is_reversed = params["is_reversed"]

    # Object parameters
    params["closest_objects"]
    params["objects_distance"]
    objects_heading = params["objects_heading"]
    objects_left_of_center = params["objects_left_of_center"]
    objects_location = params["objects_location"]
    objects_speed = params["objects_speed"]

    # Safety checks
    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(1e-3)

    if is_reversed:
        return float(1e-3)

    # Initialize reward components for detailed tracking
    reward_components = {
        "base": 1.0,
        "position": 0.0,
        "speed": 0.0,
        "steering": 0.0,
        "direction": 0.0,
        "object_avoidance": 0.0,
        "predictive": 0.0,
        "efficiency": 0.0,
        "progress": 0.0,
    }

    closest_object_distance = float("inf")
    predicted_collision_risk = 0.0
    objects_nearby = False
    behavior_mode = "centerline"

    if objects_location and len(objects_location) > 0:
        object_distances = []
        for i, obj_location in enumerate(objects_location):
            obj_x, obj_y = obj_location
            current_distance = math.sqrt((x - obj_x) ** 2 + (y - obj_y) ** 2)
            object_distances.append(current_distance)

            # Enhanced predictive collision detection
            if i < len(objects_speed) and objects_speed[i] > 0:
                # For moving objects, predict future collision paths
                obj_heading = objects_heading[i] if i < len(objects_heading) else 0
                prediction_time = 0.4

                future_obj_x = obj_x + objects_speed[i] * prediction_time * math.cos(math.radians(obj_heading))
                future_obj_y = obj_y + objects_speed[i] * prediction_time * math.sin(math.radians(obj_heading))

                future_agent_x = x + speed * prediction_time * math.cos(math.radians(heading))
                future_agent_y = y + speed * prediction_time * math.sin(math.radians(heading))

                future_distance = math.sqrt((future_agent_x - future_obj_x) ** 2 + (future_agent_y - future_obj_y) ** 2)

                if future_distance < current_distance:
                    collision_risk = max(0, 1.0 - (future_distance / 1.5))
                    predicted_collision_risk = max(predicted_collision_risk, collision_risk)

        if object_distances:
            closest_object_distance = min(object_distances)

            # Determine behavior mode based on object proximity
            if closest_object_distance < 1.8:
                objects_nearby = True
                if closest_object_distance < 1.5:
                    behavior_mode = "avoidance"
                else:
                    behavior_mode = "hybrid"

    # Calculate centerline markers relative to track width
    marker_excellent = 0.1 * track_width
    marker_good = 0.25 * track_width
    marker_acceptable = 0.45 * track_width
    marker_avoidance = 0.6 * track_width

    if behavior_mode == "centerline":
        # Pure centerline following - high precision rewards
        if distance_from_center <= marker_excellent:
            reward_components["position"] = 2.5
        elif distance_from_center <= marker_good:
            position_factor = 1.0 - (distance_from_center - marker_excellent) / (marker_good - marker_excellent)
            reward_components["position"] = 1.5 + (2.5 - 1.5) * position_factor
        elif distance_from_center <= marker_acceptable:
            position_factor = 1.0 - (distance_from_center - marker_good) / (marker_acceptable - marker_good)
            reward_components["position"] = 0.8 + (1.5 - 0.8) * position_factor
        else:
            reward_components["position"] = 0.8 * 0.3

    elif behavior_mode == "avoidance":
        # Object avoidance mode - reward staying away from objects with more tolerance for track position
        if distance_from_center <= marker_avoidance:
            reward_components["position"] = 0.8
        else:
            # Still penalize being too far from center, but less severely
            reward_components["position"] = 0.8 * 0.5

    else:  # hybrid mode
        # Blend centerline and avoidance positioning
        centerline_weight = 0.7
        avoidance_weight = 1.0

        # Calculate centerline component
        if distance_from_center <= marker_good:
            centerline_component = 1.5
        elif distance_from_center <= marker_acceptable:
            position_factor = 1.0 - (distance_from_center - marker_good) / (marker_acceptable - marker_good)
            centerline_component = 0.8 + (1.5 - 0.8) * position_factor
        else:
            centerline_component = 0.8 * 0.5

        # Calculate avoidance component (reward staying on track with tolerance)
        avoidance_component = 0.8 if distance_from_center <= marker_avoidance else 0.8 * 0.3

        # Blend based on object proximity
        proximity_factor = min(1.0, closest_object_distance / 1.8)
        reward_components["position"] = (
            centerline_component * centerline_weight * proximity_factor
            + avoidance_component * avoidance_weight * (1.0 - proximity_factor)
        )

    if objects_nearby:
        # Distance-based object avoidance rewards
        if closest_object_distance >= 2.0:
            # Very safe distance - maximum reward
            reward_components["object_avoidance"] = 3.0
        elif closest_object_distance >= 1.5:
            # Safe distance - good reward
            distance_factor = (closest_object_distance - 1.5) / (2.0 - 1.5)
            reward_components["object_avoidance"] = 3.0 * (0.6 + 0.4 * distance_factor)
        elif closest_object_distance >= 0.8:
            # Warning zone - moderate reward
            distance_factor = (closest_object_distance - 0.8) / (1.5 - 0.8)
            reward_components["object_avoidance"] = 3.0 * (0.2 + 0.4 * distance_factor)
        elif closest_object_distance >= 0.4:
            # Danger zone - minimal reward
            distance_factor = (closest_object_distance - 0.4) / (0.8 - 0.4)
            reward_components["object_avoidance"] = 3.0 * (0.05 + 0.15 * distance_factor)
        else:
            # Too close - severe penalty
            reward_components["object_avoidance"] = -3.0 * 2

        # Predictive avoidance bonus
        if predicted_collision_risk > 0.1:
            reward_components["predictive"] = 1.8 * predicted_collision_risk

        # Strategic positioning relative to object clusters
        if objects_left_of_center and len(objects_left_of_center) > 0:
            objects_on_left = sum(objects_left_of_center)
            objects_on_right = len(objects_left_of_center) - objects_on_left

            # Reward moving to the side with fewer objects
            if objects_on_left > objects_on_right and not is_left_of_center:
                reward_components["object_avoidance"] += 0.5
            elif objects_on_right > objects_on_left and is_left_of_center:
                reward_components["object_avoidance"] += 0.5

    target_speed = 2.5

    if behavior_mode == "avoidance":
        # Near objects - conservative speed
        target_speed = 1.5
        if speed <= target_speed:
            speed_factor = speed / target_speed
            reward_components["speed"] = 2.0 * speed_factor * 0.8
        elif speed <= 2.0:
            reward_components["speed"] = 2.0 * 0.4
        else:
            reward_components["speed"] = -0.8  # Penalty for going too quickly near objects

    elif behavior_mode == "centerline":
        # Clear track - optimize for speed
        if speed >= 2.5:
            speed_factor = min(speed / 4.0, 1.0)
            reward_components["speed"] = 2.0 * (speed_factor**1.1)
        elif speed >= 1.0:
            speed_factor = (speed - 1.0) / (2.5 - 1.0)
            reward_components["speed"] = 2.0 * speed_factor * 0.7
        else:
            reward_components["speed"] = -0.5

    else:  # hybrid mode
        # Balanced speed for transitioning between behaviors
        if speed >= 2.0:
            speed_factor = min(speed / 4.0, 1.0)
            reward_components["speed"] = 2.0 * speed_factor * 0.9
        elif speed >= 1.0:
            speed_factor = (speed - 1.0) / (2.0 - 1.0)
            reward_components["speed"] = 2.0 * speed_factor * 0.6
        else:
            reward_components["speed"] = -0.3

    abs_steering = abs(steering_angle)

    if behavior_mode == "avoidance":
        # Allow more aggressive steering near objects
        if abs_steering <= 20.0:
            steering_factor = 1.0 - (abs_steering / 20.0) * 0.3
            reward_components["steering"] = 1.2 * steering_factor
        elif abs_steering <= 28.0:
            reward_components["steering"] = 1.2 * 0.4
        else:
            reward_components["steering"] = -0.3

    else:  # centerline or hybrid
        # Prefer smooth steering
        if abs_steering <= 12.0:
            reward_components["steering"] = 1.2
        elif abs_steering <= 20.0:
            steering_factor = 1.0 - (abs_steering - 12.0) / (20.0 - 12.0)
            reward_components["steering"] = 1.2 * (0.3 + 0.7 * steering_factor)
        else:
            reward_components["steering"] = 1.2 * 0.1

    if len(waypoints) > closest_waypoints[1]:
        try:
            # Look ahead for better racing line prediction
            lookahead_point = closest_waypoints[1]
            if behavior_mode == "centerline" and len(waypoints) > closest_waypoints[1] + 2:
                # For centerline following, look further ahead
                lookahead_point = min(closest_waypoints[1] + 2, len(waypoints) - 1)

            next_point = waypoints[lookahead_point]
            current_point = waypoints[closest_waypoints[0]]

            track_direction = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
            track_direction = math.degrees(track_direction)

            direction_diff = abs(heading - track_direction)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            if direction_diff <= 15.0:
                alignment_factor = 1.0 - (direction_diff / 15.0)
                reward_components["direction"] = 1.0 * alignment_factor
            else:
                reward_components["direction"] = -0.2

        except (IndexError, ZeroDivisionError):
            pass

    # Major completion bonuses
    if progress >= 99.0:
        reward_components["progress"] = 20.0
    elif progress >= 90.0:
        reward_components["progress"] = 20.0 * 0.4
    elif progress >= 75.0:
        reward_components["progress"] = 20.0 * 0.2
    elif progress >= 50.0:
        reward_components["progress"] = 20.0 * 0.1

    # Efficiency calculation
    if steps > 0:
        # Reward speed and progress efficiency with safety considerations
        safety_factor = min(1.0, closest_object_distance / 1.5) if objects_nearby else 1.0
        base_efficiency = (progress / 100.0) * (speed / 4.0) / (steps / 100.0)
        safe_efficiency = base_efficiency * safety_factor
        reward_components["efficiency"] = safe_efficiency * 1.5

        # Penalty for taking too many steps
        expected_steps = progress * 1.3  # Allow some tolerance
        if steps > expected_steps:
            step_penalty = (steps - expected_steps) * 0.01
            reward_components["efficiency"] -= step_penalty

    final_reward = sum(reward_components.values())

    # Ensure minimum reward threshold
    final_reward = max(final_reward, 1e-3)

    return float(final_reward)
