# Place import statement outside of function (supported libraries: math, random, numpy, scipy, and shapely)
import math


def reward_function(params):
    """
    Reward function for object avoidance

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

    params["closest_waypoints"]
    params["waypoints"]

    is_crashed = params["is_crashed"]
    is_offtrack = params["is_offtrack"]
    is_reversed = params["is_reversed"]

    params["closest_objects"]
    params["objects_distance"]
    objects_heading = params["objects_heading"]
    objects_left_of_center = params["objects_left_of_center"]
    objects_location = params["objects_location"]
    objects_speed = params["objects_speed"]

    if not all_wheels_on_track or is_crashed or is_offtrack:
        return float(1e-3)

    if is_reversed:
        return float(1e-3)

    reward = 2.0

    object_risk_factor = 1.0
    closest_object_distance = float("inf")
    predicted_collision_risk = 0.0

    if objects_location and len(objects_location) > 0:
        object_distances = []
        for i, obj_location in enumerate(objects_location):
            obj_x, obj_y = obj_location
            current_distance = math.sqrt((x - obj_x) ** 2 + (y - obj_y) ** 2)
            object_distances.append(current_distance)

            # Predictive collision detection
            if i < len(objects_speed) and objects_speed[i] > 0:
                # For moving objects, predict future position
                obj_heading = objects_heading[i] if i < len(objects_heading) else 0
                future_obj_x = obj_x + objects_speed[i] * 0.3 * math.cos(math.radians(obj_heading))
                future_obj_y = obj_y + objects_speed[i] * 0.3 * math.sin(math.radians(obj_heading))

                # Predict agent future position
                future_agent_x = x + speed * 0.3 * math.cos(math.radians(heading))
                future_agent_y = y + speed * 0.3 * math.sin(math.radians(heading))

                future_distance = math.sqrt((future_agent_x - future_obj_x) ** 2 + (future_agent_y - future_obj_y) ** 2)

                if future_distance < current_distance:
                    predicted_collision_risk = max(predicted_collision_risk, 1.0 - (future_distance / 1.2))

        if object_distances:
            closest_object_distance = min(object_distances)

            # Enhanced distance-based rewards
            if closest_object_distance >= 1.2:
                reward += 4.0
                object_risk_factor = 1.0
            elif closest_object_distance >= 0.6:
                distance_factor = (closest_object_distance - 0.6) / (1.2 - 0.6)
                reward += 4.0 * (0.5 + 0.5 * distance_factor)
                object_risk_factor = 0.5 + 0.5 * distance_factor
            elif closest_object_distance >= 0.3:
                distance_factor = (closest_object_distance - 0.3) / (0.6 - 0.3)
                reward += 4.0 * (0.1 + 0.4 * distance_factor)
                object_risk_factor = 0.1 + 0.4 * distance_factor
            else:
                # Very close to object - major penalty
                reward *= 0.05
                object_risk_factor = 0.05

            # Predictive avoidance bonus
            if predicted_collision_risk > 0.5:
                reward += 1.5 * (1.0 - predicted_collision_risk)

    # Adaptive speed control based on object proximity
    speed_reward = 0.0
    if closest_object_distance < 1.2:
        # Near objects - reward conservative speed
        if speed <= 1.0:
            speed_reward = 0.8
        elif speed <= 2.0:
            speed_reward = 0.5
        else:
            speed_reward = -0.3  # Penalty for going too quickly near objects
    else:
        # Clear track - reward maintaining reasonable speed
        if speed >= 2.0:
            speed_factor = min(speed / 4.0, 1.0)
            speed_reward = 0.6 * speed_factor
        else:
            speed_reward = 0.2

    reward += speed_reward

    # Enhanced steering control
    abs_steering = abs(steering_angle)
    steering_reward = 0.0

    if closest_object_distance < 0.6:
        # Near objects - allow more aggressive steering
        if abs_steering <= 25.0:
            steering_reward = 1.2 * 0.7
        elif abs_steering <= 30.0:
            steering_reward = 1.2 * 0.3
        else:
            steering_reward = -0.4
    else:
        # Clear track - prefer smooth steering
        if abs_steering <= 15.0:
            steering_reward = 1.2
        elif abs_steering <= 30.0:
            steering_reward = 1.2 * 0.5
        else:
            steering_reward = -0.2

    reward += steering_reward

    # Improved positioning rewards
    position_reward = 0.0
    track_center_threshold = 0.4 * track_width

    if closest_object_distance < 0.6:
        # When near objects, reward staying on track with more tolerance
        if distance_from_center <= track_center_threshold:
            position_reward = 1.0 * 0.5
    else:
        # When clear, reward staying closer to optimal racing line
        if distance_from_center <= track_width * 0.25:
            position_factor = 1.0 - (distance_from_center / (track_width * 0.25))
            position_reward = 1.0 * position_factor
        elif distance_from_center <= track_center_threshold:
            position_reward = 1.0 * 0.3

    reward += position_reward

    # Progress and completion rewards
    if progress >= 99.0:
        reward += 15.0
    elif progress >= 90.0:
        reward += 2.5 * 2
    elif progress >= 75.0:
        reward += 2.5

    # Efficiency calculation with safety factor
    if steps > 0:
        base_efficiency = (progress / 100.0) / (steps / 100.0)
        safety_efficiency = base_efficiency * object_risk_factor
        reward += safety_efficiency * 2.0

    # Strategic positioning relative to objects
    if objects_left_of_center and len(objects_left_of_center) > 0:
        objects_on_left = sum(objects_left_of_center)
        objects_on_right = len(objects_left_of_center) - objects_on_left

        # Reward moving to the side with fewer objects
        if objects_on_left > objects_on_right and not is_left_of_center:
            reward += 0.5
        elif objects_on_right > objects_on_left and is_left_of_center:
            reward += 0.5

    return float(max(reward, 1e-3))
