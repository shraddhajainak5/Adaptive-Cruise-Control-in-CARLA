from mp1_simulator.simulator import Observation

class Controller:
    def __init__(self, target_speed: float, distance_threshold: float):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold

        self.kp = 1.5
        self.kd = 0.3

        self.previous_speed_error = 0.0
        self.previous_dist_to_lead = None

        self.min_velocity = 0.5

    def run_step(self, obs: Observation) -> float:
        """Main control logic for the ACC system."""
        ego_velocity = obs.ego_velocity
        dist_to_lead = obs.distance_to_lead

        max_acceleration = 10.0
        max_deceleration = -10.0

        if dist_to_lead is None or dist_to_lead <= 0:
            return max_deceleration  

        reaction_time = 1.5  
        critical_distance = 18.0 + reaction_time * ego_velocity
        safe_distance = critical_distance + 10.0

        speed_error = self.target_speed - ego_velocity
        speed_error_factor = speed_error - self.previous_speed_error
        self.previous_speed_error = speed_error

        if self.previous_dist_to_lead is not None:
            closing_rate = self.previous_dist_to_lead - dist_to_lead
        else:
            closing_rate = 0.0
        self.previous_dist_to_lead = dist_to_lead

        acceleration = 0.0

        if dist_to_lead < critical_distance:
            acceleration = max_deceleration
        elif dist_to_lead < safe_distance:
            distance_factor = (dist_to_lead - critical_distance) / (safe_distance - critical_distance)
            acceleration = max_deceleration * (1 - distance_factor)
            acceleration += 0.1 * closing_rate  
        else:
            if ego_velocity < self.target_speed:
                acceleration = self.kp * speed_error + self.kd * speed_error_factor

                if dist_to_lead > safe_distance * 1.5:
                    if ego_velocity + acceleration < self.target_speed:
                        acceleration += 5.0

                if ego_velocity + acceleration > self.target_speed:
                    acceleration = self.target_speed - ego_velocity

            elif ego_velocity >= self.target_speed:
                acceleration = 0.0

                if ego_velocity > self.target_speed + 0.1:
                    acceleration = max_deceleration  

        acceleration = max(min(acceleration, max_acceleration), max_deceleration)

        if ego_velocity < self.min_velocity and acceleration < 0:
            acceleration = 0.0

        return acceleration
