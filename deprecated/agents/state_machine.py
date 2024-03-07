import numpy as np


class StateMachine:
    def loadTransferConditions(self, strategy_file):
        """
        Load the transfer conditions into the dict machine_strat from a given .adv file
        :param adv_file:
        :return:
        """
        # Initialize dictionary
        self.machine_strat = dict()

        loaded_strat = np.genfromtxt(strategy_file, dtype="str")

        # Grab the key-value pairs
        s = loaded_strat[:, 0]
        a = loaded_strat[:, 1]

        # Create dictionary
        self.machine_strat = dict(zip(s, a))

    # Define state machine -> based on this state, select this action
    def selectAction(self, discrete_state):
        """
        Select an action based on the discretized state
        :param discrete_state:
        :return action:
        """
        # Return the action -> accessed by passing the state to the machine strategy
        # dictionary
        return int(self.machine_strat[str(discrete_state)])

    def SimpleEOSDiscretizer(self, obs):
        """
        Discretizes the simplEOS states into 16 bins
        :param SimpleEOS environment, obs:
        :return system_state:
        """
        obs = obs.flatten()

        # Set limits for wheel speed, power limits, and tumble rates
        errWheelSpeed = 0.5  # percent of max
        errPowerLimit = 0.5  # percent of max
        errTumbleRate = 1e-2  # rad/s
        errBufferLimit = 0.8  # percent of max

        # Assume the spacecraft is not tumbling, not in low power, the wheels are not
        # saturated, and no buffer overflow
        tumble = False
        lowPow = False
        saturated = False
        bufferOverflow = False

        # Check observations to see if above assumptions are incorrect
        if obs[7] > errTumbleRate:
            tumble = True
        if np.linalg.norm(obs[8:11]) > errWheelSpeed:
            saturated = True
        if obs[11] < errPowerLimit:
            lowPow = True
        if obs[13] > errBufferLimit:
            bufferOverflow = True

        # If spacecraft is not in an optimal state
        if any([tumble, saturated, lowPow, bufferOverflow]):
            # check to find out what the error mode is
            if tumble:
                if saturated:
                    if lowPow:
                        if bufferOverflow:
                            system_state = 15
                            return system_state
                        else:
                            system_state = 14
                            return system_state
                    if bufferOverflow:
                        system_state = 13
                        return system_state
                    else:
                        system_state = 12
                        return system_state
                if lowPow:
                    if bufferOverflow:
                        system_state = 11
                        return system_state
                    else:
                        system_state = 10
                        return system_state
                if bufferOverflow:
                    system_state = 9
                    return system_state
                else:
                    system_state = 8
                    return system_state
            elif not tumble:
                if saturated:
                    if lowPow:
                        if bufferOverflow:
                            system_state = 7
                            return system_state
                        else:
                            system_state = 6
                            return system_state
                    else:
                        if bufferOverflow:
                            system_state = 5
                            return system_state
                        else:
                            system_state = 4
                            return system_state
                if lowPow:
                    if bufferOverflow:
                        system_state = 3
                        return system_state
                    else:
                        system_state = 2
                        return system_state
                if bufferOverflow:
                    system_state = 1
                    return system_state
        # Otherwise, return nominal
        else:
            system_state = 0
            return system_state

    def earthObsEnvDiscretizer(self, obs):
        """
        Discretizes the MultiSensorEOS environment states into 8 bins
        :param MultiSensorEOS, obs:
        :return system_state:
        """
        obs = obs.flatten()

        errWheelSpeed = 0.8  # percent of max
        errPowerLimit = 0.2  # percent of max
        errTumbleRate = 1e-2  # rad/s

        # Assume the spacecraft is not tumbling, not in low power, the wheels are not
        # saturated, and no buffer overflow
        tumble = False
        lowPow = False
        saturated = False

        # Check observations to see if above assumptions are incorrect
        if obs[7] > errTumbleRate:
            tumble = True
        if obs[8] > errWheelSpeed:
            saturated = True
        if obs[9] < errPowerLimit:
            lowPow = True

        # If spacecraft is not in an optimal state
        if any([tumble, saturated, lowPow]):
            # check to find out what the error mode is
            if tumble:
                if saturated:
                    if lowPow:
                        system_state = 7
                        return system_state
                    else:
                        system_state = 3
                        return system_state
                if lowPow:
                    system_state = 5
                    return system_state
                else:
                    system_state = 0
                    return system_state
            elif not tumble:
                if saturated:
                    if lowPow:
                        system_state = 6
                        return system_state
                    else:
                        system_state = 2
                        return system_state
                if lowPow:
                    system_state = 4
                    return system_state
        # Otherwise, return nominal
        else:
            system_state = 0
            return system_state

    def AgileEOSEnvDiscretizer(self, obs):
        """
        Discretizes the AgileEOS states into 16 bins
        :param AgileEOS environment, obs:
        :return system_state:
        """
        obs = obs.flatten()

        # Set limits for wheel speed, power limits, and tumble rates
        errWheelSpeed = 0.6  # percent of max
        errPowerLimit = 0.25  # percent of max
        errTumbleRate = 1e-2  # rad/s
        errBufferLimit = 0.94  # percent of max

        # Assume the spacecraft is not tumbling, not in low power, the wheels are not
        # saturated, and no buffer overflow
        tumble = False
        lowPow = False
        saturated = False
        bufferOverflow = False

        # Check observations to see if above assumptions are incorrect
        if obs[7] > errTumbleRate:
            tumble = True
        if np.linalg.norm(obs[8:11]) > errWheelSpeed:
            saturated = True
        if obs[11] < errPowerLimit:
            lowPow = True
        if obs[13] > errBufferLimit:
            bufferOverflow = True

        # If spacecraft is not in an optimal state
        if any([tumble, saturated, lowPow, bufferOverflow]):
            # check to find out what the error mode is
            if tumble:
                if saturated:
                    if lowPow:
                        if bufferOverflow:
                            system_state = 15
                            return system_state
                        else:
                            system_state = 14
                            return system_state
                    if bufferOverflow:
                        system_state = 13
                        return system_state
                    else:
                        system_state = 12
                        return system_state
                if lowPow:
                    if bufferOverflow:
                        system_state = 11
                        return system_state
                    else:
                        system_state = 10
                        return system_state
                if bufferOverflow:
                    system_state = 9
                    return system_state
                else:
                    system_state = 8
                    return system_state
            elif not tumble:
                if saturated:
                    if lowPow:
                        if bufferOverflow:
                            system_state = 7
                            return system_state
                        else:
                            system_state = 6
                            return system_state
                    else:
                        if bufferOverflow:
                            system_state = 5
                            return system_state
                        else:
                            system_state = 4
                            return system_state
                if lowPow:
                    if bufferOverflow:
                        system_state = 3
                        return system_state
                    else:
                        system_state = 2
                        return system_state
                if bufferOverflow:
                    system_state = 1
                    return system_state
        # Otherwise, return nominal
        else:
            system_state = 0
            return system_state

    def smallBodyScienceEnvDiscretizer(self, obs):
        """
        Discretizes the SmallBodyScience environment into bins
        :param SmallBodyScience, obs:
        :return system_state:
        """
        obs = obs.flatten()

        # Set limits for wheel speed, power limits, and tumble rates
        errPowerLimit = 0.5  # percent of max
        errBufferLimit = 0.8  # percent of max
        errFuelLimit = 0.9

        # Assume the spacecraft is not tumbling, not in low power, the wheels are not
        # saturated, and no buffer overflow
        lowFuel = False
        lowPow = False
        bufferOverflow = False

        # Check observations to see if above assumptions are incorrect
        if obs[8] < errPowerLimit:
            lowPow = True
        if obs[7] > errBufferLimit:
            bufferOverflow = True
        if obs[9] > errFuelLimit:
            lowFuel = True

        # If spacecraft is not in an optimal state
        if any([bufferOverflow, lowFuel, lowPow]):
            # check to find out what the error mode is
            if bufferOverflow:
                if lowFuel:
                    if lowPow:
                        system_state = 1
                        return system_state
                    else:
                        system_state = 2
                        return system_state
                if lowPow:
                    system_state = 3
                    return system_state
                else:
                    system_state = 4
                    return system_state
            elif not bufferOverflow:
                if lowFuel:
                    if lowPow:
                        system_state = 5
                        return system_state
                    else:
                        system_state = 6
                        return system_state
                if lowPow:
                    system_state = 7
                    return system_state
        # Otherwise, return nominal
        else:
            system_state = 0
            return system_state

    def wrap_phi(self, phi_c):
        if phi_c < -90:
            phi_c += 180
        elif phi_c > 90:
            phi_c -= 180

        return phi_c

    def wrap_lambda(self, lambda_c):
        if lambda_c < -180:
            lambda_c += 360
        elif lambda_c > 180:
            lambda_c -= 360

        return lambda_c

    def setupSmallBodyScience(self):
        self.start_set = False
        self.start_point = None
        self.tour_start = False
        self.tour_finished = False

        self.d_phi = 30
        self.d_lambda = 60

        # Define the waypoint deltas for the action space
        self.waypoint_latitude_deltas = [
            self.d_phi,
            self.d_phi,
            0,
            -self.d_phi,
            -self.d_phi,
            -self.d_phi,
            0,
            self.d_phi,
        ]
        self.waypoint_longitude_deltas = [
            0,
            self.d_lambda,
            self.d_lambda,
            self.d_lambda,
            0,
            -self.d_lambda,
            -self.d_lambda,
            -self.d_lambda,
        ]

        # Define the three legs
        leg_1 = [[75, -90], [45, -90], [15, -90], [-15, -90], [-45, -90], [-75, -90]]
        leg_2 = [
            [75, -150],
            [45, -150],
            [15, -150],
            [-15, -150],
            [-45, -150],
            [-75, -150],
        ]
        leg_3 = [[75, 150], [45, 150], [15, 150], [-15, 150], [-45, 150], [-75, 150]]

        # Define each potential rotation
        self.rotations = [
            leg_1 + list(reversed(leg_2)) + leg_3,
            list(reversed(leg_1)) + leg_2 + list(reversed(leg_3)),
            leg_3 + list(reversed(leg_2)) + leg_1,
            list(reversed(leg_3)) + leg_2 + list(reversed(leg_1)),
        ]

        # Define the actions for ease of implementation
        self.rotation_actions = [
            [
                "down",
                "down",
                "down",
                "down",
                "down",
                "left",
                "up",
                "up",
                "up",
                "up",
                "up",
                "left",
                "down",
                "down",
                "down",
                "down",
                "down",
            ],
            [
                "up",
                "up",
                "up",
                "up",
                "up",
                "left",
                "down",
                "down",
                "down",
                "down",
                "down",
                "left",
                "up",
                "up",
                "up",
                "up",
                "up",
            ],
            [
                "down",
                "down",
                "down",
                "down",
                "down",
                "right",
                "up",
                "up",
                "up",
                "up",
                "up",
                "right",
                "down",
                "down",
                "down",
                "down",
                "down",
            ],
            [
                "up",
                "up",
                "up",
                "up",
                "up",
                "right",
                "down",
                "down",
                "down",
                "down",
                "down",
                "right",
                "up",
                "up",
                "up",
                "up",
                "up",
            ],
        ]

        self.action_dict = {
            "charge": 0,
            "up": 1,
            "up_right": 2,
            "right": 3,
            "down_right": 4,
            "down": 5,
            "down_left": 6,
            "left": 7,
            "up_left": 8,
            "map": 9,
            "downlink": 10,
            "image": 11,
        }

        return

    def lcs(self, X, Y, m, n):
        if m == 0 or n == 0:
            return 0
        elif X[m - 1] == Y[n - 1]:
            return 1 + self.lcs(X, Y, m - 1, n - 1)
        else:
            return max(self.lcs(X, Y, m, n - 1), self.lcs(X, Y, m - 1, n))

    def smallBodyScienceAct(
        self,
        discretized_state,
        phi_c,
        lambda_c,
        simTime,
        waypointTime,
        requiredWaypointTime,
        obs,
        target_hist,
    ):
        """
        Called to determine the next action to take in the small body environment
        :param discretized_state: The discretized state of the environment
        :param phi_c: The current latitude of the spacecraft
        :param lambda_c: The current longitude of the spacecraft
        :param simTime: The current simulation time
        :param waypointTime: The current time of the transit to the next waypoint
        :param requiredWaypointTime: The time required to reach the current waypoint
        :param obs: The current observation
        :param target_hist: The history of the waypoints
        :return action: The next action to take"""
        started_tour = False
        on_tour = False
        finished_tour = False

        # Find the longest common subsequence
        subsequence_lens = np.zeros(4)
        for idx, path in enumerate(self.rotations):
            temp_tgt_hist = target_hist[-5:]
            subsequence_lens[idx] = self.lcs(
                path, temp_tgt_hist, len(path), len(temp_tgt_hist)
            )
        path_lens = list(subsequence_lens)
        max_idx = path_lens.index(max(path_lens))

        temp_tgt_hist = []
        for point in target_hist:
            if point in self.rotations[0]:
                temp_tgt_hist.append(point)

        # If the longest common subsequence is length 1, grab the one nearest to the
        # start
        if 5 >= path_lens[max_idx] > 0:
            path_lens = np.zeros(4)
            for idx, path in enumerate(self.rotations):
                for idx2, point in enumerate(path):
                    if point in target_hist:
                        # path_lens[idx] += (36-idx2)
                        path_lens[idx] += 36.0 / (
                            abs(temp_tgt_hist.index(point) - idx2) + 1
                        )
            path_lens = list(path_lens)
            max_idx = path_lens.index(max(path_lens))
        elif path_lens[max_idx] == 0:
            nearest_dist = []
            for idx, path in enumerate(self.rotations):
                nearest_dist.append(
                    abs(path[0][0] - phi_c) + abs(path[0][1] - lambda_c)
                )
            max_idx = nearest_dist.index(min(nearest_dist))

        self.start_point = max_idx

        # Determine if the tour was started
        if path_lens[max_idx] > 0:
            started_tour = True

        # Determine if currently on tour
        if [phi_c, lambda_c] in self.rotations[max_idx]:
            self.action_idx = self.rotations[max_idx].index([phi_c, lambda_c]) - 1
            on_tour = True

        # Determine if the tour was finished
        if self.rotations[max_idx][-1] in target_hist:
            finished_tour = True

        # Nominal mode
        if discretized_state == 0:
            if finished_tour:
                if obs[10] > 0.9:
                    return 10
                else:
                    return 11

            if started_tour and on_tour:
                # If a downlink window is available:
                if obs[10] > 0.9:
                    return 10
                # If we have reached the next point, target the next one
                elif (simTime - waypointTime) >= requiredWaypointTime:
                    self.action_idx += 1  # Moved to prevent index issue
                    if self.action_idx >= (
                        len(self.rotation_actions[self.start_point])
                    ):
                        return 9
                    else:
                        return self.action_dict[
                            self.rotation_actions[max_idx][self.action_idx]
                        ]
                # If we have not reached the next point, map instead
                else:
                    return 9
            elif started_tour and not on_tour:
                # If a downlink window is available:
                if obs[10] > 0.9:
                    return 10
                # If we have reached the next point, target the next one
                elif (simTime - waypointTime) >= requiredWaypointTime:
                    distances = np.zeros(len(self.rotations[max_idx]))
                    for idx, lat_long in enumerate(self.rotations[max_idx]):
                        # Compute the manhattan distance to two starting points
                        if (
                            lambda_c == -150 and self.rotations[max_idx][idx][1] == 150
                        ) or (
                            lambda_c == 150 and self.rotations[max_idx][idx][1] == -150
                        ):
                            dist_c_lambda = 2
                        else:
                            dist_c_lambda = (
                                abs(lambda_c - self.rotations[max_idx][idx][1]) / 60.0
                            )
                        distances[idx] = (
                            dist_c_lambda
                            + abs(phi_c - self.rotations[max_idx][idx][0]) / 30.0
                        )
                    distances = list(distances)
                    min_dist_idx = distances.index(min(distances))
                    next_step_latlong_dist = np.zeros(8)
                    for idx in range(0, 8):
                        # Compute the next latitude
                        next_lat = phi_c + self.waypoint_latitude_deltas[idx]
                        next_lat = self.wrap_phi(next_lat)

                        # Compute the next longitude
                        next_long = lambda_c + self.waypoint_longitude_deltas[idx]
                        next_long = self.wrap_lambda(next_long)

                        # Compute the manhattan distance
                        next_step_latlong_dist[idx] = abs(
                            self.rotations[max_idx][min_dist_idx][0] - next_lat
                        ) + abs(self.rotations[max_idx][min_dist_idx][1] - next_long)

                    next_step_latlong_dist = list(next_step_latlong_dist)
                    return next_step_latlong_dist.index(min(next_step_latlong_dist)) + 1
                else:
                    return 9
            elif not started_tour:
                if (simTime - waypointTime) >= requiredWaypointTime:
                    next_step_latlong_dist = []
                    for idx in range(0, 8):
                        # Compute the next latitude
                        next_lat = phi_c + self.waypoint_latitude_deltas[idx]
                        next_lat = self.wrap_phi(next_lat)

                        # Compute the next longitude
                        next_long = lambda_c + self.waypoint_longitude_deltas[idx]
                        next_long = self.wrap_lambda(next_long)

                        # Compute the manhattan distance
                        next_step_latlong_dist.append(
                            abs(self.rotations[self.start_point][0][0] - next_lat)
                            + abs(self.rotations[self.start_point][0][1] - next_long)
                        )

                    return next_step_latlong_dist.index(min(next_step_latlong_dist)) + 1
                # Otherwise, image
                else:
                    return 11

        # Low power, low fuel, and buffer overflow
        elif discretized_state == 1:
            return 0  # Charge
        # Nominal power, low fuel, and buffer overflow
        elif discretized_state == 2:
            return 10  # Downlink
        # Lower power, buffer overflow, Nominal fuel
        elif discretized_state == 3:
            return 0  # Charge
        # Nominal power, buffer overflow, nominal fuel
        elif discretized_state == 4:
            return 10  # Downlink
        # Low power, nominal buffer, low fuel
        elif discretized_state == 5:
            return 0  # Charge
        # Nominal power, nominal buffer, low fuel
        elif discretized_state == 6:
            return np.random.choice([9, 11])  # Image or map
        # Low power, nominal buffer, nominal fuel
        elif discretized_state == 7:
            return 0  # Charge
        else:
            print("invalid state discretization: ", discretized_state)
