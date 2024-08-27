from copy import deepcopy
import numpy as np

VEH_CAR_KINEMATIC = {
    "type": "bicycle",
    "wheelbase": 2.4,
}


PARALLEL_PARKING = {
    "action": {"type": "continuous_steering_accel"},
    "longitudinal": {"type": "simple"},
    # 'steering': 'direct()',
    "steering": "linear(60)",
    "problem": {"type": "parallel_parking", "start": "after", "max_time": 15},
    "collision_bb": (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    "dt": 0.1,
    "vehicle": VEH_CAR_KINEMATIC,
    "sensors": {
        "cones_set": {
            "type": "conemap",
            "bbox": (-30, 30, -30, 30),
        },
    },
}

LIDAR_PARALLEL_PARKING = {
    "action": {"type": "continuous_steering_accel"},
    "longitudinal": {"type": "simple"},
    # 'steering': 'direct()',
    "steering": "linear(60)",
    "problem": {
        "type": "lidar_parallel_parking",
        "start": "after",
        "max_time": 15,
        # "soft_collisions": True,
    },
    "collision_bb": (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    "dt": 0.05,
    "vehicle": VEH_CAR_KINEMATIC,
    "sensors": {
        "lidar_points": {
            "type": "lidar",
            "angle_min": -3.14,
            "angle_max": 3.14,
            "angle_increment": 0.01,  # 0.00872665,
            "range_max": 10,
            "measurement_noise_stdev": 0.025,
        },
    },
}

RESETLESS_LIDAR_PARALLEL_PARKING = {
    "action": {"type": "continuous_steering_accel"},
    "longitudinal": {"type": "simple"},
    # 'steering': 'direct()',
    "steering": "linear(60)",
    "problem": {
        "type": "lidar_parallel_parking",
        "start": "after",
        "max_time": np.inf,
        # "soft_collisions": True,
    },
    "collision_bb": (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    "dt": 0.05,
    "vehicle": VEH_CAR_KINEMATIC,
    "sensors": {
        "lidar_points": {
            "type": "lidar",
            "angle_min": -3.14,
            "angle_max": 3.14,
            "angle_increment": 0.01,  # 0.00872665,
            "range_max": 10,
            "measurement_noise_stdev": 0.025,
        },
    },
}

VEH_CAR_DYNAMIC = {
    "type": "dyn_dugoff",
    "wheelbase": 2.4,
    "mass": 750.0,
    "inertia": 812.0,  # Approximated by 0.1269*m*R*L according to "Approximation von Trägheitsmomenten bei Personenkraftwagen", Burg, 1982
    "inertia_front": 2.0,  # Inertia of front axle
    "inertia_rear": 2.0,  # Inertia of rear axle
    "engine_power": 60 * 1000,
    "engine_torque": 1_200.0,
    "brake_torque": 3_000.0,
    "brake_balance": 0.5,
    "mu_front": 1.05,
    "mu_rear": 1.05,
    "c_alpha_front": 10.0,
    "c_sigma_front": 10.0,
    "c_alpha_rear": 12.0,
    "c_sigma_rear": 12.0,
    "rwd": False,
}

# Racing with dynamic single track model
RACING = {
    "action": {"type": "continuous_steering_pedals"},
    "steering": "linear(60)",
    "collision_bb": (-3.81 / 2, 3.81 / 2, -1.48 / 2, 1.48 / 2),
    "vehicle": VEH_CAR_DYNAMIC,
    "problem": {
        "type": "racing",
        "track_width": 8.0,
        "cone_width": 7.0,
        "k_forwards": 0.1,
        "k_base": 0.0,
        "extend": 150,
        "time_limit": 60.0,
    },
    "dt": 0.1,
    "physics_divider": 20,
    "sensors": {
        "cones_set": {
            "type": "conemap",
            "bbox": (-15, 45, -30, 30),
        },
    },
}

RACING_WITH_DUCKS = deepcopy(RACING)
RACING_WITH_DUCKS["problem"]["type"] = "racing_duck"
RACING_WITH_DUCKS["sensors"]["ducks_set"] = {
    "type": "duckmap",
    "observe_heading": True,
    "bbox": (-15, 45, -30, 30),
}


_STANDARD_ENVS = {
    "parking": PARALLEL_PARKING,
    "lidar_parking": LIDAR_PARALLEL_PARKING,
    "resetless_lidar_parking": RESETLESS_LIDAR_PARALLEL_PARKING,
    "racing": RACING,
    "racing_duck": RACING_WITH_DUCKS,
}


def get_standard_env_config(name):
    return deepcopy(_STANDARD_ENVS[name])


def get_standard_env_names():
    return list(_STANDARD_ENVS.keys())


# Gym Registry
_REGISTRY = {
    "CarEnv-Racing-v1": {"config": get_standard_env_config("racing")},
    "CarEnv-Parking-v1": {"config": get_standard_env_config("parking")},
    "CarEnv-LidarParking-v1": {"config": get_standard_env_config("lidar_parking")},
    "CarEnv-ResetlessLidarParking-v1": {
        "config": get_standard_env_config("resetless_lidar_parking")
    },
    "CarEnv-Racing-Ducks-v1": {"config": get_standard_env_config("racing_duck")},
}


def _register():
    import gymnasium

    for k, v in _REGISTRY.items():
        gymnasium.register(k, "CarEnv.Env:CarEnv", kwargs=v)


_register()


def get_registered():
    return list(_REGISTRY.keys())
