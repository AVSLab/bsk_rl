from Basilisk.simulation import reactionWheelStateEffector, thrusterDynamicEffector
from Basilisk.utilities import simIncludeRW, simIncludeThruster
from numpy.random import uniform


def balancedHR16Triad(
    useRandom=False, randomBounds=(-400, 400), wheelSpeeds=[500, 500, 500]
):
    """
    Creates a set of three HR16 reaction wheels.
    Returns a set of thrusters and thrusterFac instance to add thrusters to a
    spacecraft.
    :return thrusterSet: thruster dynamic effector instance
    :return thrusterFac: factory containing defined thrusters
    """
    rwFactory = simIncludeRW.rwFactory()
    if useRandom:
        wheelSpeeds = uniform(randomBounds[0], randomBounds[1], 3)

    rwFactory.create(
        "Honeywell_HR16", [1, 0, 0], maxMomentum=50.0, Omega=wheelSpeeds[0]  # RPM
    )
    rwFactory.create(
        "Honeywell_HR16", [0, 1, 0], maxMomentum=50.0, Omega=wheelSpeeds[1]  # RPM
    )
    rwFactory.create(
        "Honeywell_HR16", [0, 0, 1], maxMomentum=50.0, Omega=wheelSpeeds[2]  # RPM
    )

    rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()

    return rwStateEffector, rwFactory, wheelSpeeds


def idealMonarc1Octet():
    """
    Creates a set of eight ADCS thrusters using MOOG Monarc-1 attributes.
    Returns a set of thrusters and thrusterFac instance to add thrusters to a
    spacecraft.
    :return thrusterSet: thruster dynamic effector instance
    :return thrusterFac: factory containing defined thrusters
    """
    location = [
        [3.874945160902288e-2, -1.206182747348013, 0.85245],
        [3.874945160902288e-2, -1.206182747348013, -0.85245],
        [-3.8749451609022656e-2, -1.206182747348013, 0.85245],
        [-3.8749451609022656e-2, -1.206182747348013, -0.85245],
        [-3.874945160902288e-2, 1.206182747348013, 0.85245],
        [-3.874945160902288e-2, 1.206182747348013, -0.85245],
        [3.8749451609022656e-2, 1.206182747348013, 0.85245],
        [3.8749451609022656e-2, 1.206182747348013, -0.85245],
    ]

    direction = [
        [-0.7071067811865476, 0.7071067811865475, 0.0],
        [-0.7071067811865476, 0.7071067811865475, 0.0],
        [0.7071067811865475, 0.7071067811865476, 0.0],
        [0.7071067811865475, 0.7071067811865476, 0.0],
        [0.7071067811865476, -0.7071067811865475, 0.0],
        [0.7071067811865476, -0.7071067811865475, 0.0],
        [-0.7071067811865475, -0.7071067811865476, 0.0],
        [-0.7071067811865475, -0.7071067811865476, 0.0],
    ]
    thrusterSet = thrusterDynamicEffector.ThrusterDynamicEffector()
    thFactory = simIncludeThruster.thrusterFactory()
    for pos_B, dir_B in zip(location, direction):
        thFactory.create("MOOG_Monarc_1", pos_B, dir_B)
    return thrusterSet, thFactory
