import sys
import os
from os.path import dirname, exists, join

import numpy as np
import pinocchio
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.rpy import matrixToRpy, rpyToMatrix, rotate
from robot_properties_solo.config import SoloConfig

def m2a(m): return np.array(m.flat)
def a2m(a): return np.matrix(a).T

def readParamsFromSrdf(robot, SRDF_PATH, verbose, has_rotor_parameters=True, referencePose='standing'):
    rmodel = robot.model

    if has_rotor_parameters:
        pinocchio.loadRotorParameters(rmodel, SRDF_PATH, verbose)
    rmodel.armature = np.multiply(rmodel.rotorInertia.flat, np.square(rmodel.rotorGearRatio.flat))
    pinocchio.loadReferenceConfigurations(rmodel, SRDF_PATH, verbose)
    if referencePose is not None:
        robot.q0.flat[:] = rmodel.referenceConfigurations[referencePose].copy()
    return

def addFreeFlyerJointLimits(robot):
    rmodel = robot.model

    ub = rmodel.upperPositionLimit
    ub[:7] = 1
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1
    rmodel.lowerPositionLimit = lb
    # print(rmodel.upperPositionLimit, rmodel.lowerPositionLimit)

def loadHumanoidBall():
    vec2list = lambda m: np.array(m.T).reshape(-1)
    pack_path = SoloConfig.packPath
    urdf_path = SoloConfig.urdf_path_pin_ball
    srdf_path = os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf/humanoid_throw.srdf'
    package_dirs = [os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf']
    robot = SoloConfig.buildHumanoidBallWrapper()
    readParamsFromSrdf(robot,srdf_path, False)
    assert ((robot.model.armature[:6] == 0.).all())
    addFreeFlyerJointLimits(robot)
    return robot

def loadHumanoidWoBall():
    vec2list = lambda m: np.array(m.T).reshape(-1)
    pack_path = SoloConfig.packPath
    urdf_path = SoloConfig.urdf_path_pin_wo_ball
    srdf_path = os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf/humanoid_throw.srdf'
    package_dirs = [os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf']
    robot = SoloConfig.buildHumanoidWoBallWrapper()
    readParamsFromSrdf(robot,srdf_path, False)
    assert ((robot.model.armature[:6] == 0.).all())
    addFreeFlyerJointLimits(robot)
    return robot
def loadHumanoidBullet():
    vec2list = lambda m: np.array(m.T).reshape(-1)
    pack_path = SoloConfig.packPath
    urdf_path = SoloConfig.urdf_path
    srdf_path = os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf/humanoid_throw.srdf'
    package_dirs = [os.path.dirname(
        os.path.dirname(urdf_path)) + '/urdf']
    robot = SoloConfig.buildRobotWrapper()
    readParamsFromSrdf(robot,srdf_path, False)
    assert ((robot.model.armature[:6] == 0.).all())
    addFreeFlyerJointLimits(robot)
    return robot