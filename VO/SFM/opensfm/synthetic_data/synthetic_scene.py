from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import functools

from opensfm import types

import opensfm.synthetic_data.synthetic_metrics as sm
import opensfm.synthetic_data.synthetic_generator as sg


def get_camera(type, id, focal, k1, k2):
    camera = None
    if type == 'perspective':
        camera = types.PerspectiveCamera()
    if type == 'fisheye':
        camera = types.FisheyeCamera()
    camera.id = id
    camera.focal = focal
    camera.k1 = k1
    camera.k2 = k2
    camera.focal_prior = camera.focal
    camera.k1_prior = camera.k1
    camera.k2_prior = camera.k2
    camera.height = 1600
    camera.width = 2000
    return camera


def get_scene_generator(type, length):
    generator = None
    if type == 'ellipse':
        ellipse_ratio = 4
        generator = functools.partial(sg.ellipse_generator, length,
                                      length / ellipse_ratio)
    if type == 'line':
        generator = functools.partial(sg.line_generator, length)
    if type == 'curve':
        generator = functools.partial(sg.weird_curve, length)
    return generator


class SyntheticScene(object):
    def __init__(self, generator):
        self.generator = generator
        self.wall_points = None
        self.floor_points = None
        self.width = None
        self.shot_positions = []
        self.shot_rotations = []
        self.cameras = []

    def add_street(self, points_count, height, width):
        self.wall_points, self.floor_points = sg.generate_street(
            sg.samples_generator_random_count(
                int(points_count // 3)), self.generator,
            height, width)
        self.width = width
        return self

    def perturb_walls(self, walls_pertubation):
        sg.perturb_points(self.wall_points, walls_pertubation)
        return self

    def perturb_floor(self, floor_pertubation):
        sg.perturb_points(self.floor_points, floor_pertubation)
        return self

    def add_camera_sequence(self, camera, start, length, height, interval,
                            position_noise=None, rotation_noise=None,
                            gps_noise=None):
        default_noise_interval = 0.25*interval
        positions, rotations = sg.generate_cameras(
            sg.samples_generator_interval(start, length, interval,
                                          default_noise_interval),
            self.generator, height)
        sg.perturb_points(positions, position_noise)
        sg.perturb_rotations(rotations, rotation_noise)
        self.shot_rotations.append(rotations)
        self.shot_positions.append(positions)
        self.cameras.append(camera)
        return self

    def get_reconstruction(self):
        floor_color = [120, 90, 10]
        wall_color = [10, 90, 130]
        return sg.create_reconstruction(
            [self.floor_points, self.wall_points],
            [floor_color, wall_color],
            self.cameras, self.shot_positions,
            self.shot_rotations)

    def get_scene_exifs(self, gps_noise):
        return sg.generate_exifs(self.get_reconstruction(),
                                 gps_noise)

    def get_tracks_data(self, maximum_depth, noise):
        return sg.generate_track_data(self.get_reconstruction(),
                                      maximum_depth, noise)

    def compare(self, reconstruction):
        reference = self.get_reconstruction()
        position = sm.position_errors(reference, reconstruction)
        gps = sm.gps_errors(reconstruction)
        rotation = sm.rotation_errors(reference, reconstruction)
        points = sm.points_errors(reference, reconstruction)
        completeness = sm.completeness_errors(reference, reconstruction)
        return {
            'position_average': np.linalg.norm(np.average(position, axis=0)),
            'position_std': np.linalg.norm(np.std(position, axis=0)),
            'gps_average': np.linalg.norm(np.average(gps, axis=0)),
            'gps_std': np.linalg.norm(np.std(gps, axis=0)),
            'rotation_average': np.average(rotation),
            'rotation_std': np.std(rotation),
            'points_average': np.linalg.norm(np.average(points, axis=0)),
            'points_std': np.linalg.norm(np.std(points, axis=0)),
            'ratio_cameras': completeness[0],
            'ratio_points': completeness[1]
        }
