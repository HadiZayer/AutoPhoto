import habitat_sim
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import imageio
from pygifsicle import optimize

import numpy as np
import attr
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

import os
import glob
import pickle
import enum
import tqdm

@enum.unique
class Actions(enum.Enum):
    FORWARD = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2
    TERMINATE = 3
    MOVE_BACK = 4
    MOVE_RIGHT = 5
    MOVE_LEFT = 6
    GAUSSIAN_FORWARD = 7
    GAUSSIAN_LEFT = 8
    GAUSSIAN_RIGHT = 9
    SMALL_LEFT = 10
    SMALL_RIGHT = 11
    LARGE_LEFT = 12
    LARGE_RIGHT = 13

# First, define a class to keep the parameters of the control
# @attr.s is just syntatic sugar for creating these data-classes
@attr.s(auto_attribs=True, slots=True)
class MoveAndSpinSpec:
    forward_amount: float
    spin_amount: float


def label_img(img, dataset, action_name, res):
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("Helvetica.ttf", 40)
    small_font = ImageFont.truetype("Helvetica.ttf", 30)
    # draw.text((x, y),"Sample Text",(r,g,b))
    # draw.text((0, 0),"Sample Text",(255,255,255),font=font)
    # draw.text((0, 112),"Sample Text",(255,255,255),font=font)
    # draw.text((0, 224),"Sample Text",(255,255,255),font=font)
    draw.text((20, res - 80), action_name, (255, 255, 255, 200), font=font)
    draw.text((20, res - 40), dataset, (255, 255, 255, 200), font=small_font)

    return img


def apply_actions(images, sim, splits, label, dataset, action, obs, res, current_state, next_state
                  , init=False, client=None):
    if label == 'Capture':
        # do nothing
        rgb_img = Image.fromarray(obs["color_sensor"], mode="RGBA")
        rgb_img = label_img(rgb_img, dataset, label, res)
        for i in range(splits*2):
            images.append(rgb_img)
    else:
        if init:
            rgb_img = Image.fromarray(obs["color_sensor"], mode="RGBA")
            rgb_img = label_img(rgb_img, dataset, label, res)
            for i in range(splits):
                images.append(rgb_img)

        p1 = current_state.position
        p2 = next_state.position
        q1 = current_state.rotation
        q2 = next_state.rotation
        for i in range(1, splits+1):
            rgb_img = Image.fromarray(obs["color_sensor"], mode="RGBA")
            rgb_img = label_img(rgb_img, dataset, label, res)
            images.append(rgb_img)

            t = float(i)/float(splits)
            # print('t is ', t)
            p = (1.0-t) * p1 + t * p2
            q = habitat_sim.utils.common.quaternion.slerp(q1, q2, 0, 1.0001, t)

            agent_state = client.get_agent_state()
            previous_quat = agent_state.rotation

            agent_state.position = p
            agent_state.rotation = q
            client.set_agent_state(agent_state)

            # obs = sim.step(action)
            agent_state = client.get_agent_state()
            current_quat = agent_state.rotation
            obs = sim.get_sensor_observations()
            angle = habitat_sim.utils.common.angle_between_quats(previous_quat, current_quat)
            # print('angle change is {}'.format(angle))
            # print('angle change in degree is {}'.format(np.rad2deg(angle)))
    return obs

# Register the control functor
# This action will be an action that effects the body, so body_action=True
@habitat_sim.registry.register_move_fn(body_action=True)
class MoveForwardAndSpin(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node: habitat_sim.SceneNode, actuation_spec: MoveAndSpinSpec
    ):
        forward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.FRONT
        )
        if actuation_spec.forward_amount > 0.01:
            gaussian_forward = np.random.normal(actuation_spec.forward_amount, abs(actuation_spec.forward_amount/2))
            scene_node.translate_local(forward_ax * gaussian_forward)

        if actuation_spec.spin_amount > 0.01:
            # Rotate about the +y (up) axis
            rotation_ax = habitat_sim.geo.UP
            abs_deg = min(360-actuation_spec.spin_amount, actuation_spec.spin_amount)
            gaussian_spin = np.random.normal(actuation_spec.spin_amount, abs(abs_deg/2))
            scene_node.rotate_local(mn.Deg(gaussian_spin), rotation_ax)
            # Calling normalize is needed after rotating to deal with machine precision errors
            scene_node.rotation = scene_node.rotation.normalized()


class HabitatClient:

    def __init__(self, path_generator, multi_movement, noisy, res=224, splits=1):
        self.path_generator = path_generator
        self.sim = None
        self.agent = None
        self.res = res
        self.splits = splits

        self.actions = [Actions.FORWARD, Actions.TURN_LEFT, Actions.TURN_RIGHT, Actions.TERMINATE]

        assert not (noisy and multi_movement)

        if multi_movement == 'back':
            self.actions += [Actions.MOVE_BACK]
        elif multi_movement == 'fine_turns':
            self.actions += [Actions.MOVE_BACK, Actions.SMALL_LEFT, Actions.SMALL_RIGHT, Actions.LARGE_LEFT, Actions.LARGE_RIGHT]

        if noisy:
            self.actions += [Actions.GAUSSIAN_FORWARD, Actions.GAUSSIAN_LEFT, Actions.GAUSSIAN_RIGHT]

    """
    @:returns: (Terminate_flag, view) True if the action terminates the episode, False otherwise
    and returns the view if False, None otherwise
    """
    def take_action(self, action_id):
        action = self.actions[action_id]
        if action == Actions.FORWARD:
            action = 'move_forward'
            # print("action: FORWARD")
        elif action == Actions.TURN_LEFT:
            action = 'turn_left'
            # print("action: LEFT")
        elif action == Actions.TURN_RIGHT:
            action = 'turn_right'
            # print("action: RIGHT")
        elif action == Actions.TERMINATE:
            return True, None
        elif action == Actions.MOVE_BACK:
            action = 'move_backward'
        elif action == Actions.MOVE_RIGHT:
            action = 'move_right'
        elif action == Actions.MOVE_LEFT:
            action = 'move_left'
        elif action == Actions.GAUSSIAN_FORWARD:
            action = 'gaussian_forward'
        elif action == Actions.GAUSSIAN_LEFT:
            action = 'gaussian_turn_left'
        elif action == Actions.GAUSSIAN_RIGHT:
            action = 'gaussian_turn_right'
        elif action == Actions.SMALL_LEFT:
            action = 'small_turn_left'
        elif action == Actions.SMALL_RIGHT:
            action = 'small_turn_right'
        elif action == Actions.LARGE_LEFT:
            action = 'large_turn_left'
        elif action == Actions.LARGE_RIGHT:
            action = 'large_turn_right'
        else:
            raise ValueError("action is not valid")

        observations = self.sim.step(action)
        rgb_img = Image.fromarray(observations['color_sensor'], mode="RGBA").convert('RGB')
        return False, rgb_img

    # def undo_action(self, action_id):
    #     action = self.actions[action_id]
    #     if action == Actions.FORWARD:
    #         action = 'move_backward'
    #         # print("action: FORWARD")
    #     elif action == Actions.TURN_LEFT:
    #         action = 'turn_right'
    #         # print("action: LEFT")
    #     elif action == Actions.TURN_RIGHT:
    #         action = 'turn_left'
    #         # print("action: RIGHT")
    #     else:
    #         raise ValueError("action is not valid to undo")
    #
    #     observations = self.sim.step(action)
    #     rgb_img = Image.fromarray(observations['color_sensor'], mode="RGBA").convert('RGB')
    #     return False, rgb_img

    def load_scene(self, scene_name):
        scene_path = self.path_generator(scene_name)  # os.path.join(self.scenes_dir, '{}.glb'.format(scene_name))

        sim_settings = {
            "width": self.res,  # Spatial resolution of the observations
            "height": self.res,
            "scene": scene_path,  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": True,  # RGB sensor
            "semantic_sensor": False,  # Semantic sensor
            "depth_sensor": False,  # Depth sensor
            "seed": 12345,
        }
        cfg = self.make_cfg(sim_settings)

        if self.sim:
            self.sim.close()

        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])

    """
        resets the agent to a random position and orientation within the same scene
        @:return: np array of the view
    """
    def reset_same_scene(self, theta=None):
        agent_state = self.agent.get_state()
        if theta is None:
            theta = np.random.rand() * 2.0 * np.pi
        rotation = habitat_sim.utils.common.quat_from_angle_axis(theta, np.array([0, 1, 0]))

        agent_state.rotation = rotation
        agent_state.position = self.sim.pathfinder.get_random_navigable_point()
        self.agent.set_state(agent_state)
        observations = self.sim.get_sensor_observations()
        rgb_img = Image.fromarray(observations['color_sensor'], mode="RGBA").convert('RGB')
        return rgb_img

    def get_agent_state(self):
        return self.agent.get_state()

    def set_agent_state(self, state):
        self.agent.set_state(state)

    def get_agent_position(self):
        agent_state = self.agent.get_state()
        return agent_state.position

    def get_view(self):
        observations = self.sim.get_sensor_observations()
        rgb_img = Image.fromarray(observations['color_sensor'], mode="RGBA").convert('RGB')
        return rgb_img

    def make_cfg(self, settings):
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene.id = settings["scene"]

        # Note: all sensors must have the same resolution
        sensors = {
            "color_sensor": {
                "sensor_type": habitat_sim.SensorType.COLOR,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "depth_sensor": {
                "sensor_type": habitat_sim.SensorType.DEPTH,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
            "semantic_sensor": {
                "sensor_type": habitat_sim.SensorType.SEMANTIC,
                "resolution": [settings["height"], settings["width"]],
                "position": [0.0, settings["sensor_height"], 0.0],
            },
        }

        sensor_specs = []
        for sensor_uuid, sensor_params in sensors.items():
            if settings[sensor_uuid]:
                sensor_spec = habitat_sim.SensorSpec()
                sensor_spec.uuid = sensor_uuid
                sensor_spec.sensor_type = sensor_params["sensor_type"]
                sensor_spec.resolution = sensor_params["resolution"]
                sensor_spec.position = sensor_params["position"]

                sensor_specs.append(sensor_spec)

        # Here you can specify the amount of displacement in a forward action and the turn angle
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_right": habitat_sim.agent.ActionSpec(
                "move_right", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "move_left": habitat_sim.agent.ActionSpec(
                "move_left", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
            ),
            "small_turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "small_turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "large_turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "large_turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=90.0)
            ),
            "gaussian_forward": habitat_sim.ActionSpec(
            "move_forward_and_spin", MoveAndSpinSpec(0.25, 0.0)
            ),
            "gaussian_turn_right": habitat_sim.ActionSpec(
            "move_forward_and_spin", MoveAndSpinSpec(0.0, 360.0-30.0)
            ),
            "gaussian_turn_left": habitat_sim.ActionSpec(
            "move_forward_and_spin", MoveAndSpinSpec(0.0, 30.0)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


if __name__ == '__main__':
    data_dir = '/phoenix/S7/ha366/habitat_data/gibson'

    path_generator = lambda scene: os.path.join(data_dir, '{}.glb'.format(scene))
    dirs = glob.glob(os.path.join(data_dir, '*'))
    dir_names = [dir_name.split('/')[-1] for dir_name in dirs]

    path_generator = lambda scene: os.path.join(data_dir, scene, '{}.glb'.format(scene))

    storage_dir = 'sampling_maps'
    os.makedirs(storage_dir, exist_ok=True)

    client = HabitatClient(path_generator, multi_movement='fine_turns', noisy=False, res=224)

    num_samples = 5000

    from scoring_model import ScoringModel
    import matplotlib.pyplot as plt
    scoring_model = ScoringModel()

    samples = dict()

    thetas = np.linspace(0,1,9)


    for scene in dir_names:
        client.load_scene(scene_name=scene)

        sample_views = []
        scene_min, scene_max = None, None

        for theta in thetas:
            state_score_pairs = []
            for i in tqdm.tqdm(range(num_samples)):
                view = client.reset_same_scene(theta=theta * 2.0 * np.pi)
                # if i < 10:
                #     sample_views.append(view)
                #     view.save(os.path.join(storage_dir, 'scene_{}_{}.jpg'.format(scene, i)))
                state = client.get_agent_state()
                view = scoring_model.transform(view)
                score = scoring_model.forward([view])
                state_score_pairs.append((state, score))

            # visualize positions
            positions = [state.position for state, score in state_score_pairs]
            positions = np.array(positions)
            scores = [score.item() for state, score in state_score_pairs]

            scores = np.array(scores)

            if scene_min is None:
                std = np.std(scores)
                scene_min = np.min(scores) - std * 0.5
                scene_max = np.max(scores) + std * 0.5
                print(f'scene min {scene_min}, scene max {scene_max}, std {std}')

            scores = np.clip(scores, scene_min, scene_max)

            scores -= scene_min
            scores /= (scene_max - scene_min)

            samples[scene] = [(state.position, score) for state, score in state_score_pairs]

            plt.scatter(positions[:, 0], positions[:, 2], c=scores, alpha=0.5)
            plt.savefig(os.path.join(storage_dir, '{}_theta_{}.png'.format(scene, theta)))
            plt.close()
   
        with open(os.path.join(storage_dir,'samples.pkl'), 'wb') as f:
            pickle.dump(samples, f)


