from habitat_client import HabitatClient
from scoring_model import ScoringModel
from instafilter import Instafilter

import gym
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import habitat_sim
import cv2

import os
import pickle
import time


class HabitatEnv(gym.Env):

    def __init__(self, scenes_stats_path, path_generator, scene_change_freq, random_seed, samples_path, single_scene=None, gpu=None,
                 scaling_std_steps=None, feed_forward=False, ordering=(1,1),multi_movement=False, res=224, noisy=False,
                 threshold_factor=1.0, mixed_state=False, multi_features=False, use_filters=False):
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu)

        with open(scenes_stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.client = HabitatClient(path_generator=path_generator, multi_movement=multi_movement, res=res, noisy=noisy)
        self.noisy = noisy
        self.threshold_factor = threshold_factor
        self.scoring_model = ScoringModel()
        self.transform = self.scoring_model.transform

        if not self.noisy:
            self.action_space = gym.spaces.Discrete(len(self.client.actions))
        else:
            self.action_space = gym.spaces.Discrete(4)

        self.feed_forward = feed_forward
        self.mixed_state = mixed_state

        self.multi_features = multi_features

        if self.feed_forward:
            if self.multi_features:
                feature_size = self.scoring_model.get_multi_features(torch.rand(3, 224, 224)).shape[0]
            else:
                feature_size = self.scoring_model.get_features(torch.rand(3, 224, 224)).shape[0]
            features_space = gym.spaces.Box(low=0.0, high=1.0, shape=(feature_size,))
            if mixed_state:
                img_space = gym.spaces.Box(low=0.0, high=1.0, shape=(224, 224, 4))
                self.observation_space = img_space
            else:
                self.observation_space = features_space
        else:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(224, 224, 3))

        self.timestep = 0
        self.change_freq = scene_change_freq
        self.num_episodes = 0
        # self.scene = 'Elmira'

        self.gamma = 0.9999
        self.global_time = 0

        self.max_episode_size = 100

        self.id, self.num_envs = ordering

        self.running_acc = 0
        self.acc_timesteps = 0
        self.acc_period = 50

        self.current_score = None
        self.previous_score = None

        self.scenes = list(self.stats.keys())
        self.num_scene_changes = 0
        np.random.seed(random_seed)
        np.random.shuffle(self.scenes)

        if single_scene:
            self.scenes = [single_scene]

        self.scene_index = self.id
        self.scene = self.scenes[self.scene_index]

        self.scaling_std_steps = scaling_std_steps

        with open(samples_path, 'rb') as f:
            self.samples = pickle.load(f)

        # knn 5 percent of samples
        self.knn = int(0.15 * self.samples[self.scene][0].shape[0])
        self.init_pos = None
        self.init_quat = None
        self.init_score = None

        # load initial scene
        self.init_view = None
        self.view = None
        self.client.load_scene(self.scene)

        self.seeded_state = None

        self.use_filters = use_filters
        self.filter = None

        if self.use_filters:
            self.filter_options = list(self.samples[self.scene][1].keys())

            self.filters_dict = dict()
            for filter_name in self.filter_options:
                if filter_name == 'no_filter':
                    continue
                self.filters_dict[filter_name] = Instafilter(filter_name)

    def set_scene(self, scene):
        assert scene in self.scenes
        for idx, s in enumerate(self.scenes):
            if s == scene:
                self.scene_index = idx
        self.scene = scene
        self.client.load_scene(scene)


    def reset(self, state=None):
        if (self.num_episodes+1) % self.change_freq == 0:
            next_index = (self.scene_index+self.num_envs)
            if self.num_scene_changes % len(self.scenes) == 0:
                np.random.shuffle(self.scenes)
            # next_index = np.random.randint(len(self.scenes))
            self.scene_index = next_index % len(self.scenes)
            self.scene = self.scenes[self.scene_index]
            self.client.load_scene(self.scene)
            self.acc_timesteps = 0
            self.running_acc = 0
            self.num_scene_changes += 1

        self.timestep = 0
        self.num_episodes += 1

        if self.use_filters:
            coin = np.random.randint(2)
            if coin == 0:
                filter_index = np.random.randint(len(self.filter_options))
                self.filter = self.filter_options[filter_index]
            else:
                self.filter = 'no_filter'

        # resample if view score < global_mean + global_std
        if state:
            self.client.set_agent_state(state)
            view = self.client.get_view()

            if self.use_filters:
                view = self.apply_filter(view, self.filter)

            self.view = view

            self.previous_score = self.compute_score(view)
            self.current_score = self.previous_score
            _, scores = self.samples[self.scene]
        elif self.seeded_state:
            self.client.set_agent_state(self.seeded_state)
            view = self.client.get_view()
            self.view = view

            self.previous_score = self.compute_score(view)
            self.current_score = self.previous_score
            _, scores = self.samples[self.scene]
            self.seeded_state = None
        else:
            for i in range(10):
                view = self.client.reset_same_scene()

                if self.use_filters:
                    view = self.apply_filter(view, self.filter)

                self.view = view


                self.previous_score = self.compute_score(view)
                self.current_score = self.previous_score

                if self.use_filters:
                    break

                _, scores = self.samples[self.scene]

                scene_mean = np.mean(scores)
                scene_std = np.std(scores)

                # if self.current_score > scene_mean - scene_std:
                break
                #otherwise want to resample -- sample atmost 10 times
        self.init_view = self.client.get_view()
        agent_state = self.client.get_agent_state()
        self.init_pos = agent_state.position
        self.init_quat = agent_state.rotation
        self.init_score = self.current_score

        return self.to_tf(view)

    def apply_filter(self, img, filter_name):
        if filter_name == "no_filter":
            return img
        original_img = np.array(img)[:, :, ::-1]
        # img = self.filters_dict[filter_name](original_img, is_RGB=False)
        view = Image.fromarray(original_img[:, :, ::-1]).convert('RGB')
        return view

    def to_tf(self, view):
        # view = view.resize((224, 224), Image.NEAREST)
        if self.feed_forward:
            t_view = self.transform(view)
            if self.multi_features:
                features = self.scoring_model.get_multi_features(t_view)
            else:
                features = self.scoring_model.get_features(t_view)

            if self.mixed_state:
                buffer = np.zeros((224,224,4))
                np_view = np.array(view) / 255
                buffer[:,:,:3] = np_view

                features_left = features.shape[0]
                assert features_left == 512

                buffer[:,0,3] = features[:224]
                features_left -= 224
                buffer[:, 1, 3] = features[224:2*224]
                features_left -= 224
                buffer[:features_left, 2, 3] = features[2*224:]
                features_left -= len(features[2*224:])

                assert features_left == 0

                return buffer
            else:
                return features
        else:
            # view = view.transpose(0, 2)
            # view = view.transpose(0, 1)
            view = np.array(view).astype(float)
            # print(view)
            # print(view.shape)
            # to tf processing
            return view

    def compute_score(self, view):
        view = self.transform(view)
        score = self.scoring_model.forward([view]).item()

        return score

    def step(self, action_idx):
        if action_idx != 3 and self.noisy:
            # execute half command with probability 50 percent
            action_idx += 4
        done, view = self.client.take_action(action_id=action_idx)
        if view:
            if self.use_filters:
                view = self.apply_filter(view, self.filter)
            self.view = view
        info = dict()

        if done:
            # self.view = self.client.get_view()
            # self.current_score = self.compute_score(self.view)

            self.acc_timesteps += 1
            reward = self.distance_based_terminal_reward()
            info['high_reward'] = reward > 0.99
            info['better_than_init'] = self.current_score > self.init_score
            info['match'] = False
            info['score_diff'] = self.current_score - self.init_score
            info['init_score'] = self.init_score
            info['final_score'] = self.current_score

            # reward += penalty

            # if match:
            #     reward = reward
            # else:
            #     reward = -1.0


            if info['high_reward']:
                self.running_acc += 1

            if self.acc_timesteps % self.acc_period == 0:
                print('scene {}, running acc {}'.format(self.scene, self.running_acc / self.acc_period))
                self.running_acc = 0

        else:
            if self.timestep > self.max_episode_size:
                print('running for too long')
                done = True
                info['high_reward'] = False
                reward = -1
                info['init_score'] = self.init_score
                info['final_score'] = self.current_score
                info['better_than_init'] = self.current_score > self.init_score
            else:
                self.current_score = self.compute_score(view)
                reward = self.step_reward()
                self.previous_score = self.current_score

            view = self.to_tf(view)

        return view, reward, done, info

    def step_reward(self):
        # decaying reward
        exploration_reward = self.gamma ** (self.global_time) * 0.1
        time_penalty = self.timestep * -0.005

        self.global_time += 1
        self.timestep += 1

        score_grad = self.current_score - self.previous_score

        return time_penalty + exploration_reward + score_grad

    def terminal_reward(self):
        # compute current view score
        if self.scaling_std_steps:
            factor = min(1.0, self.global_time/self.scaling_std_steps)
        else:
            factor = 1.0

        threshold = self.stats[self.scene]['mean'] + self.stats[self.scene]['std'] * factor
        if self.current_score > threshold:
            # print('high reward')
            return 1.0
        else:
            # print('penalty')
            return -1.0

    def relative_terminal_reward(self):
        # compute nearest neighbors
        if self.use_filters:
            scores = self.samples[self.scene][1][self.filter]
            positions = self.samples[self.scene][0]
        else:
            positions, scores = self.samples[self.scene]

        dists = positions-self.init_pos
        dists = np.linalg.norm(dists, axis=-1)
        indices = list(range(len(dists)))

        indices = sorted(indices, key=lambda idx: dists[idx])
        indices = np.array(indices).astype(np.int)

        closest_scores = scores[indices[:self.knn]]

        mean = np.mean(closest_scores)
        std = np.std(closest_scores)

        if self.current_score > (mean + self.threshold_factor * std):
            # print('high reward')
            return 1.0
        else:
            # print('penalty')
            return -1.0

    def distance_based_terminal_reward(self):
        # compute nearest neighbors
        if self.use_filters:
            scores = self.samples[self.scene][1][self.filter]
            positions = self.samples[self.scene][0]
        else:
            positions, scores = self.samples[self.scene]

        dists = positions-self.init_pos
        dists = np.linalg.norm(dists, axis=-1)
        indices = list(range(len(dists)))

        indices = sorted(indices, key=lambda idx: dists[idx])
        indices = np.array(indices).astype(np.int)
        max_idx = 0

        # for i in range(10):
        #     print('i: {} index: {} dist: {}'.format(i, indices[i], dists[indices[i]]))

        for i in range(len(indices)):
            if dists[indices[i]] < 2:
                max_idx = i

        # print('have {} samples out of {}'.format(max_idx, len(indices)))

        closest_scores = scores[indices[:max_idx]]

        mean = np.mean(closest_scores)
        std = np.std(closest_scores)

        if self.current_score > (mean + self.threshold_factor * std):
            # print('high reward')
            return 1.0
        else:
            # print('penalty')
            return -1.0

    def relative_rotational_terminal_reward(self):
        # compute nearest neighbors
        relative_score = self.relative_terminal_reward()
        current_quat = self.client.get_agent_state().rotation

        angle = habitat_sim.utils.common.angle_between_quats(self.init_quat, current_quat)
        angle = np.rad2deg(angle)
        assert 0 <= angle <= 180

        rot_penalty = -angle/180.0

        return relative_score, rot_penalty

    def relative_terminal_reward_feature_match(self):
        # compute nearest neighbors
        relative_score = self.relative_terminal_reward()
        num_features = self.feature_match(self.init_view, self.client.get_view())

        match = num_features >= 5

        return relative_score, match

    def sample_views(self, n, scene=None, filter_dict=None):
        if scene:
            self.client.load_scene(scene)

        positions = []
        scores = []
        score_dict = dict()

        if filter_dict:

            for filter_name in filter_dict:
                score_dict[filter_name] = []
            score_dict['no_filter'] = []

        init_time = time.time()
        for i in range(n):
            view = self.client.reset_same_scene()
            if filter_dict:

                original_img = np.array(view)[:, :, ::-1]
                for filter_name in filter_dict:
                    img = filter_dict[filter_name](original_img, is_RGB=False)
                    img = Image.fromarray(img[:, :, ::-1]).convert('RGB')
                    sampled_score = self.compute_score(img)
                    score_dict[filter_name].append(sampled_score)
                sampled_score = self.compute_score(view)
                score_dict['no_filter'].append(sampled_score)
                end_time = time.time()
                if (i+1) % 200 == 0:
                    print('took {} to finish 200 samples filters'.format(end_time-init_time))
                    init_time = time.time()
            else:
                sampled_score = self.compute_score(view)
                scores.append(sampled_score)

            position = self.client.get_agent_position()
            positions.append(position)

        positions = np.array(positions)

        if filter_dict:
            for filter_name in (list(filter_dict)+['no_filter']):
                score_dict[filter_name] = np.array(score_dict[filter_name])
            scores = score_dict
        else:
            scores = np.array(scores)
        # scores = np.array(scores)

        return positions, scores

    def get_num_matching(self, samples):
        positions = []
        scores = []
        k = int(0.05 * len(samples[self.scene]))
        print('value for k is', k)
        for i in range(len(samples[self.scene])):
            pos, s, _ = samples[self.scene][i]
            positions.append(pos)
            scores.append(s)

        positions = np.array(positions)
        scores = np.array(scores)

        dists = positions - self.init_pos
        dists = np.linalg.norm(dists, axis=-1)
        indices = list(range(len(dists)))

        indices = sorted(indices, key=lambda idx: dists[idx])
        indices = np.array(indices).astype(np.int)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        img1 = cv2.cvtColor(np.array(self.view), cv2.COLOR_RGB2BGR)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)

        num_matches = 0

        for i in indices[:k]:
            _, _, des2 = samples[self.scene][i]
            num_feature_matched = self.feature_match_using_des(des1, des2)
            if num_feature_matched >= 10:
                num_matches += 1



        print('num matches', num_matches)

    @staticmethod
    def feature_match(img1, img2):
        img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if len(kp1) < 2 or len(kp2) < 2:
            return 0

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return len(good)

    @staticmethod
    def feature_match_using_des(des1, des2):

        if len(des1) < 2 or len(des2) < 2:
            return 0

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        return len(good)

if __name__ == '__main__':

    # data_dir = '/phoenix/S7/ha366/habitat_data/gibson'
    # scene_stats_path = 'finetuned_model_habitat_stats/train_stats.pkl'
    # path_generator = lambda scene: os.path.join(data_dir, '{}.glb'.format(scene))
    # samples_path = 'full_samples.pkl'
    # feed_forward = True

    data_dir = '/phoenix/S7/ha366/Replica-Dataset/replica_v1'
    scene_stats_path = 'finetuned_model_habitat_stats/train_stats.pkl'
    path_generator = lambda scene: os.path.join(data_dir, '{}/habitat/mesh_semantic.ply'.format(scene))
    samples_path = 'full_samples.pkl'
    feed_forward = True

    print('--------creating env---------')

    env = HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, scene_change_freq=100000,
                      random_seed=1111,
                      feed_forward=feed_forward, samples_path=samples_path)

    # scenes = ['room_0', 'hotel_0', 'frl_apartment_0', 'apartment_0', 'office_0']
    scenes = ['apartment_0', 'apartment_1', 'apartment_2', 'room_0', 'room_1', 'room_2', 'hotel_0', 'frl_apartment_0', 'frl_apartment_1', 'frl_apartment_2', 'frl_apartment_3', 'frl_apartment_4', 'frl_apartment_5', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4']
    stats_dict = dict()
    for scene in scenes:
        positions, scores = env.sample_views(n=2000, scene=scene)
        stats_dict[scene] = (positions, scores)

    with open('replica_stats.pkl', 'wb') as f:
        pickle.dump(stats_dict, f)

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()
    #
    # samples = dict()
    #
    # for scene in env.scenes:
    #     samples[scene] = []
    #     env.set_scene(scene)
    #     for i in range(num_samples):
    #         env.reset()
    #         view = env.init_view
    #         img1 = cv2.cvtColor(np.array(view), cv2.COLOR_RGB2BGR)
    #         # find the keypoints and descriptors with SIFT
    #         kp1, des1 = sift.detectAndCompute(img1, None)
    #         env.reset()
    #         pos = env.init_pos
    #
    #         score = env.current_score
    #         samples[scene].append((pos, score, des1))
    #
    #     with open('eval_samples_and_descriptors.pkl', 'wb') as f:
    #         pickle.dump(samples, f)
    #
    # with open('/phoenix/S7/ha366/habitat_samples/eval_samples_and_descriptors.pkl', 'rb') as f:
    #     samples = pickle.load(f)
    #
    # for scene in env.scenes[:1]:
    #     env.set_scene(scene)
    #     for i in range(10):
    #         env.reset()
    #         env.get_num_matching(samples)



    # env0 = HabitatEnv(scenes_stats_path=scene_stats_path, data_dir=data_dir, scene_change_freq=1000000000,
    #                   random_seed=1111)
    # view = env0.reset()
    # print(view.shape)
    # # env0 = HabitatEnv(data_dir=data_dir, scenes_stats_path=scene_stats_path, scenes_list_path=None)
    # scenes_stats = dict()
    # for scene in env0.scenes:
    #     scenes_stats[scene] = env0.sample_views(num_samples, scene)
    #
    # with open('finetuned_model_habitat_stats/new_stats.pkl', 'wb') as f:
    #     pickle.dump(scenes_stats, f)

    # root_dir = 'nn_visualization'
    # os.makedirs(root_dir, exist_ok=True)
    # stats = env0.stats
    #
    # n = 10

    # for num_samples in [1000, 5000, 10000]:
    #     for k in [50, 100, 200, 500]:
    #         storage_dir = os.path.join(root_dir, 'N_{}_k_{}'.format(num_samples, k))
    #         os.makedirs(storage_dir, exist_ok=True)
    #         for scene in env0.scenes:
    #             # sample views and scores as dataset
    #             positions, scores = env0.sample_views(num_samples, scene)
    #             positions = np.array(positions)
    #
    #             scores = np.array(scores)
    #             # mean = stats[scene]['mean']
    #             # std = stats[scene]['std']
    #
    #             query_positions, query_scores = env0.sample_views(n)
    #
    #             for i in range(n):
    #                 pos, score = query_positions[i], query_scores[i]
    #                 # find k nearest neighbors
    #                 dist_index = []
    #                 for idx, p in enumerate(positions):
    #                     d = np.linalg.norm(pos-p)
    #                     dist_index.append((d, idx))
    #
    #                 sorted_dist_index = sorted(dist_index, key=lambda x: x[0])
    #                 sorted_dist_index = np.array(sorted_dist_index)
    #
    #                 print('shape of dist index array', sorted_dist_index.shape)
    #
    #                 nn_indices = sorted_dist_index[:k, 1].astype(np.int)
    #                 far_indices = sorted_dist_index[k:, 1].astype(np.int)
    #
    #                 near_positions = positions[nn_indices]
    #                 far_positions = positions[far_indices]
    #
    #                 plt.scatter(far_positions[:, 0], far_positions[:, 2], c='r', alpha=0.5)
    #                 plt.scatter(near_positions[:, 0], near_positions[:, 2], c='y', alpha=0.5)
    #                 plt.scatter([pos[0]], [pos[2]], c='b', alpha=0.5)
    #
    #                 plt.savefig(os.path.join(storage_dir, '{}-{}.png'.format(scene, i)))
    #                 plt.close()

    # for scene in env0.scenes:
    #     for num_samples in [2000]:
    #         positions, scores = env0.sample_views(num_samples, scene)
    #
    #         for k in [50, 100, 200]:
    #             storage_dir = os.path.join(root_dir, 'N_{}_k_{}'.format(num_samples, k))
    #             os.makedirs(storage_dir, exist_ok=True)
    #
    #             # sample views and scores as dataset
    #             n = 1000
    #             query_positions, query_scores = env0.sample_views(n)
    #
    #             good_positions = []
    #             bad_positions = []
    #
    #             for i in range(n):
    #                 pos, score = query_positions[i], query_scores[i]
    #                 # find k nearest neighbors
    #                 dist_index = []
    #                 for idx, p in enumerate(positions):
    #                     d = np.linalg.norm(pos - p)
    #                     dist_index.append((d, idx))
    #
    #                 sorted_dist_index = sorted(dist_index, key=lambda x: x[0])
    #                 sorted_dist_index = np.array(sorted_dist_index)
    #
    #                 nn_indices = sorted_dist_index[:k, 1].astype(np.int)
    #
    #                 nn_scores = scores[nn_indices]
    #                 mean = np.mean(nn_scores)
    #                 std = np.std(nn_scores)
    #
    #                 if score > mean+std:
    #                     good_positions.append(pos)
    #                 else:
    #                     bad_positions.append(pos)
    #
    #
    #             good_positions = np.array(good_positions)
    #             bad_positions = np.array(bad_positions)
    #
    #             plt.scatter(bad_positions[:, 0], bad_positions[:, 2], c='r', alpha=0.5)
    #             plt.scatter(good_positions[:,0], good_positions[:,2], c='g', alpha=0.5)
    #
    #             print('good positions shape', good_positions.shape)
    #             print('bad positions shape', bad_positions.shape)
    #
    #             plt.savefig(os.path.join(storage_dir, '{}.png'.format(scene)))
    #             plt.close()

    # num_samples = 2000
    # samples_dict = dict()
    #
    # filter_names = Instafilter.get_models()
    # filter_names = sorted(filter_names)[:15]
    # filters = dict()
    # for filter_name in filter_names:
    #     filters[filter_name] = Instafilter(filter_name)
    #
    # scenes = sorted(env.scenes)[45:]
    #
    # for scene in scenes:
    #     if scene in samples_dict:
    #         continue
    #     positions, scores = env.sample_views(num_samples, scene, filters)
    #     samples_dict[scene] = (positions, scores)
    #     print('finished {} scene'.format(scene))
    #     with open('samples_with_filters_4.pkl', 'wb') as f:
    #         pickle.dump(samples_dict, f)

    # samples_dict = dict()
    # filenames = ['samples_with_filters_{}.pkl'.format(i) for i in range(1,5)]
    #
    # for filename in filenames:
    #     with open(filename, 'rb') as f:
    #         s = pickle.load(f)
    #     for key in s:
    #         samples_dict[key] = s[key]
    # print('have {} many scenes'.format(len(samples_dict)))
    #
    # with open('samples_with_filters.pkl', 'wb') as f:
    #     pickle.dump(samples_dict, f)


        # good_positions = positions[scores > mean + std]
        # print('good positions shape', good_positions.shape)
        # bad_positions = positions[scores < mean + std]
        # print('bad positions shape', bad_positions.shape)

        # plt.scatter(bad_positions[:,0], bad_positions[:,2], c='r', alpha=0.5)
        # plt.scatter(good_positions[:,0], good_positions[:,2], c='g', alpha=0.5)
        #
        # plt.savefig(os.path.join(storage_dir, '{}.png'.format(scene)))
        # plt.close()



    # plt.scatter(good_positions[])
    # fig, ax = plt.subplots(1,3)
    #
    # ax[0].scatter(xs[-335:], ys[-335:], c='g')


