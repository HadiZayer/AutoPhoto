from habitat_env import HabitatEnv
from stable_baselines.common.policies import CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines import A2C, DQN, PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import numpy as np
import feature_extractors
import pickle
import os


habitat = 'replica'
multi_movement = False
noisy = False
model_names = [('mlp_fulltrainset_batch8_ordered_sampling_multi_features-habitat-1500k', 'nochange20_500k')]
# model_names = [('ablation_dqn-habitat-500k.zip', 'nochange20_500k')]
multi_feature = True
correct = []

if habitat == 'gibson':
    data_dir = '/phoenix/S7/ha366/habitat_data/gibson'
    scene_stats_path = 'finetuned_model_habitat_stats/eval_stats.pkl'
    # scene_stats_path = 'finetuned_model_habitat_stats/train_stats.pkl'
    path_generator = lambda scene: os.path.join(data_dir, '{}.glb'.format(scene))
    samples_path = 'full_samples.pkl'
    states_save_path = 'gibson_init_states.pkl'
elif habitat == 'mat':
    data_dir = '/phoenix/S7/ha366/habitat_data/matterport/v1/tasks/mp3d'
    scene_stats_path = '/home/ha366/rl_photo/matterport_small_samples/modified_samples.pkl'
    path_generator = lambda scene: os.path.join(data_dir, scene, '{}.glb'.format(scene))
    samples_path = scene_stats_path
    states_save_path = 'matterport_init_states.pkl'
elif habitat == 'replica':
    data_dir = '/phoenix/S7/ha366/Replica-Dataset/replica_v1'
    scene_stats_path = 'replica_stats.pkl'
    path_generator = lambda scene: os.path.join(data_dir, '{}/habitat/mesh_semantic.ply'.format(scene))
    samples_path = scene_stats_path
    states_save_path = 'replica_init_states.pkl'
else:
    raise ValueError

# net_arch = [512, 512, 512, 'lstm', dict(pi=[128, 128, 128], vf=[128, 128, 128])]
# net_arch = [512, 'lstm', 256, 256, dict(pi=[128, 128, 128], vf=[128, 128, 128])]
# net_arch = [512, 512, 512, 'lstm', dict(pi=[128], vf=[128])]
net_arch = [512, 'lstm', 256, 256, dict(pi=[128], vf=[128])]
# net_arch = [512, 512, 256, 256, dict(pi=[128], vf=[128])]
net_arch_dqn = [512, 512, 256, 256, 128]

# policy_kwargs = {'cnn_extractor': feature_extractors.mixed_cnn_only_aes}
policy_kwargs = {'net_arch': net_arch}  # {'cnn_extractor': feature_extractors.big_nature_cnn}
dqn_policy_kwargs = {'layers': net_arch_dqn, 'feature_extraction': 'mlp'}

feed_forward=True
mixed_state =False
prefix = 'MLP_multi_scene'



print('--------creating env---------')

num_samples = 100
env0 = HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, scene_change_freq=100000, random_seed=3333,
                  feed_forward=feed_forward, samples_path=samples_path, multi_movement=multi_movement, noisy=noisy,
                  mixed_state=mixed_state, multi_features=multi_feature)

print('--------finished creating env---------')

print('----------loading model-----------')
vec_envs = DummyVecEnv([lambda: env0])


accuracies = dict()

ts = []
num_better = 0
high_reward = 0
num_matching = 0
joint_aesthetic_feature = 0
total_count = 0

improvements = []

init_states = dict()

for model_name, label in model_names:
    if feed_forward:
        # model = DQN(FeedForwardPolicy, vec_envs, learning_rate=1e-3, prioritized_replay=True, verbose=1,
        #             policy_kwargs=dqn_policy_kwargs)
        # model = A2C(MlpLstmPolicy, vec_envs, verbose=1, policy_kwargs=policy_kwargs)
        model = PPO2(MlpLstmPolicy, vec_envs, verbose=1, nminibatches=1, policy_kwargs=policy_kwargs, seed=1111)
        # model = PPO2(CnnLstmPolicy, vec_envs, verbose=1, nminibatches=1, policy_kwargs=policy_kwargs)
    else:
        model = PPO2(CnnLstmPolicy, vec_envs, verbose=1, nminibatches=1, policy_kwargs=policy_kwargs)
    model.load_parameters(model_name)

    print('----------finished loading model -----------')
    cur_env = model.get_env()
    for scene in env0.scenes:
        init_states[scene] = []

        env0.set_scene(scene)
        num_scenes = 0
        num_high_rewards = 0
        num_better_than_center = 0

        t = 0
        obs = cur_env.reset()
        state = None
        init_states[scene].append(env0.client.get_agent_state())

        while True:
            action, state = model.predict(obs, state=state)
            if t > 98:
                action = [3]
            # print('action {}'.format(action))
            cur_env.step_async(action)
            obs, rewards, dones, info = cur_env.step_wait()
            # print('rewards {}, dones {}'.format(rewards, dones))
            t += 1
            if np.any(dones):
                init_states[scene].append(env0.client.get_agent_state())
                ts.append(t)

                # print('took {} steps'.format(t))
                t = 0
                num_scenes += 1
                state = None

                if info[0]['high_reward']:
                    num_high_rewards += 1
                    correct.append(1)
                else:
                    correct.append(0)

                num_better += info[0]['better_than_init']
                high_reward += info[0]['high_reward']
                num_matching += info[0]['match']
                joint_aesthetic_feature += info[0]['high_reward'] and info[0]['match']
                total_count += 1

                if info[0]['better_than_init']:
                    improvements.append(info[0]['score_diff'])


            if num_scenes >= num_samples:
                break

        accuracies[scene] = num_high_rewards / num_samples

for scene in accuracies:
    print('scene {} accuracy is {}'.format(scene, accuracies[scene]))

accs = [accuracies[scene] for scene in accuracies]
accs = np.array(accs)
print('mean accuracy {} and std {}'.format(np.mean(accs), np.std(accs)))
print('median accuracy {}'.format(np.median(accs)))
print('model {}'.format(model_names[0][0]))

ts = np.array(ts)

print('num high rewards {}, num better than init {}, total count {}'.format(high_reward, num_better, total_count))
print('num matching {}'.format(num_matching))
print('percentage of high rewards {}, percentage better than init {}'.format(high_reward/total_count, num_better/total_count))
print('percentage matching {}'.format(num_matching/total_count))
print('percentage matching and high aesthetic {}'.format(joint_aesthetic_feature/total_count))
print('mean num of timesteps {}, median of timesteps {}'.format(np.mean(ts), np.median(ts)))

improvements = np.array(improvements)
print('mean improvement {}, std improvements {}'.format(np.mean(improvements), np.std(improvements)))

with open('accs/{}.pkl'.format(model_names[0][0]), 'wb') as f:
    pickle.dump(accuracies, f)

# with open(states_save_path, 'wb') as f:
#     pickle.dump(init_states, f)

correct = np.array(correct)

print('mean of correct runs {} with std {}'.format(np.mean(correct), np.std(correct)))
