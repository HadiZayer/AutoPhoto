# from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import CnnLstmPolicy, CnnLnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy
from habitat_env import HabitatEnv
from stable_baselines import PPO2, DQN, A2C
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import feature_extractors
import os
import tensorflow as tf

data_dir = '/phoenix/S7/ha366/habitat_data/gibson'
path_generator = lambda scene: os.path.join(data_dir, '{}.glb'.format(scene))


# net_arch = [512, 512, 512, 'lstm', dict(pi=[128, 128, 128],
#                  vf=[128, 128, 128])]
# net_arch = [512, 'lstm', 256, 256, dict(pi=[128, 128, 128],
#                  vf=[128, 128, 128])]
# net_arch = [512, 512, 512, 'lstm', dict(pi=[128], vf=[128])]

net_arch = [512, 'lstm', 256, 256, dict(pi=[128], vf=[128])]
net_arch_dqn = [512, 512, 256, 256, 128]
# net_arch = [512, 512, 256, 256, dict(pi=[128], vf=[128])]

# policy_kwargs = {'cnn_extractor': feature_extractors.mixed_cnn}
policy_kwargs = {'net_arch': net_arch}  # {'cnn_extractor': feature_extractors.big_nature_cnn}
dqn_policy_kwargs = {'layers': net_arch_dqn, 'feature_extraction': 'mlp'}
feed_forward = True

threshold_factor = 1
num_scenes = 61
# prefix = 'threshold_{}_{}_scenes'.format(threshold_factor, num_scenes)

tf.test.is_gpu_available()
change_freq = 250
n = 2
noisy = False
mixed_state = False

multi_features = True
use_filters = False

multi_movement='fine_turns'

prefix = 'ablation_single_feats_{}'.format(multi_movement)


scene_stats_path = 'finetuned_model_habitat_stats/train_stats.pkl'
if use_filters:
    samples_path = 'samples_with_filters.pkl'
else:
    samples_path = 'full_samples.pkl'

print('--------creating env---------')

env0 = lambda: \
    HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
               gpu=0, feed_forward=feed_forward, ordering=(0,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

env1 = lambda: \
    HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
               gpu=1, feed_forward=feed_forward, ordering=(1,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env2 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=0, feed_forward=feed_forward, ordering=(2,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env3 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=1, feed_forward=feed_forward, ordering=(3,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env4 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=1, feed_forward=feed_forward, ordering=(4,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env5 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=1, feed_forward=feed_forward, ordering=(5,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env6 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=1, feed_forward=feed_forward, ordering=(6,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env7 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=1, feed_forward=feed_forward, ordering=(7,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env8 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=2, feed_forward=feed_forward, ordering=(8,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env9 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=2, feed_forward=feed_forward, ordering=(9,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env10 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=2, feed_forward=feed_forward, ordering=(10,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env11 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=2, feed_forward=feed_forward, ordering=(11,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env12 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(12,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env13 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(13,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env14 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(14,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env15 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(15,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env16 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(16,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env17 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(17,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env18 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(18,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

# env19 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(19,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)
#
# env20 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=0, feed_forward=feed_forward, ordering=(20,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)
#
# env21 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=0, feed_forward=feed_forward, ordering=(21,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)
#
# env22 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(22,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)
#
# env23 = lambda: \
#     HabitatEnv(scenes_stats_path=scene_stats_path, path_generator=path_generator, samples_path=samples_path, scene_change_freq=change_freq, random_seed=1111,
#                gpu=3, feed_forward=feed_forward, ordering=(23,n), multi_movement=multi_movement, threshold_factor=threshold_factor, mixed_state = mixed_state, multi_features=multi_features, use_filters=use_filters)

envs = [env0, env1]
# envs = [env0, env1, env2, env3, env4, env5, env6, env7]
    # env8, env9, env10, env11, env12, env13, env14, env15]
# , env16, env17, env18, env19]
#     , env20, env21, env22, env23]

assert n == len(envs)

def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    vec_envs = SubprocVecEnv(envs, 'forkserver')

    print('--------finished creating env---------')

    # model = DQN(FeedForwardPolicy, vec_envs, learning_rate=1e-3, prioritized_replay=True, verbose=1,
    #             tensorboard_log="./dqn_tensorboard/",
    #             policy_kwargs=dqn_policy_kwargs)
    # model = A2C(MlpLstmPolicy, vec_envs, verbose=1,
    #              tensorboard_log="./{}_a2c_tensorboard/".format(prefix),
    #              policy_kwargs=policy_kwargs)
    model = PPO2(MlpLstmPolicy, vec_envs, verbose=1, nminibatches=len(envs),
                 tensorboard_log="./{}_ppo_tensorboard/".format(prefix),
                 policy_kwargs=policy_kwargs)


    print('-------starting training---------')


    model.learn(total_timesteps=50000)
    model.save("{}-habitat-50k".format(prefix))
    model.learn(total_timesteps=50000, reset_num_timesteps=False)
    model.save("{}-habitat-100k".format(prefix))
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save("{}-habitat-200k".format(prefix))
    model.learn(total_timesteps=300000, reset_num_timesteps=False)
    model.save("{}-habitat-500k".format(prefix))
    model.learn(total_timesteps=500000, reset_num_timesteps=False)
    model.save("{}-habitat-1000k".format(prefix))
    model.learn(total_timesteps=500000, reset_num_timesteps=False)
    model.save("{}-habitat-1500k".format(prefix))



if __name__ == '__main__':
    main()
