import os
import glob
import numpy as np
from habitat_client import HabitatClient
import pathlib

data_dir =  '<gibson dataset path>'

path_generator = lambda scene: os.path.join(data_dir, '{}.glb'.format(scene))
dirs = glob.glob(os.path.join(data_dir, '*'))
dir_names = [pathlib.Path(dir_name).stem for dir_name in dirs]

# path_generator = lambda scene: os.path.join(data_dir, scene, '{}.glb'.format(scene))

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

    state_score_pairs = []
    for i in tqdm.tqdm(range(num_samples)):
        view = client.reset_same_scene()
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
    plt.savefig(os.path.join(storage_dir, '{}_samples_visualized.png'.format(scene)))
    plt.close()

    with open(os.path.join(storage_dir,'samples.pkl'), 'wb') as f:
        pickle.dump(samples, f)