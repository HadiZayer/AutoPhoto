import torch, torchvision
import numpy as np
import os
from PIL import Image
import pickle
import glob
import matplotlib.pyplot as plt

# from ViewEvaluationNet.nets.FullVggCompositionNet import FullVggCompositionNet as CompositionNet
# from ViewEvaluationNet.nets.SiameseNet import SiameseNet
from ViewEvaluationNet.datasets import data_transforms
# from ViewEvaluationNet.pt_utils import cuda_model
from robust_cnns.models_lpf import resnet18


# parser = argparse.ArgumentParser(description="Full VGG trained on CPC")
# parser.add_argument('--l1', default=1024, type=int)
# parser.add_argument('--l2', default=512, type=int)
# parser.add_argument("--gpu_id", default='1', type=str)
# parser.add_argument('--multiGpu', '-m', action='store_true', help='positivity constraint')
# parser.add_argument('--resume', '-r', default='snapshots/MTweak3-FullVGG-1024x512/params/EvaluationNet.pth.tar', type=str, help='resume from checkpoint')

# model_path = '/Users/hadi/Documents/kb/rl_photo/resnet_model/resnet-model42.pt'
# model_path = '/home/ha366/finetuned_ven/storage/2020-08-12_18-19-33/largelr-model5.pt'
#'/home/ha366/finetuned_storage/small_lr/model29.pt'
# model_path = './ViewEvaluationNet/snapshots/MTweak3-FullVGG-1024x512/params/EvaluationNet.pth.tar'
model_path = 'resnet-model42.pt'

class ScoringModel:
    def __init__(self, l1=1024, l2=512, gpu_id=0, path=model_path, gpu=True):
        # self.model = CompositionNet(pretrained=False, LinearSize1=l1, LinearSize2=l2)
        # siamese_net = SiameseNet(self.model)
        # ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        # model_state_dict = ckpt['state_dict']  # TODO if reverting to baseline ['state_dict']
        # siamese_net.load_state_dict(model_state_dict)
        #
        # self.model = cuda_model.convertModel2Cuda(self.model, gpu_id, multiGpu)

        model = resnet18(filter_size=3)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.fc.weight = torch.nn.Parameter(-model.fc.weight)
        model.fc.bias = torch.nn.Parameter(-model.fc.bias)

        self.gpu = gpu
        if self.gpu:
            model = model.cuda()
        self.model = model

        self.model.eval()

        self.transform = data_transforms.get_val_transform(224)

    def forward(self, images):
        with torch.no_grad():
            t_images = torch.stack(images)
            if self.gpu:
                t_images = t_images.cuda()
            scores = self.model(t_images)
            return scores

    def get_features(self, img):
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
            img = img.unsqueeze(0)
            for layer in list(self.model.named_children())[:-1]:
                img = layer[1](img)
            features = img.detach().cpu().numpy().flatten()
            # print(features.shape)
            # print(features.flatten().shape)

            return features

    def get_multi_features(self, img):
        with torch.no_grad():
            if self.gpu:
                img = img.cuda()
            img = img.unsqueeze(0)
            layers = ['layer{}'.format(i) for i in range(1,5)]
            saved_features = []
            avg_pool = list(self.model.named_children())[-2]
            for layer in list(self.model.named_children())[:-1]:
                img = layer[1](img)
                if layer[0] in layers:
                    avg_pooled_features = avg_pool[1](img)
                    features = avg_pooled_features.detach().cpu().numpy().flatten()
                    saved_features.append(features)
            assert len(saved_features) > 0

            concat_features = np.concatenate(saved_features)
            # print(features.shape)
            # print(features.flatten().shape)

            return concat_features

if __name__ == '__main__':

    m_path = '/Users/hadi/Documents/kb/rl_photo/resnet_model/resnet-model42.pt'

    storage_dir = '/Users/hadi/Downloads/real_life_study_imgs'
    pairs_txt = '/Users/hadi/Downloads/real_life_study_imgs/keep_imgs.txt'

    scoring_model = ScoringModel(path=m_path, gpu=False)

    better = 0.0
    equal = 0.0
    worse = 0.0

    with open(pairs_txt, 'r') as f:
        for line in f:
            init, final = line.strip().split(',')

            init = os.path.join(storage_dir, init)
            final = os.path.join(storage_dir, final)

            init_img = Image.open(init).convert('RGB')
            final_img = Image.open(final).convert('RGB')

            init_img = scoring_model.transform(init_img)
            final_img = scoring_model.transform(final_img)

            scores = scoring_model.forward([init_img, final_img])

            init_score = scores[0]
            final_score = scores[1]

            if final_score > init_score:
                better += 1
            elif final_score == init_score:
                equal += 1
            else:
                worse += 1

        print('have better {} equal {} worse {}'.format(better, equal, worse))
        total = better + equal + worse
        print('have percent better {} equal {} worse {}'.format(better/total, equal/total, worse/total))


    # from instafilter import Instafilter
    # def apply_filter(img, filter_name):
    #     if filter_name == "no_filter":
    #         return img
    #     filter = Instafilter("1977")
    #     original_img = np.array(img)[:, :, ::-1]
    #     img = filter(original_img, is_RGB=False)
    #     view = Image.fromarray(img[:, :, ::-1]).convert('RGB')
    #     return view
    #
    # scoring_model = ScoringModel()
    #
    # img_path = 'cat.jpg'
    # o_img = Image.open(img_path).convert('RGB')
    #
    # t_img = scoring_model.transform(o_img).cuda()
    # print(scoring_model.get_multi_features(t_img).shape, scoring_model.get_multi_features(t_img).dtype)
    #
    # f_img = apply_filter(o_img, 'Dogpatch')
    # t_img = scoring_model.transform(f_img).cuda()
    # print(scoring_model.get_multi_features(t_img).shape, scoring_model.get_multi_features(t_img).dtype)



    # scoring_model = ScoringModel(path='/Users/hadi/Documents/kb/rl_photo/resnet_model/resnet-model42.pt', gpu=False)
    #
    # input_dir = '/Users/hadi/Documents/kb/rl_photo/photos_used_in_paper'
    # img_paths = glob.glob(os.path.join(input_dir, '*png'))
    #
    # for img_path in img_paths:
    #     img = Image.open(img_path).convert('RGB')
    #     img = img.resize((224,224), Image.NEAREST)
    #     img = scoring_model.transform(img)
    #     score = scoring_model.forward([img])
    #     print('img {} score {}'.format(img_path, score))

    # with open('/Users/hadi/Documents/kb/ViewEvaluationNet/different_ground_nh_pairs/pairs_paths.pkl', 'rb') as f:
    #     paths = pickle.load(f)
    #
    # with open('/Users/hadi/Documents/kb/ViewEvaluationNet/different_ground_nh_pairs/nh_info.pkl', 'rb') as f:
    #     old_info = pickle.load(f)
    #
    # agreements = 0
    # info = []
    #
    # for i, p in enumerate(paths):
    #     p1, p2 = p
    #     img1 = Image.open(p1)
    #     img1 = scoring_model.transform(img1)
    #
    #     img2 = Image.open(p2)
    #     img2 = scoring_model.transform(img2)
    #
    #     score1, score2 = scoring_model.forward([img1, img2])
    #     print('loading {}, score1 {}, score2 {}'.format(i, score1, score2))
    #
    #     first_better = score1 > score2
    #     threshold = old_info[i][0]
    #     info.append((threshold, first_better))
    #
    #     agreements += 1 if old_info[i][1] == first_better else 0
    #
    # print('agreements', agreements)
    # with open('/Users/hadi/Documents/kb/ViewEvaluationNet/different_ground_nh_pairs/vgg_info.pkl', 'wb') as f:
    #     pickle.dump(info, f)


    # path = 'processed_ithaca_916'
    # indices_path = ''
    #
    # with open(indices_path, 'rb') as f:
    #     indices = pickle.load(f)
    #
    #
    # xs = range(11)
    # ys = range(9)
    #
    # sliding_scores = dict()
    #
    # for s in indices:
    #     print('loading scenes {}'.format(s))
    #     sliding_scores[s] = np.zeros((len(ys), len(xs)))
    #     for x in xs:
    #         for y in ys:
    #             img_name = 'scene{}-x{}-y{}.jpeg'.format(s, x, y)
    #             img_path = os.path.join(path, img_name)
    #             img = Image.open(img_path)
    #             img = scoring_model.transform(img)
    #             sliding_scores[s][y,x] = scoring_model.forward([img]).item()
    # with open(os.path.join(path,'ven_cached.pkl'), 'wb') as f:
    #     pickle.dump(sliding_scores, f)


    # src_dir = '/home/ha366/habitat/samples'
    # paths = glob.glob(os.path.join(src_dir, '*'))
    #
    # storage_path = 'old_model_habitat_views'
    # os.makedirs(storage_path, exist_ok=True)
    #
    # scenes_stats = dict()
    #
    # for path in paths:
    #     scene = path.split('/')[-1]
    #     imgs = glob.glob(os.path.join(path, '*.jpg'))
    #
    #     sliding_scores = dict()
    #     scores = np.zeros(len(imgs))
    #
    #     path_scores = []
    #
    #     for idx, img_path in enumerate(imgs):
    #         img = Image.open(img_path)
    #         img = scoring_model.transform(img)
    #         scores[idx] = scoring_model.forward([img]).item()
    #         print(img_path, scores[idx])
    #         path_scores.append((img_path, scores[idx]))
    #
    #     scenes_stats[scene] = dict()
    #     scenes_stats[scene]['mean'] = np.mean(scores.flatten())
    #     scenes_stats[scene]['std'] = np.std(scores.flatten())
    #     print('scene {}, stats {}'.format(scene, scenes_stats[scene]))
    #
    # with open(os.path.join(storage_path, 'stats.pkl'), 'wb') as f:
    #     pickle.dump(scenes_stats, f)

        # path_scores = sorted(path_scores, key=lambda x: x[1])
        #
        # fig, ax = plt.subplots(3, 3)
        #
        # for i in range(3):
        #     img_path, score = path_scores[::-1][i]
        #     img = Image.open(img_path)
        #     ax[0,i].imshow(img)
        #     ax[0,i].set_title('top {}'.format(i))
        #
        # for i in range(3):
        #     n = len(path_scores)//2
        #     img_path, score = path_scores[n+i]
        #     img = Image.open(img_path)
        #     ax[1,i].imshow(img)
        #     ax[1,i].set_title('median+{}'.format(i))
        #
        # for i in range(3):
        #     img_path, score = path_scores[i]
        #     img = Image.open(img_path)
        #     ax[2,i].imshow(img)
        #     ax[2,i].set_title('worse {}'.format(i))
        #
        # plt.tight_layout()
        # plt.savefig(os.path.join(storage_path, '{}_views.jpg'.format(scene)))





