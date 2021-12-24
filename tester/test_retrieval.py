"""
Code adapted from: https://github.com/ecom-research/ComposeAE
Evaluates the retrieval model.
"""

import numpy as np
import torch, os
import random
from tqdm import tqdm as tqdm
from collections import OrderedDict

from img_text_composition_model.compose_transformers import ComposeTransformers
from trainer.engine import load_dataset
from dataloader.fashion200k_loader import Fashion200k

from utils.util import get_checkpoints, show_demo_test_retrieval
from trainer.modelloader import load_checkpoint



def test(config, model, testset):
    """Tests a model over the given testset.
        By various metrics.
    """
    torch.cuda.empty_cache()
    test_queries = testset.get_test_queries()
    model.eval()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []

    quer_imgs = []
    srcs_queries = []
    mods_queries = []

    for idx, t in enumerate(tqdm(test_queries)):
        torch.cuda.empty_cache()
        quer_imgs += [testset.get_img(t['source_img_id'])]
        mods_queries += [t['target_caption']]
        srcs_queries += [t['source_caption']]

        if len(quer_imgs) >= config.dataloader.test.batch_size or t is test_queries[-1]:
            if 'torch' not in str(type(quer_imgs[0])):
                quer_imgs = [torch.from_numpy(d).float() for d in quer_imgs]
            quer_imgs = torch.stack(quer_imgs).float()
            quer_imgs = torch.autograd.Variable(quer_imgs).cuda()
            dct_with_representations = model.compose_img_text(quer_imgs, mods_queries)
            f = dct_with_representations["repres"].data.cpu().numpy()

            all_queries += [f]
            quer_imgs = []
            mods_queries = []


    all_queries = np.concatenate(all_queries)
    all_target_captions = [t['target_caption'] for t in test_queries]

    # compute all image features
    imgs = []
    for i in tqdm((range(len(testset.imgs)))):
        imgs += [testset.get_img(i)]

        if len(imgs) >= config.dataloader.test.batch_size or i == len(testset.imgs) - 1:
            if 'torch' not in str(type(imgs[0])):
                img_to_ftr = [torch.from_numpy(d).float() for d in imgs]
            img_to_ftr = torch.stack(imgs).float()
            img_to_ftr = torch.autograd.Variable(img_to_ftr).cuda()
            img_to_ftr = model.extract_img_feature(img_to_ftr).data.cpu().numpy()

            all_imgs += [img_to_ftr]
            imgs = []

    all_imgs = np.concatenate(all_imgs)
    all_captions = [img['captions'][0] for img in testset.imgs]

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

    # compute precision & recalls
    recall_out = []
    prec_out = []
    caption_result = [[all_captions[nn] for nn in nns] for nns in nn_result]

    for k in [1, 10, 50, 100]:
        relevance = 0.0
        prec = 0.0
        for i, nns in enumerate(caption_result):
            if all_target_captions[i] in nns[:k]:
                relevance += 1
                prec += relevance/(i+1)

        recall = relevance / len(caption_result)
        prec /= len(caption_result)

        recall_out += [('Recall_top_' + str(k) + '_correct_composition : ', recall)]
        prec_out += [('Precision_top_' + str(k) + '_correct_composition : ', prec)]


    return prec_out, recall_out, nn_result




def demo_retrieval(config, indexes, nn_result = None):
    testset = Fashion200k(config=config, mode='test')
    test_queries = testset.get_test_queries()

    quer_imgs = []
    mods_queries = []
    srcs_queries = []
    visual_imgs = []

    for idx, t in enumerate(test_queries):
        if idx in indexes:
            quer_imgs += [testset.get_img(t['source_img_id'])]
            mods_queries += [t['target_caption']]
            srcs_queries += [t['source_caption']]
            indexes = np.delete(indexes, np.where(indexes == idx))

        if len(indexes) == 0:
            break

    for idx in (range(len(testset.imgs))):
        visual_imgs += [testset.get_img(idx)]

    for idx in range(len(quer_imgs)):
        print('-'*30 + '\nDemo on image: ', idx)
        show_demo_test_retrieval(quer_imgs[idx], srcs_queries[idx], mods_queries[idx], show=True)

        revelance_idxs = nn_result[idx][:10]

        print('TOP 10 REVELANT RESULTS: ')
        for idx in revelance_idxs:
            show_demo_test_retrieval(visual_imgs[idx], show=True)




def validation(config, model=None, testset=None, log=None):
    print('!!!!!!!! Validating progress !!!!!!!!')

    if log == None and model == None:
        # Get state_dict params of the Model checkpoint.
        model = ComposeTransformers(config)
        device = torch.device("cuda:0" if (torch.cuda.is_available() and config.cuda) else "cpu")

        get_checkpoints()
        checkpoint_name = input("Choose one of these checkpoints: ")
        cpkt_fol_name = os.path.join(config.cwd, f'checkpoints/{checkpoint_name}')
        checkpoint_dirmodel = f'{cpkt_fol_name}/latest_checkpoint.pth'

        model, _, _, _ = load_checkpoint(config, checkpoint_dirmodel, model)
        model = model.to(device)

        # Get testset.
        testset = Fashion200k(config=config, mode='test')

    precision_result, recall_result = [], []
    precision, recall, nn_results = test(config, model, testset)

    precision_result += [('PRECISION ' + metric_name, metric_value)
                          for metric_name, metric_value in precision]
    recall_result += [('RECALL ' + metric_name, metric_value)
                          for metric_name, metric_value in recall]

    if log != None:
        for metric_name, metric_value in precision_result:
            log.info(f'{metric_name} : {round(metric_value, 4)}')

        for metric_name, metric_value in recall_result:
            log.info(f'{metric_name} : {round(metric_value, 4)}')

    else:
        for metric_name, metric_value in precision_result:
            print(f'{metric_name} : {round(metric_value, 4)}')

        for metric_name, metric_value in recall_result:
            print(f'{metric_name} : {round(metric_value, 4)}')

    return nn_results
