"""Provides data for training and testing.
Adapted from: https://github.com/ecom-research/ComposeAE
"""

import numpy as np
from os import listdir
from os.path import isfile
from os.path import join
import PIL
import skimage
import torch
import json
from torchvision import transforms
import warnings
import random
from .baseloader import BaseDataset

import matplotlib.pyplot as plt


class Fashion200k(BaseDataset):
    """Parser and loader of Fashion200k dataset.
        For the mode=Train; provide images and annotations.
        For the mode=Test; provide test queries.
    """

    def __init__(self, config, mode, display=False):
        super().__init__(config)
        self.display = display
        self.path = config.dataset.path
        self.do_aug = None
        self.mode = mode
        self.img_path = config.dataset.path + '/'
        self.img_size = config.dataset.img_size

        # get label files for the split
        label_path = config.dataset.path + '/labels/'
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if mode in f]

        # read image info from label files
        self.imgs = []
        print('-'*10, f'Collecting {mode} dataset! ', '-'*10)

        def caption_post_process(s):
            """Process captions.
            """
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for id, filename in enumerate(label_files):
            print(id+1, ',   Reading  ' + filename)
            with open(label_path + filename, 'r',  encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                if 'shirts/' not in line[0]:
                    continue
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'mode': mode,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Collected ', len(self.imgs), ' images!')

        # generate query for training, display_training or testing
        if self.mode == 'train':
            self.caption_index_init_()
            if self.display == True:
                self.do_aug = False
            else:
                self.do_aug = True


        elif self.mode == 'test':
            self.generate_test_queries_()
            self.do_aug = False



    def get_different_word(self, source_caption, target_caption):
        """For generating target queries.
            Replace the word in source_caption to word in target_caption.
            If word not in target_caption.
        """
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str


    def generate_test_queries_(self):
        """For loading test set.
            Get the query images for test set.
        """
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt',  encoding='utf-8') as f:
            lines = f.readlines()
            self.test_queries = []
            for line in lines:
                source_file, target_file = line.split()
                if not('shirts/' in source_file  and 'shirts/' in target_file):
                    continue
                idx = file2imgid[source_file]
                target_idx = file2imgid[target_file]
                source_caption = self.imgs[idx]['captions'][0]
                target_caption = self.imgs[target_idx]['captions'][0]
                source_word, target_word, mod_str = self.get_different_word(
                    source_caption, target_caption)
                self.test_queries += [{
                    'source_img_id': idx,
                    'source_caption': source_caption,
                    'target_caption': target_caption,
                    'mod': {
                        'str': mod_str
                    }
                }]


    def caption_index_init_(self):
        """ For loading train set.
            Index caption to generate training query-target example on the fly later.
        """
        # index caption to caption_id and caption to image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print('There are', len(caption2imgids), 'unique captions.')

        # dictionary of parent caption to children caption
        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        # print('Modifiable images ', num_modifiable_imgs, end='\n\n')



    def caption_index_sample_(self, idx):
        """Ganerate the queries by modify the source caption to another caption
            which both have same parent.
        """
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str


    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        """For loading train dataset.
        """
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}

        if self.display == True:
            figure = plt.figure(figsize=(150, 150))
            fig, axarr = plt.subplots(1,2)

            axarr[0].set_title(str(self.imgs[idx]['captions'][0] + '                  '))
            axarr[0].imshow(self.get_img(idx).permute(1, 2, 0))

            print(f'Image: {idx}; Source maps to target with the mod string: {mod_str}')
            axarr[1].set_title(str('                  ' + self.imgs[target_idx]['captions'][0]))
            axarr[1].imshow(self.get_img(target_idx).permute(1, 2, 0))
            for ax in fig.get_axes():
                ax.label_outer()
            plt.show()

        else:
            return out



    def get_img(self, idx):
        img_path = self.img_path + self.imgs[idx]['file_path']
        img = PIL.Image.open(img_path).convert('RGB')
        img = img.resize(self.img_size)
        if self.display == True:
            transform = transforms.ToTensor()
        else:
            transform = self.transform()
        return transform(img)


    def transform(self):
        if self.do_aug == True:
            return transforms.Compose([
                    transforms.RandomApply([transforms.RandomAffine(degrees=(-10, 10), translate=(0.15, 0.15), fillcolor=(255, 255, 255)),\
                                    transforms.RandomResizedCrop(size=self.img_size, scale=(0.85, 1.15))], p=0.5),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
        else:
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                    ])
