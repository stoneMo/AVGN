import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import pickle
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_image(path):
    return Image.open(path).convert('RGB')


def load_waveform(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    waveform = audio[:int(samplerate * dur)]

    return waveform, samplerate

def log_mel_spectrogram(waveform, samplerate):
    frequencies, times, spectrogram = signal.spectrogram(waveform, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format in {'vggss', 'vggsound_single'}:
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    elif format == 'vggsound_duet':
        gt_bboxes_raw = {}
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            gt_bboxes_raw[annotation['file']] = annotation['bbox']
        # fns2cls = {item[0]:item[1] for item in csv.reader(open('metadata/vggsound_duet_test.csv'))}
        fns2mix = {item[0]:item[2] for item in csv.reader(open('metadata/vggsound_duet_test.csv'))}
        for annotation in annotations:
            fn = annotation['file']
            fn_mix = fns2mix[fn]
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in gt_bboxes_raw[fn]]
            bboxes_mix = [(np.clip(np.array(bbox_mix), 0, 1) * 224).astype(int) for bbox_mix in gt_bboxes_raw[fn_mix]]
            bboxes_src = [bboxes, bboxes_mix]
            # classes_src = [fns2cls[fn], fns2cls[fn_mix]]
            gt_bboxes[fn] = bboxes_src

    elif format in {'vgginstruments', 'vgginstruments_multi'}:
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt_bboxes[file] = f"{annotation_dir}/{filename}"

    elif format == 'music_solo':
        with open('metadata/music_solo.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(annotation['bbox']), 0, 1) * 224).astype(int)]
            gt_bboxes[annotation['file']] = bboxes

    elif format == 'music_duet':
        with open('metadata/music_duet.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes_src = [annotation['bbox_src1'], annotation['bbox_src2']]
            classes_src = [annotation['class_src1'], annotation['class_src2']]
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in bboxes_src]
            gt_bboxes[annotation['file']] = [bboxes, classes_src]

    return gt_bboxes


def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format in {'vggss', 'music_duet'}:
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map


def mask2gtmap(gt_mask_path):
    with open(gt_mask_path, 'rb') as f:
        gt_mask = pickle.load(f)
    gt_map = cv2.resize(gt_mask, (224,224), interpolation=cv2.INTER_NEAREST)
    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, mode='train', audio_dur=3., 
            image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr', 
            num_classes=0, class_labels=None, num_mixtures=1, class_labels_ss=None, 
            image_files_ss=None, audio_files_ss=None, all_bboxes_ss=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path

        self.mode = mode

        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format
        self.class_labels = class_labels
        self.num_classes = num_classes

        self.num_mixtures = num_mixtures
        self.class_labels_ss = class_labels_ss
        self.image_files_ss = image_files_ss
        self.audio_files_ss = audio_files_ss
        self.all_bboxes_ss = all_bboxes_ss

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):

        image_path = self.image_path
        audio_path = self.audio_path

        anno = {}
        if self.all_bboxes is not None:
            if self.bbox_format in {'flickr', 'vggss'}:
                bboxes = self.all_bboxes[idx]
                bb = -torch.ones((10, 4)).long()
                if len(bboxes) > 0:
                    bb[:len(bboxes)] = torch.from_numpy(np.array(bboxes))
                    anno['bboxes'] = bb
                    anno['gt_map'] = bbox2gtmap(bboxes, self.bbox_format)
                    anno['gt_mask'] = 1             # 1 for samples w. gt_map
                else:
                    anno['bboxes'] = bb
                    anno['gt_map'] = np.zeros([224, 224])
                    anno['gt_mask'] = 0             # 0 for samples w/o. gt_map
            elif self.bbox_format == 'vgginstruments':
                gt_mask_path = self.all_bboxes[idx]
                anno['gt_map'] = mask2gtmap(gt_mask_path)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map
                bb = torch.ones((1, 4)).long()
                anno['bboxes'] = bb
            elif self.bbox_format == 'vgginstruments_multi':
                gt_mask_path = self.all_bboxes[idx]
                gt_mask_path_ss = self.all_bboxes_ss[idx]
                gt_map = mask2gtmap(gt_mask_path)
                gt_map_ss = mask2gtmap(gt_mask_path_ss)
                anno['gt_map'] = np.stack((gt_map, gt_map_ss),axis=0)      # (224*2, 224)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map
                bb = torch.ones((1, 4)).long()
                anno['bboxes'] = bb

            elif self.bbox_format in {'vggsound_single', 'music_solo'}:
                bboxes = self.all_bboxes[idx]
                bb = -torch.ones((10, 4)).long()
                bb[:len(bboxes)] = torch.from_numpy(np.array(bboxes))
                anno['bboxes'] = bb
                gt_map = bbox2gtmap(bboxes, self.bbox_format)
                anno['gt_map'] = gt_map         # (224, 224)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map
            
            elif self.bbox_format == 'vggsound_duet':
                bboxes = self.all_bboxes_ss[idx]
                bb = -torch.ones((10, 4)).long()
                bb[:len(bboxes[0])] = torch.from_numpy(np.array(bboxes[0]))
                anno['bboxes'] = bb
                gt_map = bbox2gtmap(bboxes[0], self.bbox_format)
                gt_map_ss = bbox2gtmap(bboxes[1], self.bbox_format)
                anno['gt_map'] = np.stack((gt_map, gt_map_ss),axis=0)      # (224*2, 224)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map

            elif self.bbox_format == 'music_duet':
                bboxes = self.all_bboxes_ss[idx]
                bb = -torch.ones((10, 4)).long()
                bb[:len(bboxes)] = torch.from_numpy(np.array([bboxes]))
                anno['bboxes'] = bb
                gt_map = bbox2gtmap([bboxes[0]], self.bbox_format)
                gt_map_ss = bbox2gtmap([bboxes[1]], self.bbox_format)
                anno['gt_map'] = np.stack((gt_map, gt_map_ss),axis=0)      # (224*2, 224)
                anno['gt_mask'] = 1             # 1 for samples w. gt_map

        if self.class_labels is not None:
            class_label = torch.zeros(self.num_classes)
            class_idx = self.class_labels[idx]
            class_label[class_idx] = 1
            anno['class'] = class_label

            if self.class_labels_ss is not None:
                class_label_mix = torch.zeros(self.num_classes)
                class_idx_mix = self.class_labels_ss[idx]
                class_label_mix[class_idx_mix] = 1
                anno['class'] = torch.stack([class_label, class_label_mix])

        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = image_path + self.image_files[idx]
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = audio_path + self.audio_files[idx]
        waveform, samplerate = load_waveform(audio_fn)
        spectrogram = self.audio_transform(log_mel_spectrogram(waveform, samplerate))

        # NOTE: mix two audios
        if self.num_mixtures > 1:
            # Mix waveform with other random audios
            mix = [waveform]
            mix_frame = [frame]
            if self.audio_files_ss is None:
                for mix_idx in np.random.choice([r for r in range(len(self.image_files)) if r != idx and self.class_labels[r] != self.class_labels[idx]], size=self.num_mixtures-1, replace=False).tolist():
                    audio_fn2 = audio_path + self.audio_files[mix_idx]
                    waveform2, sample_rate2 = load_waveform(audio_fn2)
                    mix.append(waveform2)
                    if self.class_labels is not None:
                        class_label_mix = torch.zeros(self.num_classes)
                        class_idx_mix = self.class_labels[mix_idx]
                        class_label_mix[class_idx_mix] = 1
                        anno['class'] = torch.stack([class_label, class_label_mix])
            else:
                if self.bbox_format == 'vggsound_duet':
                    for audio_mix in self.audio_files_ss[idx]:
                        audio_fn2 = audio_path + audio_mix
                        waveform2, sample_rate2 = load_waveform(audio_fn2)
                        mix.append(waveform2)
                        
                    # mixed frame
                    for img_mix in self.image_files_ss[idx]:
                        img_fn2 = image_path + img_mix
                        frame2 = self.image_transform(load_image(img_fn2))
                        mix_frame.append(frame2)

                elif self.bbox_format == 'music_duet':
                    mix_frame.append(frame)

            frame = torch.stack(mix_frame)

            if self.bbox_format != 'music_duet':
                mix_waveform = torch.stack([torch.from_numpy(mix_arr) for mix_arr in mix]).sum(0)
                spectrogram = self.audio_transform(log_mel_spectrogram(mix_waveform, samplerate))

        return frame, spectrogram, anno, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path) if fn.endswith('.jpg')}
    if args.trainset in {'music_solo', 'music_duet'}:
        avail_audio_files = []
        for image_file in image_files:
            if image_file[:-10] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    elif args.trainset in {'vggsound_single', 'vggsound_duet', 'vgginstruments', 'vgginstruments_multi', 'music_solo', 'music_duet'}:
        subset = set([line.split(',')[0] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()])
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    if args.trainset in {'music_solo', 'music_duet'}:
        audio_files = [dt[:-10]+'.wav' for dt in avail_files]
    else:
        audio_files = [dt+'.wav' for dt in avail_files]
    image_files = sorted([dt+'.jpg' for dt in avail_files])
    
    all_bboxes = [[] for _ in range(len(image_files))]
    
    if args.trainset in {'vgginstruments', 'vggsound_single'}:
        class_labels = []
        all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
        num_classes = len(all_classes)
        fns2cls = {line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
        for dt in avail_files:
            cls = all_classes.index(fns2cls[dt])
            class_labels.append(cls)
        num_mixtures = 1
        class_labels_ss = None
    elif args.trainset in {'vgginstruments_multi', 'vggsound_duet'}:
        class_labels = []
        all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
        num_classes = len(all_classes)
        fns2cls = {line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
        for dt in avail_files:
            cls = all_classes.index(fns2cls[dt])
            class_labels.append(cls)
        num_mixtures = 2
        class_labels_ss = None
    elif args.trainset == 'music_solo':
        class_labels = []
        all_classes = []
        for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines():
            all_classes.append(line.split(',')[1])
        all_classes = sorted(set(all_classes))
        num_classes = len(all_classes)
        fns2cls1 = {line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
        for dt in avail_files:
            cls_src1 = all_classes.index(fns2cls1[dt])
            class_labels.append(cls_src1)
        num_mixtures = 1
        class_labels_ss = None
    elif args.trainset == 'music_duet':
        class_labels = []
        class_labels_ss = []
        all_classes = []
        for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines():
            all_classes.append(line.split(',')[1])
            all_classes.append(line.split(',')[2])
        all_classes = sorted(set(all_classes))
        num_classes = len(all_classes)
        fns2cls1 = {line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
        fns2cls2 = {line.split(',')[0]:line.split(',')[2] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
        for dt in avail_files:
            cls_src1 = all_classes.index(fns2cls1[dt])
            cls_src2 = all_classes.index(fns2cls2[dt])
            class_labels.append(cls_src1)
            class_labels_ss.append(cls_src2)
        num_mixtures = 2
    else:
        num_mixtures = 1
        num_classes = 0
        class_labels = None
        class_labels_ss = None

    print('class_labels:', class_labels[:10])
    print('all_classes:', all_classes)

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='train',
        image_files=image_files,
        audio_files=audio_files,
        all_bboxes=all_bboxes,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        num_classes=num_classes,
        class_labels=class_labels,
        class_labels_ss = class_labels_ss,
        num_mixtures=num_mixtures
    )


def get_test_dataset(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    if args.testset == 'vggsound_single':
        testcsv = 'metadata/vggsound_single_test.csv'
    elif args.testset == 'vggsound_duet':
        testcsv = 'metadata/vggsound_duet_test.csv'
    elif args.testset == 'vgginstruments':
        testcsv = 'metadata/vgginstruments_test.csv'
    elif args.testset == 'vgginstruments_multi':
        testcsv = 'metadata/vgginstruments_multi_test.csv'
    elif args.testset == 'music_solo':
        testcsv = 'metadata/music_solo_test.csv'
    elif args.testset == 'music_duet':
        testcsv = 'metadata/music_duet_test.csv'

    else:
        raise NotImplementedError
    bbox_format = {'vggsound_single': 'vggsound_single',
                   'vggsound_duet': 'vggsound_duet',
                   'vgginstruments': 'vgginstruments',
                   'vgginstruments_multi': 'vgginstruments_multi',
                   'music_solo': 'music_solo',
                   'music_duet': 'music_duet'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    if args.testset in {'music_solo', 'music_duet'}:
        avail_audio_files = []
        for image_file in image_files:
            if image_file[:-10] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}

    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)
    print(f"{len(testset)} files for testing")

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    if args.testset in {'music_solo', 'music_duet'}:
        audio_files = [dt[:-10]+'.wav' for dt in testset]
    else:
        audio_files = [dt+'.wav' for dt in testset]

    # Bounding boxes
    print('bbox_format:', bbox_format)
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)
    all_bboxes = [all_bboxes[fn.split('.jpg')[0]] for fn in image_files]

    if 'num_test_samples' in vars(args) and args.num_test_samples is not None and args.num_test_samples > 0 and len(image_files) > args.num_test_samples:
        idx = random.sample(range(len(image_files)), k=args.num_test_samples)
        image_files = [image_files[i] for i in idx]
        audio_files = [audio_files[i] for i in idx]
        all_bboxes = {fn.split('.')[0]: all_bboxes[fn.split('.')[0]] for fn in image_files}
    
    # load non-sounding files
    if args.testset in ['flickr_plus_silent', 'vggss_plus_silent']:
        name_testset = args.testset.split('_')[0]
        for item in csv.reader(open(f'metadata/{name_testset}_test_plus_silent.csv')):
            if item[2] == 'non-sounding':
                image_files.append(f'{item[0]}.jpg')
                audio_files.append(f'{item[1]}.wav')
                all_bboxes.append([])

        idx = list(range(len(image_files)))
        random.shuffle(idx)
        image_files = [image_files[i] for i in idx]
        audio_files = [audio_files[i] for i in idx]
        all_bboxes = [all_bboxes[i] for i in idx]
    
    if args.testset in {'vggsound_single', 'vgginstruments', 'music_solo'}:
        class_labels = []
        all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
        num_classes = len(all_classes)
        fns2cls = {item[0]:item[1] for item in csv.reader(open(testcsv))}
        for dt in testset:
            cls = all_classes.index(fns2cls[dt])
            class_labels.append(cls)
        num_mixtures = 1
        class_labels_ss = None
        image_files_ss = None
        audio_files_ss = None
        all_bboxes_ss = None
    elif args.testset in {'vggsound_duet', 'vgginstruments_multi'}:
        class_labels = []
        class_labels_ss = []
        image_files_ss = []
        audio_files_ss = []
        all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
        num_classes = len(all_classes)
        fns2cls = {item[0]:item[1] for item in csv.reader(open(testcsv))}
        fns2mix = {item[0]:item[2] for item in csv.reader(open(testcsv))}
        for dt in testset:
            cls = all_classes.index(fns2cls[dt])
            class_labels.append(cls)
            dt_mix = fns2mix[dt]
            cls_mix = all_classes.index(fns2cls[dt_mix])
            class_labels_ss.append(cls_mix)
            image_files_ss.append([f'{dt_mix}.jpg'])
            audio_files_ss.append([f'{dt_mix}.wav'])
        num_mixtures = 2
        # Bounding boxes ss
        all_bboxes_ss = load_all_bboxes(args.test_gt_path, format=bbox_format)
        all_bboxes_ss = [all_bboxes_ss[fn[0].split('.jpg')[0]] for fn in image_files_ss]
    elif args.testset == 'music_duet':
        class_labels = []
        class_labels_ss = []
        image_files_ss = []
        audio_files_ss = []
        all_classes = []
        for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines():
            all_classes.append(line.split(',')[1])
            all_classes.append(line.split(',')[2])
        all_classes = sorted(set(all_classes))
        num_classes = len(all_classes)
        # Bounding boxes & classes
        all_bboxes_duet = load_all_bboxes(args.test_gt_path, format=bbox_format)
        for dt in testset:
            cls = all_classes.index(all_bboxes_duet[dt][1][0])
            class_labels.append(cls)
            cls_mix = all_classes.index(all_bboxes_duet[dt][1][1])
            class_labels_ss.append(cls_mix)
        num_mixtures = 2
        audio_files_ss = audio_files
        image_files_ss = image_files
        all_bboxes_ss = [all_bboxes_duet[dt][0] for dt in testset]
    else:
        num_classes = 0
        class_labels = None
        num_mixtures = 1
        class_labels_ss = None
        image_files_ss = None
        audio_files_ss = None
        all_bboxes_ss = None
        
    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='test',
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=5.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format,
        num_classes=num_classes,
        class_labels=class_labels,
        num_mixtures=num_mixtures,
        class_labels_ss=class_labels_ss,
        image_files_ss=image_files_ss, 
        audio_files_ss=audio_files_ss,
        all_bboxes_ss=all_bboxes_ss
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor

def convert_normalize(tensor, new_mean, new_std):
    raw_mean = IMAGENET_DEFAULT_MEAN
    raw_std = IMAGENET_DEFAULT_STD
    # inverse_normalize with raw mean & raw std
    inverse_mean = [-mean/std for mean, std in zip(raw_mean, raw_std)]
    inverse_std = [1.0/std for std in raw_std]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    # normalize with new mean & new std
    tensor = transforms.Normalize(new_mean, new_std)(tensor)
    return tensor