import logging
from itertools import product
from re import findall

import numpy as np
import pandas as pd
from .corpus import EyeTrackingCorpus
from .utils import *


"""
Other data sets -- consider if they will help the model if we're just
EXTRACTING REPRESENTATIONS. Better to include these if for downstream task,
we're incorporating the image context.
- Talk 2017 - uses custom stimuli (objects shown within quadrants)
    - 3072 trials, 60Hz, 3s
- Baranes 2015 (Curiosity) - question answering stimuli
    - 2400 trials, 500Hz
- MIT-LOWRES - including gaze trials from low res images
    - around 9k trials

Winkler's website which lists a lot of data sets:
https://stefan.winkler.site/resources.html

Considerations:
- monocular vs binocular
- validity codes?


"""


class MIT_LearningToPredict(EyeTrackingCorpus):
    """
    Learning to predict where people look
    Tilke Judd, Krista Ehinger, Fredo Durand, Antonio Torralba
    International Conference on Computer Vision (ICCV) 2009

    779 landscape images and 228 portrait images

    Should have 1,003 images and 15 subjects
    = 15,045 trials

    Eye Tracker: ETL400 ISCAN
    Sampling Frequency: 240Hz
    Viewing Time: 3s
    Each sample = 240 * 3 = 720 points
    """
    def __init__(self, args):
        self.px_per_dva = 41.889  # reported in .mat files
        self.hz = 240
        self.w, self.h = (1280, 1024)
        # but image: long side of stimuli is 1024

        self.subjects = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw',
                         'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']
        self.root = 'MIT-learningtopredict/'
        self.stim_dir = self.root + 'ALLSTIMULI/'

        super(MIT_LearningToPredict, self).__init__(args)

    def extract(self):
        data = []
        images = listdir(self.root + 'ALLSTIMULI')
        for (subj, img) in product(self.subjects, images):
            mat = load(
                self.root + 'DATA/{}/{}'.format(
                    subj, img.replace('.jpeg', '')),
                'matlab')

            key = list(mat.keys())[-1]
            data.append([subj,
                         img,
                         'free-viewing',
                         # cut the data until 720 frames only
                         # by observation, samples after 720th frame
                         # are outliers
                         mat[key]['DATA'].item()['eyeData'].item().T[0][:720],
                         mat[key]['DATA'].item()['eyeData'].item().T[1][:720]])
        return data


class MIT_LowRes(EyeTrackingCorpus):
    """
    Fixations on Low-resolution Images
    Tilke Judd, Frédo Durand, Antonio Torralba
    JoV 2011

    64 subjects, 20~ images at 512px resolution
    = 1200~ trials

    Eye Tracker: ETL400 ISCAN (video-based)
    Sampling Frequency: 240Hz
    Viewing Time: 3s
    Each sample = 240 * 3 = 720 points
    average calibration error: less than 1 degree of visual angle (∼35 pixels)
    """
    def __init__(self, args):
        self.px_per_dva = 35  # stated in paper
        self.hz = 240
        self.w, self.h = (1280, 1024)  # stimuli is 860 x 1024
        self.root = 'MIT-LOWRES/DATA/'
        self.stim_dir = self.root.replace('DATA', 'ALLSTIMULI')

        super(MIT_LowRes, self).__init__(args)

    def extract(self):
        data = []
        for subj in listdir(self.root):
            subj_dir = self.root + '/' + subj
            for stim in listdir(subj_dir):
                # ignore non-natural and low resolution images...
                # or should i not?
                # if 'colornoise' in stim or '512.mat' not in stim:
                #     continue
                if stim.startswith('.'):
                    continue
                mat = load(subj_dir + '/' + stim, 'matlab')
                key = list(mat.keys())[-1]
                data.append([subj,
                             stim.replace('.mat', '.jpeg'),
                             'free-viewing',  # double check
                             mat[key]['DATA'].item()['eyeData'].item().T[0][:720],
                             mat[key]['DATA'].item()['eyeData'].item().T[1][:720]])
        return data


class MIT_CVCL(EyeTrackingCorpus):
    """
    (MIT Computational Visual Cognition Laboratory)
    Modeling search for people in 900 scenes:
    A combined source model of eye guidance
    Krista A. Ehinger, Barbara Hidalgo-Sotelo, Antonio Torralba, Aude Oliva1
    (cvcl.mit.edu/searchmodels/)

    14 subjects, 912 images, search for pedestrian
    = 12,768
    800 x 600 images = 23.5 x 17.7 DVA

    Eye Tracker: ISCAN RK-464 videobased eyetracker
    Sampling Frequency: 240Hz
    Viewing Time: mean 1050ms for correct-present and 1517ms for correct-absent

    "Estimated fixation position had to be within 0.758 of visual angle for
    all nine points, otherwise the experiment
    halted and the observer was recalibrated"

    "The first block consisted of the same 48 images for all participants, and
    was used as a practice block to verify that the eye could be
    tracked accurately. The experiment was composed of 19 blocks of 48 trials
    each and 50% target prevalence within each block"
    """
    def __init__(self, args):
        self.px_per_dva = 35
        self.w, self.h = (1024, 768)  # image is 800x600
        self.hz = 240
        self.root = 'MIT-CVCL/Dataset_EyeData/eyeData/'
        self.stim_dir = 'MIT-CVCL/Dataset_STIMULI/'

        super(MIT_CVCL, self).__init__(args)

    def extract(self):
        data = []
        for stim in listdir(self.root):
            mat = load(self.root + stim, 'matlab')
            key = list(mat.keys())[-1]
            subjects = mat[key]['subdata'].item()['subName']
            eyedata = mat[key]['subdata'].item()['eyedata']

            # eyedata has three columns, maybe third column is a validity code
            # (values 0, 1, 2, 3). Can't find what this 3rd column means.
            for subj, subj_eyedata in zip(subjects, eyedata):
                data.append([
                    subj,
                    stim.replace('.mat', '.jpg'),
                    'search',
                    list(subj_eyedata[:, 0]),
                    list(subj_eyedata[:, 1]),
                ])

        return np.array(data)


class DOVES(EyeTrackingCorpus):
    """
    Should have 101 images and 29 subjects
    = 2,929 trials

    Sampling Frequency: 200Hz
    Viewing Time: 5s
    """
    def __init__(self, args):
        self.w, self.h = (1024, 768)

        self.root = 'DOVES/RawData/'
        super(DOVES, self).__init__(args)

    def extract(self):
        def extract_trial(trial):
            trial = trial.split('\n')

            stimuli = trial[0]
            num_samples = trial[1]

            # should contain rows which have 3 space-separated values:
            # (x, y, ?)
            samples = trial[2:-1]

            if num_samples != len(samples):
                logging.info('Number of samples extracted does not match stated number!')

            coords = np.array(list(map(lambda x: x.split()[:2], samples)))
            return stimuli, 'free-viewing', coords[:, 0], coords[:, 1]

        data = []
        for subj in listdir(self.root):
            with open(self.root + subj, 'r') as f:
                text = f.read()

            for trial in findall(r'imk\d+\.iml[\n-.\d]+\[', text):
                data.append([subj, *extract_trial(trial)])

        return np.array(data)


class ETRA2019(EyeTrackingCorpus):
    """
    if just using natural scene images:
    8 subjects, 15 natural scenes, 2 tasks (fixation and free-viewing)
    = 240 trials

    Viewing Time: 45s
    Eye Tracker: EyeLink 1000
    Sampling Frequency: 500Hz
    """
    def __init__(self, args):
        self.w, self.h = (1024, 768)
        self.hz = 500
        self.px_per_dva = 35
        self.root = 'ETRA2019/data/{}'
        self.stim_dir = 'ETRA2019/images/'
        self.subjects = ['009', '019', '022', '058', '059',
                         '060', '062', 'SMC']

        # the grey stimuli is not included in the data set, here's code to
        # generate a grey image. copy paste grey.bmp to a new 'BLANK' folder.
        # from PIL import Image
        # Image.new('RGB', (921, 630), color=(127, 127, 127)).save('grey.bmp', quality=100)

        super(ETRA2019, self).__init__(args)

    def extract(self):
        data = []

        for subj in self.subjects:
            subj_dir = self.root.format(subj)
            for stim in listdir_(subj_dir):
                _, _, condition, stim_type, stim_file = stim.split('_')
                if condition == 'Fixation':
                    continue

                stim_folder = stim_type.upper()
                if stim_type in ['Blank', 'Natural']:
                    task = stim_type + '_free-viewing'
                else:  # Waldo, Puzzle
                    task = stim_type + '_search'
                # task = ('free-viewing' if stim_type in ['Blank', 'Natural']
                #         else 'search')

                csv = load(subj_dir + '/' + stim, 'csv', delimiter=',')

                data.append([
                    subj,
                    '/'.join([stim_folder, stim_file.replace('csv', 'bmp')]),
                    task,
                    ((csv['LXpix'] + csv['RXpix']) / 2).to_list(),
                    ((csv['LYpix'] + csv['RYpix']) / 2).to_list()
                ])
        return np.array(data)


class ETRA2019_Fixation(ETRA2019):
    def extract(self):
        data = []

        for subj in self.subjects:
            subj_dir = self.root.format(subj)
            for stim in listdir_(subj_dir):
                _, _, condition, stim_type, stim_file = stim.split('_')
                if condition != 'Fixation':
                    continue

                stim_folder = stim_type.upper()
                if stim_type in ['Blank', 'Natural']:
                    task = stim_type + '_free-viewing'
                else:  # Waldo, Puzzle
                    task = stim_type + '_search'
                # task = ('free-viewing' if stim_type in ['Blank', 'Natural']
                #         else 'search')

                csv = load(subj_dir + '/' + stim, 'csv', delimiter=',')

                data.append([
                    subj,
                    '/'.join([stim_folder, stim_file.replace('csv', 'bmp')]),
                    task,
                    ((csv['LXpix'] + csv['RXpix']) / 2).to_list(),
                    ((csv['LYpix'] + csv['RYpix']) / 2).to_list()
                ])
        return np.array(data)


class EMVIC2014(EyeTrackingCorpus):
    """
    Downloaded from:
    LPiTrack: Eye Movement Pattern Recognition Algorithm and Application to
    Biometric Identification

    But official link (need to register) is here:
    Also here: http://www.kasprowski.pl/emvic/dataset.php
    Official paper:
    https://www.researchgate.net/publication/266967887_The_Second_Eye_Movement_Verification_and_Identification_Competition_EMVIC

    Two .CSV files training and test (unlabeled) used to determine the winners of the competition.
    Every line is a list of comma separated elements as follows:
    sid, known,x1, y1, x2, y2, ... xn, yn
    where:
    sid - subject identifier (sXX)
    known - decision of the subject concerning the observed image (true/false).
    xi - the i-th value of the recorded horizontal eye gaze point.
    yi - the i-th value of the recorded vertical eye gaze point.
    The values are 0 for point in the middle, positive for point on the right or lower side of the screen and
    negative for points on the left or upper side of the screen.
    Contact: Subhadeep Mukhopadhyay, Email: deep@temple.edu

    34 subjects lookeda at 20 to 50 different photographs
    = 1430 samples (837 training set, 593 test set)

    Eye Tracker: JazzNovo
    Sampling Frequency: 1000Hz
    Viewing time: 891 to 22012ms
    Task: look at face and answer if subject knows the person
    """
    def __init__(self, args):
        self.w, self.h = (None, None)
        self.hz = 1000
        # self.root = 'EMVIC2014/Data-Suppl-material/'  # <-- for files given by LPiTrack
        self.root = 'EMVIC2014/official_files/'
        self.stim_dir = None

        super(EMVIC2014, self).__init__(args)

    def extract(self):
        data = []
        for split in ['train', 'testSolved']:
            with open(self.root + split + '.csv', 'r') as f:
                trials = [t.split(',') for t in f.read().split('\n')]

            for i, trial in enumerate(trials):
                if len(trial) == 1:
                    continue

                x = list(map(float, trial[2::2]))
                y = list(map(float, trial[3::2]))
                data.append([
                    # trial[0] if split == 'train' else 'test-subj-' + str(i),
                    trial[0] if split == 'train' else 'test-' + trial[0],
                    '',
                    'face',
                    # normalize so that (0, 0) is upper left of screen
                    x - np.min(x),
                    y - np.min(y)
                ])

        return np.array(data)


# class Kubota2012_WideField(EyeTrackingCorpus):
#     """
#     Hideyuki Kubota, Yusuke Sugano, Takahiro Okabe, Yoichi Sato,
#     Akihiro Sugimoto, and Kazuo Hiraki,
#     “Incorporating Visual Field Characteristics into a Saliency Map,”
#     in Proc. the 7th International Symposium on Eye Tracking Research & Applications (ETRA2012), March 2012.

#     FoV of humans is approx 200 degrees left and right, 120 degrees up and down.
#     Novelty of their data set: exploits 57 degrees of human's viewing angle.
#     (Other data sets only use at most 36 degrees)
#     15 subjects, 400 images
#     = 6,000 trials

#     Eye Tracker: Tobii TX300
#     Sampling Frequency: 60Hz
#     Viewing Time: 4s
#     """
#     def __init__(self, args):
#         self.w, self.h = (1366, 768)
#         self.root = 'utokyo-kubota2012/WideGazeDataset/rawFixation'

#         super(Kubota2012_WideField, self).__init__(args)

#     def extract(self):
#         data = []
#         for subj in listdir(self.root):
#             subj_dir = self.root + '/' + subj + '/'
#             for stim in listdir(subj_dir):
#                 csv = load(subj_dir + stim, 'csv',
#                                 delimiter=',', header=None)
#                 data.append([
#                     subj,
#                     stim,
#                     'free-viewing',
#                     csv[0].to_list(),
#                     csv[1].to_list()
#                 ])

#         return np.array(data)


# class Sugano2013_BSDS500(EyeTrackingCorpus):
#     """
#     Yusuke Sugano, Yasuyuki Matsushita and Yoichi Sato,
#     "Graph-based Joint Clustering of Fixations and Visual Entities"

#     10 subjects, 100 images
#     = 1,000 trials

#     Eye Tracker: Tobii TX300
#     Sampling Frequency: 60Hz
#     Viewing Time: 4s
#     """
#     def __init__(self, args):
#         self.w, self.h = (1366, 768)
#         self.root = 'utokyo-sugano2013/raw'

#         super(Sugano2013_BSDS500, self).__init__(args)

#     def extract(self):
#         data = []
#         for subj in listdir(self.root):
#             _dir = self.root + '/' + subj + '/'
#             for split in ['train', 'test', 'val']:
#                 __dir = _dir + split + '/'
#                 for stim in listdir(__dir):
#                     csv = load(__dir + stim, 'csv',
#                                     delimiter=',', header=None)
#                     data.append([
#                         subj,
#                         stim,
#                         'free-viewing',
#                         csv[0].to_list(),
#                         csv[1].to_list()
#                     ])

#         return np.array(data)


# class IRCCyN_IVC_Berkeley(EyeTrackingCorpus):
#     """
#     The IRCCyN IVC Berkeley Eyetracker database contains 80 images in 481x321 pixels.

#     24 subjects, 80 images
#     = 1,920 trials

#     Eye Tracker: Cambridge Research System Eyetracker
#     Sampling Frequency: 50Hz
#     Viewing Time: 4s
#     """
#     def __init__(self, args):
#         self.w, self.h = (1280, 1024)
#         self.root = 'IRCCyN_IVC_Eyetracker_Berkeley_Database/Eyetracker_Data/'

#         super(IRCCyN_IVC_Berkeley, self).__init__(args)

#     def extract(self):
#         data = []
#         for subj in listdir(self.root):
#             _dir = self.root + subj + '/'
#             for stim in listdir(_dir):
#                 if 'playlist' in stim:
#                     continue

#                 csv = load(_dir + stim, 'csv',
#                                 delimiter=',', skiprows=19)

#                 data.append([
#                     subj,
#                     stim,
#                     'free-viewing',
#                     # convert the ff. from abscissa/ordinate in millimeters
#                     # into pixels. Instructions from the accompanying PDF file.
#                     (csv['ScreenPositionXmm'] * 1920 / 810).to_list(),
#                     (csv['ScreenPositionYmm'] * 1080 / 460).to_list()
#                 ])

#         return np.array(data)


class IRCCyN_IVC_LIVE(EyeTrackingCorpus):
    """
    From UN
    21 subjects * 29 = 609 trials
    Eye Tracker: SMI Hi-Speed
    Sampling Frequency: 500Hz
    Viewing Time: 15s


    https://www.tilburguniversity.edu/sites/tiu/files/download/iViewX-System-Manual.pdf
    "Point of Regard is data mapped to calibration area"
    -- Have to double check this by comparing coordinates with actual stimuli
    (or by superimposition)

    Paper: A Comparative Study of Fixation Density Maps, Engelke 2013

    """
    def __init__(self, args):
        self.w, self.h = (1280, 1024)
        self.hz = 500
        self.px_per_dva = 41.8
        self.root = 'IRCCyN_IVC_Eyetracker_Images_LIVE_Database/Eyetracking_Data/'
        self.stim_dir = 'IRCCyN_IVC_Eyetracker_Images_LIVE_Database/Images/'

        super(IRCCyN_IVC_LIVE, self).__init__(args)

    def extract(self):
        data = []
        for trial in listdir(self.root):
            csv = load(self.root + trial, 'csv',
                            delimiter='\t', skiprows=42)

            trial = trial.split('_')
            # import pdb; pdb.set_trace()
            ave_x = ((csv['L POR X [px]'] + csv['R POR X [px]']) / 2).to_list()
            ave_y = ((csv['L POR Y [px]'] + csv['R POR Y [px]']) / 2).to_list()

            if len(np.unique(ave_x)) == 1:
                # no data
                continue

            data.append([
                trial[2],
                trial[1],
                'free-viewing',
                ((csv['L POR X [px]'] + csv['R POR X [px]']) / 2).to_list(),
                ((csv['L POR Y [px]'] + csv['R POR Y [px]']) / 2).to_list()
            ])

        return np.array(data)

    def _clean_trial(self, trial):
        booleans = np.logical_and(trial.x < 1280, trial.y < 1024)
        trial.x = trial.x[booleans]
        trial.y = trial.y[booleans]
        trial.x = trial.x - trial.x.min()
        trial.y = trial.y - trial.y.min()
        return trial


class Dorr2010_GazeCom(EyeTrackingCorpus):
    """
    From UN
    11 subjects * ? = 523 trials
    Eye Tracker:  SR Research EyeLink II
    Sampling Frequency: 250Hz
    Viewing Time: 2s

    "x and y and screen coordinates with (0,0) at the top left and
    (hres-1,vres-1) at the bottom right corner; horizontal and vertical
    resolution are specified in the header of the gaze file
    (in all these recordings, 1280 and 720, respectively)."
    """
    def __init__(self, args):
        self.w, self.h = (1280, 960)
        self.px_per_dva = 26.7
        self.hz = 250
        self.root = 'Dorr2010-GazeCom/static_images_gaze/'

        super(Dorr2010_GazeCom, self).__init__(args)

    def extract(self):
        data = []
        for trial in listdir(self.root):
            csv = load(self.root + trial, 'csv',
                            delimiter=' ', skiprows=2, header=None)

            trial = trial.split('_')
            data.append([
                trial[0],
                trial[1][:-6],
                'free-viewing',
                csv[1].to_list(),
                csv[2].to_list()
            ])

        return np.array(data)


class Cerf2007_FIFA(EyeTrackingCorpus):
    """
    7 subjects, 250 images, but 500 trials
    Eye Tracker:  Eyelink 1000
    Sampling Frequency: 1000 Hz
    Viewing Time:  2s
    Monocular (right eye)

    7 subjects viewed a set of 250 images (1024 × 768 pixels)
    in a three phase experiment:
    1. free-viewing (200 images)
    2. search (200 images), after a probe image shown for 600ms
    3. image recognition memory (100 images)

    250 images: 200 frontal view, 50 no face but identical scene?

    https://www.morancerf.com/publications

    "Predicting human gaze using low-level saliency combined with face
    detection",
    Cerf M., Harel J., Einhauser W., Koch C.,
    Neural Information Processing Systems (NIPS) 21, 2007

    "The stimuli includes images of faces in natural scenes, including
    a set of images with anomalous faces and scenes, and an additional
    database with a variety of exposures for each image for alternating
    luminance.​ 
    The faces database is a set of 1024x768 pixels images which contain
    frontal faces in various sizes, locations, skin colors, races, etc.
    Each frame includes one image with no faces for comparison.alternating
    luminance.​"

    8 subjects? x 200 images
    = 1600 trials

    """
    def __init__(self, args):
        self.w, self.h = (1024, 768)

        self.root = 'Cerf2007-FIFA/subjects/'
        self.stim_dir = 'Cerf2007-FIFA/stimuli/faces-jpg/'
        self.hz = 1000
        self.px_per_dva = 36.57  # approximated based on ratio with others' px_per_dva  

        # Got this list from the accompanying general.mat
        self.stimuli = [
            ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0022', '0023', '0024', '0025', '0026', '0028', '0030', '0034', '0035', '0036', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0053', '0055', '0060', '0061', '0062', '0063', '0065', '0069', '0070', '0071', '0074', '0075', '0076', '0082', '0083', '0085', '0091', '0092', '0106', '0113', '0115', '0116', '0117', '0118', '0119', '0120', '0122', '0123', '0124', '0125', '0126', '0127', '0131', '0132', '0133', '0134', '0135', '0136', '0138', '0139', '0140', '0141', '0142', '0143', '0144', '0145', '0146', '0147', '0148', '0149', '0150', '0151', '0152', '0153', '0154', '0155', '0158', '0159', '0160', '0161', '0162', '0163', '0164', '0165', '0166', '0167', '0168', '0169', '0170', '0171', '0172', '0173', '0174', '0175', '0176', '0177', '0178', '0179', '0180', '0181', '0182', '0183', '0184', '0185', '0186', '0187', '0188', '0189', '0190', '0191', '0192', '0193', '0194', '0195', '0196', '0197', '0198', '0199', '0200', '0201', '0202', '0203', '0204', '0205', '0206', '0207', '0208', '0209', '0210', '0211', '0212', '0213', '0214', '0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222', '0223', '0224', '0225', '0226', '0227', '0228', '0229', '0230', '0231', '0232', '0233', '0234', '0235', '0236', '0237', '0238', '0239', '0240', '0241', '0242', '0243', '0244', '0245', '0246', '0247', '0248', '0249', '0250', '0251', '0252', '0253', '0254', '0255', '0256', '0257', '0258', '0259', '0260', '0261', '0262', '0263', '0264', '0265', '0266', '0267', '0268', '0269', '0270'],
            ['0134', '0135', '0154', '0147', '0126', '0143', '0144', '0123', '0150', '0142', '0152', '0124', '0145', '0053', '0106', '0125', '0148', '0136', '0149', '0139', '0132', '0155', '0140', '0146', '0138', '0141', '0131', '0122', '0127', '0133', '0033', '0052', '0107', '0031', '0066', '0019', '0097', '0099', '0102', '0048', '0072', '0112', '0018', '0020', '0050', '0089', '0093', '0014', '0016', '0077', '0084', '0087', '0095', '0104', '0110', '0029', '0098', '0027', '0086', '0101', '0094', '0111', '0137', '0032', '0051', '0054', '0064', '0096', '0073', '0081', '0156', '0049', '0090', '0103', '0105', '0017', '0037', '0068', '0109', '0157', '0007', '0022', '0042', '0044', '0063', '0069', '0038', '0045', '0071', '0118', '0119', '0120', '0002', '0003', '0006', '0025', '0113', '0115', '0034', '0074', '0075', '0082', '0083', '0151', '0009', '0023', '0024', '0092', '0116', '0153', '0001', '0026', '0040', '0047', '0076', '0195', '0011', '0046', '0055', '0062', '0197', '0232', '0004', '0005', '0008', '0035', '0041', '0065', '0028', '0030', '0036', '0060', '0061', '0117', '0010', '0039', '0043', '0070', '0085', '0091', '0004', '0025', '0091', '0008', '0035', '0082', '0023', '0115', '0119', '0120', '0151', '0153', '0030', '0044', '0045', '0047', '0076', '0092', '0026', '0034', '0040', '0069', '0070', '0085', '0001', '0002', '0003', '0010', '0011', '0116', '0195', '0197', '0232', '0006', '0041', '0043', '0007', '0009', '0022', '0024', '0028', '0063', '0005', '0039', '0055', '0060', '0071', '0083', '0036', '0046', '0061', '0062', '0065', '0113', '0038', '0042', '0074', '0075', '0117', '0118'],
        ]

        super(Cerf2007_FIFA, self).__init__(args)

    def extract(self):
        data = []
        _tmp = {}
        # Paper states 7 subjects, but has data for 8
        # (general.mat lists 34 subjects though...)
        for subj in listdir(self.root):
            if subj.startswith('_'):
                continue

            mat = load(self.root + subj, 'matlab')

            subj = mat['name']
            # 4 experiments in file, but only 3 has gaze data.
            # the third is supposedly the memory task, but it contains data
            # for 200 images, so idk what it's for.
            # decision: use the first 2 phases only.
            for phase, experiment in enumerate(mat['experiments'][:2]):
                # not the filenames but really just the order of the index
                # stimuli_order = experiment['order'].item()
                stimuli_data = experiment['scan'].item()

                task = ('free-viewing' if phase == 0
                        else ('search' if phase == 1
                              else 'memory'))

                # _tmp[subj + str(phase)] = stimuli_order

                for n, eye_data in enumerate(stimuli_data):
                    data.append([
                        subj,
                        self.stimuli[phase][n] + '.jpg',
                        task,
                        list(eye_data['scan_x'].item()),
                        list(eye_data['scan_y'].item()),
                    ])
            # TO-DO: for the 2nd phase, need to filter out probe image data
        import pdb; pdb.set_trace()
        return np.array(data)
