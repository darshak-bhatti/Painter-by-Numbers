'''
make_submission() generates predictions for the
Kaggle Painter by Numbers competition using simple
features (image size, aspect ratio and bits/pixel^2)

Note: Make sure you're in the directory containing your data files before
running this script.

THIS CODE IS FROM THE KAGGLE COMPETITION, NOT WRITTEN BY US

Some parts are modified to incorporate more features besides aspect ratio, and picture width and height
'''

import numpy as np
import pandas as pd
from PIL import Image
import os


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def make_pairs(train_info):
    artists = train_info.artist.unique()

    n = train_info.groupby('artist').size()
    n = (2 * n**2).sum()
    pairs_cols = ['artist1', 'image1', 'artist2', 'image2']
    t = pd.DataFrame(np.zeros((n, 4)), columns=pairs_cols)
    i = 0
    j = 0
    for m in artists:

        a = train_info[train_info.artist == m][
            ['artist', 'new_filename']].values
        use = train_info[train_info.artist != m].index.values
        np.random.shuffle(use)
        nm = np.min(
            [a.shape[0]**2, train_info[train_info.artist != m].shape[0]])
        use = use[0:nm]
        b = train_info[train_info.artist != m][
            ['artist', 'new_filename']].ix[use, :].values

        a2 = pd.DataFrame(np.concatenate([np.repeat(a[:, 0], a.shape[0]).reshape((-1, 1)),
                                         np.repeat(a[:, 1], a.shape[0]).reshape((-1, 1)),
                                         np.tile(a, (a.shape[0], 1))], axis=1),
                                         columns=pairs_cols)
        a2 = a2.loc[0:nm, :]
        b2 = pd.DataFrame(np.concatenate([np.tile(a, (a.shape[0], 1))[0:b.shape[0], :], b],
                                         axis=1), columns=pairs_cols)
        # print j, i, a2.shape[0], b2.shape[0]
        # print b2
        t.iloc[i:i + a2.shape[0], :] = a2.values
        t.iloc[i + a2.shape[0]:i + a2.shape[0] + b2.shape[0], :] = b2.values
        i += a2.shape[0] + b2.shape[0]
        j += 1

    t = t[~t.image2.isin([np.nan, 0])]
    return t[t.image1 > t.image2]

def train_classifier(x_train, y_train, x_cv, y_cv):
    clf = RandomForestClassifier(n_estimators=15)

    print 'starting fit'

    clf.fit(x_train[::5], y_train[::5])
    print 'starting pred'

    y_pred = np.zeros(x_cv.shape[0])
    for i in xrange(4):
        y_pred[i::4] = clf.predict_proba(x_cv[i::4])[:, 1]

    if y_cv is not None:
        print roc_auc_score(y_cv, y_pred)

    return y_pred, clf


def prep_data(input, split):
    info = input[0]
    data = input[1]

    if split == 'cv':
        artists = info.artist.unique()
        np.random.shuffle(artists)

        info = get_image_info(info, 'train')
        info['bytes_per_pixel'] = 1.0 * info['size_bytes'] / \
            (info['pixelsx'] * info['pixelsy'])
        info['aspect_ratio'] = 1.0 * info['pixelsx'] / info['pixelsy']
        train_artists = artists[0:int(0.8 * len(artists))]
        test_artists = artists[int(0.8 * len(artists)):]

        train = make_pairs(info[info.artist.isin(train_artists)])
        test = make_pairs(info[info.artist.isin(test_artists)])
        train['in_train'] = True
        test['in_train'] = False
        data = train.append(test)
        data['sameArtist'] = data['artist1'] == data['artist2']

    if split == 'test':

        info = get_image_info(data, 'test')
        pixelsxy = info['pixelsx'] * info['pixelsy']
        info['bytes_per_pixel'] = 1.0 * info['size_bytes'] / pixelsxy
        info['aspect_ratio'] = 1.0 * info['pixelsx'] / info['pixelsy']

        data['in_train'] = False

        if 'artist1' in data.columns:
            data['sameArtist'] = data['artist1'] == data['artist2']

    columns = ['new_filename', 'pixelsx', 'pixelsy', 'size_bytes',
               'bytes_per_pixel', 'aspect_ratio']
    data2 = pd.merge(data, info[columns], how='left', left_on='image1',
                     right_on='new_filename')
    data2.drop('new_filename', 1, inplace=True)

    data2 = pd.merge(data2, info[columns], how='left', left_on='image2',
                     right_on='new_filename')
    data2.drop('new_filename', 1, inplace=True)

    columns2 = ['pixelsx_x', 'pixelsy_x', 'size_bytes_x', 'bytes_per_pixel_x',
                'aspect_ratio_x', 'pixelsx_y', 'pixelsy_y', 'size_bytes_y',
                'bytes_per_pixel_y', 'aspect_ratio_y']
    # As long as "data.in_train" is an np.array this should work:
    x_train = data2[data2.in_train][columns2].values
    x_test = data2[~data2.in_train][columns2].values

    if 'artist1' in data.columns:
        y_train = data2[data2.in_train]['sameArtist'].values
        y_test = data2[~data2.in_train]['sameArtist'].values
    else:
        y_test = None

    if split == 'cv':
        return x_train, y_train, x_test, y_test
    if split == 'test':
        return x_test, y_test


def get_image_info(test_info, dir):
    if dir == 'test':
        images = list(set(list(test_info.image1.unique()) +
                          list(test_info.image2.unique())))
        info = pd.DataFrame(np.array(images).reshape(
            (-1, 1)), columns=['filename'])
    else:
        info = test_info

    info['pixelsx'] = np.nan
    info['pixelsy'] = np.nan
    info['size_bytes'] = np.nan

    for i in info.index.values:
        try:
            im = Image.open(dir + '/' + info.loc[i, 'filename'])
            info.loc[i, 'pixelsx'], info.loc[i, 'pixelsy'] = im.size
            # im = cv2.imread(dir+'/'+info.loc[i, 'new_filename'])
            # info.loc[i, 'pixelsx'], info.loc[i, 'pixelsy'] = im.shape[0:2]
            info.loc[i, 'size_bytes'] = os.path.getsize(
                dir + '/' + info.loc[i, 'filename'])

        except:
            print dir + '/' + info.loc[i, 'filename']

    return info.rename(columns={'filename': 'new_filename'})


def make_submission():
    # Read data from provided train_info.csv
    TInfo = pd.read_csv('train_info.csv')
    # Read data from provided submission_info.csv
    submission_info = pd.read_csv('submission_info.csv')
    
    # Prepare the data from grabed from train_info.csv
    x_train, y_train, x_cv, y_cv = prep_data([TInfo, None], 'cv')
    print 'prepping test data'
    x_test, y_test = prep_data([None, submission_info], 'test')

    print 'starting classifier'
    y_pred, clf = train_classifier(x_train, y_train, x_test, y_test)

    submission = submission_info[['index']]
    submission['sameArtist'] = y_pred                 # Ignore the warning from this line, it still creates the csv file correctly
    submission.to_csv('submission.csv', index=False)
    return


make_submission()
