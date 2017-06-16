import glob
import numpy as np

X = np.empty((0, 193))
y = np.empty((0, 10))
groups = np.empty((0, 1))
npz_files = glob.glob('./urban_sound_?.npz')
for fn in npz_files:
    print(fn)
    data = np.load(fn)
    X = np.append(X, data['X'], axis=0)
    y = np.append(y, data['y'], axis=0)
    groups = np.append(groups, data['groups'], axis=0)

print(groups[groups>0])

print(X.shape, y.shape)
for r in y:
    if np.sum(r) > 1.5:
        print(r)
np.savez('urban_sound', X=X, y=y, groups=groups)
