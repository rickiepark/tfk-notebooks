import numpy as np
from sklearn.model_selection import train_test_split

sound_data = np.load('urban_sound.npz')
X_data = sound_data['X']
y_data = sound_data['y']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

np.savez('urban_sound_train', X=X_train, y=y_train)
np.savez('urban_sound_test', X=X_test, y=y_test)
