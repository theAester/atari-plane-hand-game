import pickle
import numpy as np

open_data = pickle.load(open('./open.pickle', 'rb'))
close_data = pickle.load(open('./closed.pickle', 'rb'))

for a in open_data['open']:
    print(len(a))

open_a = np.asarray(open_data['open'])
close_a = np.asarray(close_data['closed'])

data = []
labels = []

for a in open_a:
    data.append(a)
    labels.append(0)

for a in close_a:
    data.append(a)
    labels.append(1)

file = open('./data.pickle', 'wb')
pickle.dump({'data':data, 'labels':labels}, file)
file.close()

