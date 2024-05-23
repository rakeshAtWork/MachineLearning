import pickle
import os
# Example object to be pickled
data = {'key': 'value', 'number': 42}

# Open a file in write-binary mode
cwd = os.getcwd()
pickle_file_path = os.path.join(cwd, 'data.pkl')
with open(pickle_file_path, 'wb') as file:
    pickle.dump(data, file)
