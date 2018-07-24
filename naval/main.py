import sys
sys.path.insert(0,'../textgenrnn')

from textgenrnn import textgenrnn

textgen = textgenrnn()
epochs = 100
batch_size = 256
experiment_name = "epochs%d_bs%d" % (epochs, batch_size)
textgen.train_from_file('naval_tweets.csv', num_epochs=epochs, batch_size=batch_size)
textgen.save("naval_%s_weights.hdf5" % experiment_name)
textgen.generate_to_file("naval_%s_generated_texts" % experiment_name, n=10)
