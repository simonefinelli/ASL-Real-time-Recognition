from tensorflow.keras.models import load_model
from data_utils import labels_to_number, videos_to_dict
from frame_generator import VideoFrameGenerator
from models import create_model_wlasl20c


# model settings
height = 224
width = 224
dim = (height, width)
batch_size = 8
frames = 10
channels = 3
output = 20

TEST_PATH = './data/test/'
# transform labels from string to number
labels = labels_to_number(TEST_PATH)
# load dataset as dict
y_test_dict = videos_to_dict(TEST_PATH, labels)
# get video paths (without labels)
X_test = list(y_test_dict.keys())

# load the best model after training
last = load_model('./saved_models/best_model.h5')

# instantiation of the model with best model's weights
model = create_model_wlasl20c(frames, width, height, channels, output)
model.compile(optimizer=last.optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.set_weights(last.get_weights())

# instantiation of test generator
print('\nTest generator')
test_generator = VideoFrameGenerator(
    list_IDs=X_test,
    labels=y_test_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=channels,
    n_sequence=frames,
    shuffle=False,
    type_gen='test'
)

# evaluate the best model on test set
print('\nEvaluating the best model on test set . . .')
eval_loss, eval_acc = model.evaluate(test_generator)
