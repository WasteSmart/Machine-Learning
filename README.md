# WasteSmart Machine Learning Development
These are the steps taken on creating the whole machine learning model from start to finish, the model is to be placed in the cloud and the format will be in model.json with 3 shards in the .bin format

## Data Preparation

### 1. Downloading the data
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sumn2u/garbage-classification-v2")
```
The data is downloaded from kagglehub directly to your drive

### 2. Cleaning the data
```
# Delete the non-JPEG/PNG files
for file_path in non_jpeg_png_files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except OSError as e:
        print(f"Error deleting {file_path}: {e}")

print(f"Deleted {len(non_jpeg_png_files)} non-JPEG/PNG files.")
```
The data is then cleaned because of the dataset having a few files that were in a different format from JPEGs and PNGs

### 3. Display the data
```
data_dir = pathlib.Path(PATH)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(f'Total image from this datasets : {image_count}')

print(f'\nDistribution Image')
for i, label in enumerate(os.listdir(data_dir)):
  label_dir = os.path.join(data_dir, label)
  len_label_dir = len(os.listdir(label_dir))
  print(f'{i+1}. {label} : {len_label_dir}')
```
Display the data from the dataset to display category labels and how many images in that category

## Data Preprocessing

### 1. Split data into training and validation datasets
```
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=seed,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=seed,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE)
```
The dataset needs to first be split into training and validation so that the model can train based on the data, but also data that the model cannot see but can validate.
### 2. Split validation data into validation and test datasets
```
val_batches = tf.data.experimental.cardinality(val_ds)
test_dataset = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))
```
We split the validation dataset to produce a test dataset to prove further the accuracy of our model

## Layers
### 1. Input Layer
```
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
```
We first create the input layer so that the model can receive the image to decipher which class it belongs to

### 2. Data Augmentation Layer
```
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomZoom(0.2)],
  name="data_augmentation")
```
The data is then augmented to improve the model's accuracy and provide more data by using the same data by just a modification to the data

### 3. Normalization Layer
```
normalization_layer = tf.keras.layers.Rescaling(1./255)
```
The model likes data that is normalized and it would help the model train faster and might even improve accuracy

### 4. MobileNetV2 Layer via Transfer Learning
```
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```
The MobileNetV2 helps a lot on accuracy of the model with its complex architecture we are able to stand on the shoulders of giants

### 5. Global Average Pooling Layer
```
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
```
This layer then downsamples the image from the MobileNetV2 convolutional layers

### 6. Dense Layer
```
  model.add(tf.keras.layers.Dense(units=256, activation="relu"))
```
These Dense Layers help with the accuracy of the model so that it can achieve better results, with a high number of units to increase the accuracy even further

### 7. Dropout Layer
```
  model.add(tf.keras.layers.Dropout(0.2))
```
The dropout layers are here to drop neurons to avoid the model from overfitting

## Callbacks
```
model_checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint_model.keras",
                                                      monitor="val_accuracy",
                                                      save_best_only=True,
                                                      verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=5,
                                                  mode ="max",
                                                  verbose=2,
                                                  restore_best_weights=True)
```
Callbacks are here to help stop the training when the next epochs become insignificant and also to have a checkpoint where the validation accuracy of the model was at its peak not when it has overfitted to the data

## Convertion to JSON
```
path_model_keras = r'C:\Users\Jonathan Suhalim\Documents\AILEARNING\TensorflowNotebooks\garbage_classification.h5'
load_model.save(path_model_keras)
path_model = "C:/Users/Jonathan Suhalim/Documents/AILEARNING/TensorflowNotebooks/garbage_classification.h5"
output_path = "C:/Users/Jonathan Suhalim/Documents/AILEARNING/TensorflowNotebooks/"

!tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model {path_model} {output_path}
```
Since the model is saved in a .keras format by default, a converter from the tfjs library is needed to convert it into json and its .bin files
