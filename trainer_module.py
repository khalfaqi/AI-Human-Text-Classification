
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = 'labels'
FEATURE_KEY = 'text'

def transformed_name(key):
    return key + '_xf'

def get_gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def get_input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=get_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def create_model(text_vectorization):
    dropout_rate = 0.2
    input_layer = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    vectorized_text = text_vectorization(input_layer)
    embedding_layer = layers.Embedding(input_dim=10000, output_dim=64, input_length=100)(vectorized_text)
    reshaped_embedded = tf.keras.layers.Reshape((1, 6400))(embedding_layer)
    dense_layer = layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform')(reshaped_embedded)
    attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(dense_layer, dense_layer, dense_layer)
    dropout_layer = layers.Dropout(dropout_rate)(attention_output)
    normalization_layer1 = layers.LayerNormalization(epsilon=1e-7)(dense_layer + dropout_layer)
    dense_layer2 = layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform")(normalization_layer1)
    dense_layer3 = layers.Dense(64, kernel_initializer='glorot_uniform')(dense_layer2)
    dropout_layer2 = layers.Dropout(dropout_rate)(dense_layer3)
    normalization_layer2 = layers.LayerNormalization(epsilon=1e-7)(normalization_layer1 + dropout_layer2)
    pooling_layer = layers.GlobalAveragePooling1D()(normalization_layer2)
    dropout_layer3 = layers.Dropout(dropout_rate)(pooling_layer)
    dense_layer4 = layers.Dense(32, activation="relu", kernel_initializer="glorot_uniform")(dropout_layer3)
    dropout_layer4 = layers.Dropout(dropout_rate)(dense_layer4)
    output_layer = layers.Dense(1, activation='sigmoid')(dropout_layer4)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()
    return model

def get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop("labels")
        parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    text_vectorization = layers.TextVectorization(max_tokens=10000, output_mode='int', output_sequence_length=100)

    # Prepare the dataset for adapting the TextVectorization layer
    adapt_dataset = get_input_fn(fn_args.train_files, tf_transform_output, num_epochs=1, batch_size=256)  # Use a larger batch size for quicker adaptation
    text_data_for_adaptation = adapt_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    text_vectorization.adapt(text_data_for_adaptation)

    # Create and compile the model after adapting TextVectorization
    model = create_model(text_vectorization)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=[tf.keras.metrics.BinaryAccuracy()])

    dataset_size = 10000 
    batch_size = 64
    steps_per_epoch = dataset_size // batch_size

    train_set = get_input_fn(fn_args.train_files, tf_transform_output, num_epochs=10, batch_size=64)
    val_set = get_input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10, batch_size=64)

    # Set up callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=0, patience=5)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=0, save_best_only=True)

    # Train the model
    model.fit(train_set, validation_data=val_set, epochs=10, steps_per_epoch=steps_per_epoch, validation_steps=100, callbacks=[tensorboard_callback, early_stopping, model_checkpoint])

    # Save the model with the signature for serving
    signatures = {
        'serving_default': get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
