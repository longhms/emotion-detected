import os
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from dataset import create_generators
from model import build_model
import tensorflow as tf 
from tensorflow.keras.utils import plot_model

import logging
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_TRT_ALLOW_FLATBUFFER'] = '1'

tf.get_logger().setLevel(logging.ERROR)


def train_model(root_dir, batch_size=64, epochs=20, output_model="emotion_model.h5"):
    result_folder = "result"

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    output_model_path = os.path.join(result_folder, output_model)
    history_csv_path = os.path.join(result_folder, "training_history.csv")
    
    train_gen, val_gen, _, _ = create_generators(root_dir, batch_size)
    
    x_train, y_train = next(train_gen)
    x_val, y_val = next(val_gen)

    print("Train batch shape: ", len(train_gen))
    print("Validation batch shape: ", (len(val_gen)))
    
    model = build_model()
    
    model_architecture_path = os.path.join(result_folder, 'model_architecture.png')
    plot_model(model, to_file=model_architecture_path, show_shapes=True, show_layer_names=True)
    
    model.compile(loss="categorical_crossentropy", 
                  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  metrics=['accuracy'])
        
    checkpoint = ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.25,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    )
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[
            lr_scheduler,
            checkpoint
        ]
    )
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)
    
    print("Training complete. Best model saved to:", output_model_path)
    print("\nTraining history saved to:", history_csv_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--output_model", type=str, default="emotion_model.h5", help="Output model file name")
    args = parser.parse_args()

    train_model(args.root, args.batch_size, args.epochs, args.output_model)
