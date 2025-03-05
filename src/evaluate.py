import os
import numpy as np
from tensorflow.keras.models import load_model
from dataset import create_generators
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
CLASS_NAMES = ['Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def evaluate_model(model_path, root_dir, batch_size=64):
    _, _, test_gen, test_labels = create_generators(root_dir, batch_size)

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    predictions = model.predict(test_gen)
    
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    report = classification_report(
        true_labels, predicted_labels, target_names=CLASS_NAMES, zero_division=1)
    print("Classification Report:")
    print(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (e.g., emotion_model.h5)")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.root, args.batch_size)
