import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
import numpy as np
import cv2
import os

class AdvancedFaceRecognitionCNN:
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=512):
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self.build_advanced_model()
        
    def build_advanced_model(self):
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.embedding_dim, activation='linear', name='embeddings')
        ])
        
        return model
    
    def triplet_loss(self, margin=0.5):
        def loss(y_true, y_pred):
            anchor = y_pred[0::3]
            positive = y_pred[1::3] 
            negative = y_pred[2::3]
            
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            basic_loss = pos_dist - neg_dist + margin
            loss = tf.maximum(0.0, basic_loss)
            
            return tf.reduce_mean(loss)
        return loss
    
    def compile_model(self):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=self.triplet_loss(),
            metrics=['accuracy']
        )
    
    def preprocess_face(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def extract_embedding(self, image_path):
        preprocessed = self.preprocess_face(image_path)
        embedding = self.model.predict(preprocessed)
        return embedding[0]
    
    def cosine_similarity(self, embedding1, embedding2):
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)

class FaceDetectionPipeline:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cnn_model = AdvancedFaceRecognitionCNN()
        
    def detect_and_align_face(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            return face_resized
        return None
    
    def extract_face_features(self, image_path):
        aligned_face = self.detect_and_align_face(image_path)
        if aligned_face is not None:
            temp_path = 'temp_face.jpg'
            cv2.imwrite(temp_path, aligned_face)
            embedding = self.cnn_model.extract_embedding(temp_path)
            os.remove(temp_path)
            return embedding
        return None

class DataAugmentation:
    @staticmethod
    def augment_image(image):
        augmented_images = []
        
        augmented_images.append(cv2.flip(image, 1))
        
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
        
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -15, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
        
        brightness = 50
        bright_image = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * brightness)
        augmented_images.append(bright_image)
        
        dark_image = cv2.subtract(image, np.ones(image.shape, dtype=np.uint8) * brightness)
        augmented_images.append(dark_image)
        
        return augmented_images

class FaceRecognitionTrainer:
    def __init__(self, model):
        self.model = model
        self.augmentation = DataAugmentation()
    
    def create_triplets(self, image_paths, labels, num_triplets=1000):
        triplets = []
        unique_labels = list(set(labels))
        
        for _ in range(num_triplets):
            anchor_label = np.random.choice(unique_labels)
            anchor_indices = [i for i, label in enumerate(labels) if label == anchor_label]
            
            if len(anchor_indices) < 2:
                continue
                
            anchor_idx, positive_idx = np.random.choice(anchor_indices, 2, replace=False)
            
            negative_labels = [label for label in unique_labels if label != anchor_label]
            if not negative_labels:
                continue
                
            negative_label = np.random.choice(negative_labels)
            negative_indices = [i for i, label in enumerate(labels) if label == negative_label]
            negative_idx = np.random.choice(negative_indices)
            
            triplets.append((image_paths[anchor_idx], image_paths[positive_idx], image_paths[negative_idx]))
        
        return triplets
    
    def train_model(self, image_paths, labels, epochs=100, batch_size=32):
        triplets = self.create_triplets(image_paths, labels)
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for i in range(0, len(triplets), batch_size):
                batch_triplets = triplets[i:i+batch_size]
                batch_images = []
                
                for anchor_path, positive_path, negative_path in batch_triplets:
                    anchor_img = self.model.preprocess_face(anchor_path)
                    positive_img = self.model.preprocess_face(positive_path)
                    negative_img = self.model.preprocess_face(negative_path)
                    
                    batch_images.extend([anchor_img[0], positive_img[0], negative_img[0]])
                
                if batch_images:
                    batch_array = np.array(batch_images)
                    dummy_labels = np.zeros((len(batch_array),))
                    
                    loss = self.model.model.train_on_batch(batch_array, dummy_labels)
                    epoch_loss += loss
                    batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def save_trained_model(self, filepath):
        self.model.model.save(filepath)

class FaceRecognitionSystem:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.cnn_model = AdvancedFaceRecognitionCNN()
            self.cnn_model.compile_model()
            self.model = self.cnn_model.model
        
        self.face_pipeline = FaceDetectionPipeline()
        self.database_embeddings = {}
    
    def register_face(self, image_path, person_id):
        embedding = self.face_pipeline.extract_face_features(image_path)
        if embedding is not None:
            self.database_embeddings[person_id] = embedding
            return True
        return False
    
    def recognize_face(self, image_path, threshold=0.7):
        query_embedding = self.face_pipeline.extract_face_features(image_path)
        if query_embedding is None:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        for person_id, stored_embedding in self.database_embeddings.items():
            similarity = self.cnn_model.cosine_similarity(query_embedding, stored_embedding)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = person_id
        
        return best_match, best_similarity
    
    def batch_process_images(self, image_directory):
        processed_embeddings = {}
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, filename)
                embedding = self.face_pipeline.extract_face_features(image_path)
                
                if embedding is not None:
                    person_id = filename.split('.')[0]
                    processed_embeddings[person_id] = embedding
        
        return processed_embeddings

def train_custom_model(dataset_path):
    image_paths = []
    labels = []
    
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(person_path, image_file))
                    labels.append(person_folder)
    
    model = AdvancedFaceRecognitionCNN()
    model.compile_model()
    
    trainer = FaceRecognitionTrainer(model)
    trainer.train_model(image_paths, labels)
    trainer.save_trained_model('trained_face_model.h5')
    
    return model

if __name__ == "__main__":
    print("Initializing Advanced Face Recognition System...")
    
    face_system = FaceRecognitionSystem()
    print("Face Recognition System initialized successfully!")
    
    print("Model Architecture:")
    face_system.model.summary()