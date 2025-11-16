"""
Audio-Based Active Learning for Voice Authentication
Using Bayesian Active Learning with BALD acquisition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import streamlit as st
from typing import List, Tuple, Dict
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class VoiceDataset(Dataset):
    """Custom dataset for voice samples"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class VoiceFeatureExtractor:
    """Extract MFCC features from audio"""
    
    def __init__(self, n_mfcc: int = 40, sr: int = 16000):
        self.n_mfcc = n_mfcc
        self.sr = sr
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            
            # Compute statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_max = np.max(mfccs, axis=1)
            mfcc_min = np.min(mfccs, axis=1)
            
            # Concatenate features
            features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
            
            return features
        
        except Exception as e:
            raise Exception(f"Feature extraction error: {str(e)}")
    
    def extract_from_array(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract features from numpy array"""
        if sr is None:
            sr = self.sr
            
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
        
        return features

class VoiceNet(nn.Module):
    """Neural network for voice authentication with Monte Carlo Dropout"""
    
    def __init__(self, input_dim: int = 160, hidden_dims: List[int] = [256, 128, 64], 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(VoiceNet, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def mc_dropout_predict(self, x, n_samples: int = 50):
        """Monte Carlo Dropout for uncertainty estimation"""
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = F.softmax(self.forward(x), dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

class BALDAcquisition:
    """Bayesian Active Learning by Disagreement (BALD) acquisition function"""
    
    def __init__(self, model: VoiceNet, n_mc_samples: int = 50):
        self.model = model
        self.n_mc_samples = n_mc_samples
    
    def calculate_bald_score(self, features: torch.Tensor) -> np.ndarray:
        """Calculate BALD acquisition scores"""
        self.model.eval()
        
        # MC Dropout predictions
        predictions = []
        for _ in range(self.n_mc_samples):
            self.model.train()  # Enable dropout
            with torch.no_grad():
                pred = F.softmax(self.model(features), dim=1)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # Shape: (n_mc_samples, n_samples, n_classes)
        
        # Calculate entropy of mean predictions (H[E[p(y|x)]])
        mean_pred = predictions.mean(axis=0)
        entropy_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
        
        # Calculate expected entropy (E[H[p(y|x)]])
        entropy_samples = -np.sum(predictions * np.log(predictions + 1e-10), axis=2)
        expected_entropy = entropy_samples.mean(axis=0)
        
        # BALD score = mutual information
        bald_scores = entropy_mean - expected_entropy
        
        return bald_scores
    
    def select_samples(self, unlabeled_features: np.ndarray, n_samples: int = 5) -> List[int]:
        """Select most informative samples using BALD"""
        features_tensor = torch.FloatTensor(unlabeled_features)
        
        bald_scores = self.calculate_bald_score(features_tensor)
        
        # Select top samples with highest BALD scores
        selected_indices = np.argsort(bald_scores)[-n_samples:][::-1]
        
        return selected_indices.tolist(), bald_scores

class VoiceAuthenticationSystem:
    """Complete voice authentication system with active learning"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.feature_extractor = VoiceFeatureExtractor(sr=sample_rate)
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
        # Active learning state
        self.labeled_features = []
        self.labeled_labels = []
        self.unlabeled_features = []
        self.unlabeled_paths = []
        
        # Statistics
        self.training_history = []
        self.accuracy_history = []
        self.samples_labeled = 0
    
    def record_audio(self, duration: int = 3, save_path: str = None) -> np.ndarray:
        """Record audio from microphone"""
        st.info(f"🎤 Recording for {duration} seconds...")
        
        audio = sd.rec(int(duration * self.sample_rate), 
                      samplerate=self.sample_rate, 
                      channels=1,
                      dtype='float32')
        sd.wait()
        
        audio = audio.flatten()
        
        if save_path:
            sf.write(save_path, audio, self.sample_rate)
        
        return audio
    
    def collect_voice_samples(self, user_name: str, n_samples: int = 20) -> List[str]:
        """Collect voice samples from user"""
        samples_dir = f"./voice_samples/{user_name}"
        os.makedirs(samples_dir, exist_ok=True)
        
        sample_paths = []
        
        for i in range(n_samples):
            st.write(f"Sample {i+1}/{n_samples}")
            
            # Record audio
            audio_path = os.path.join(samples_dir, f"sample_{i+1}.wav")
            audio = self.record_audio(duration=3, save_path=audio_path)
            
            sample_paths.append(audio_path)
        
        return sample_paths
    
    def process_samples(self, sample_paths: List[str], labels: List[int] = None):
        """Process audio samples and extract features"""
        features = []
        
        for path in sample_paths:
            feat = self.feature_extractor.extract_features(path)
            features.append(feat)
        
        features = np.array(features)
        
        if labels is not None:
            self.labeled_features.extend(features)
            self.labeled_labels.extend(labels)
        else:
            self.unlabeled_features.extend(features)
            self.unlabeled_paths.extend(sample_paths)
        
        return features
    
    def initialize_model(self, input_dim: int = 160):
        """Initialize the VoiceNet model"""
        self.model = VoiceNet(input_dim=input_dim)
        
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score normalization"""
        if fit:
            self.scaler_mean = features.mean(axis=0)
            self.scaler_std = features.std(axis=0) + 1e-8
        
        normalized = (features - self.scaler_mean) / self.scaler_std
        return normalized
    
    def train(self, epochs: int = 50, batch_size: int = 16, learning_rate: float = 0.001):
        """Train the model on labeled data"""
        if len(self.labeled_features) == 0:
            raise Exception("No labeled data available for training")
        
        # Prepare data
        features = np.array(self.labeled_features)
        labels = np.array(self.labeled_labels)
        
        # Normalize
        features = self.normalize_features(features, fit=True)
        
        # Create dataset
        dataset = VoiceDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model if not exists
        if self.model is None:
            self.initialize_model(input_dim=features.shape[1])
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            accuracy = 100. * correct / total
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': total_loss / len(dataloader),
                'accuracy': accuracy
            })
            
            if (epoch + 1) % 10 == 0:
                st.write(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
        
        self.accuracy_history.append(accuracy)
    
    def active_learning_iteration(self, n_samples: int = 5) -> List[int]:
        """Perform one iteration of active learning"""
        if len(self.unlabeled_features) == 0:
            st.warning("No unlabeled samples available")
            return []
        
        # Initialize BALD acquisition
        bald = BALDAcquisition(self.model, n_mc_samples=50)
        
        # Select most informative samples
        unlabeled_array = np.array(self.unlabeled_features)
        unlabeled_norm = self.normalize_features(unlabeled_array)
        
        selected_indices, bald_scores = bald.select_samples(unlabeled_norm, n_samples=n_samples)
        
        return selected_indices, bald_scores
    
    def label_samples(self, indices: List[int], labels: List[int]):
        """Move samples from unlabeled to labeled pool"""
        for idx, label in zip(indices, labels):
            self.labeled_features.append(self.unlabeled_features[idx])
            self.labeled_labels.append(label)
            self.samples_labeled += 1
        
        # Remove from unlabeled pool (in reverse order to maintain indices)
        for idx in sorted(indices, reverse=True):
            del self.unlabeled_features[idx]
            del self.unlabeled_paths[idx]
    
    def authenticate(self, audio_path: str = None, audio_array: np.ndarray = None) -> Dict:
        """Authenticate a voice sample"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        # Extract features
        if audio_path:
            features = self.feature_extractor.extract_features(audio_path)
        elif audio_array is not None:
            features = self.feature_extractor.extract_from_array(audio_array)
        else:
            raise ValueError("Either audio_path or audio_array must be provided")
        
        # Normalize
        features = self.normalize_features(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features)
        
        # MC Dropout prediction
        mean_pred, uncertainty = self.model.mc_dropout_predict(features_tensor, n_samples=50)
        
        # Get prediction
        confidence, predicted_class = mean_pred.max(1)
        
        return {
            'predicted_class': predicted_class.item(),
            'confidence': confidence.item(),
            'uncertainty': uncertainty[0, predicted_class].item(),
            'is_authentic': predicted_class.item() == 1,
            'raw_probabilities': mean_pred[0].tolist()
        }
    
    def save_model(self, path: str = "./models/voice_auth.pkl"):
        """Save the model and system state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std,
            'labeled_features': self.labeled_features,
            'labeled_labels': self.labeled_labels,
            'training_history': self.training_history,
            'accuracy_history': self.accuracy_history,
            'samples_labeled': self.samples_labeled
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_model(self, path: str = "./models/voice_auth.pkl"):
        """Load the model and system state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        if state['model_state']:
            self.initialize_model()
            self.model.load_state_dict(state['model_state'])
        
        self.scaler_mean = state['scaler_mean']
        self.scaler_std = state['scaler_std']
        self.labeled_features = state['labeled_features']
        self.labeled_labels = state['labeled_labels']
        self.training_history = state['training_history']
        self.accuracy_history = state['accuracy_history']
        self.samples_labeled = state['samples_labeled']

def main():
    st.set_page_config(page_title="Voice Authentication System", layout="wide")
    
    st.title("🎙️ Voice Authentication with Active Learning")
    st.markdown("*Bayesian Active Learning by Disagreement (BALD) for efficient voice authentication*")
    
    # Initialize system
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = VoiceAuthenticationSystem()
    
    auth_system = st.session_state.auth_system
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        st.metric("Labeled Samples", len(auth_system.labeled_features))
        st.metric("Unlabeled Samples", len(auth_system.unlabeled_features))
        st.metric("Total Labeled", auth_system.samples_labeled)
        
        if auth_system.accuracy_history:
            st.metric("Latest Accuracy", f"{auth_system.accuracy_history[-1]:.2f}%")
        
        st.markdown("---")
        
        if st.button("💾 Save Model"):
            auth_system.save_model()
            st.success("Model saved!")
        
        if st.button("📂 Load Model"):
            try:
                auth_system.load_model()
                st.success("Model loaded!")
            except:
                st.error("No saved model found")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Setup", "🔄 Active Learning", "✅ Authenticate", "📈 Analytics"])
    
    with tab1:
        st.header("Initial Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Collect Authentic Samples")
            user_name = st.text_input("User Name", value="user1")
            n_auth_samples = st.number_input("Number of authentic samples", min_value=5, max_value=20, value=10)
            
            if st.button("🎤 Record Authentic Samples"):
                with st.spinner("Recording..."):
                    paths = []
                    for i in range(n_auth_samples):
                        st.write(f"Recording sample {i+1}/{n_auth_samples}")
                        audio = auth_system.record_audio(duration=3)
                        path = f"./temp_auth_{i}.wav"
                        sf.write(path, audio, auth_system.sample_rate)
                        paths.append(path)
                    
                    auth_system.process_samples(paths, labels=[1] * n_auth_samples)
                    st.success(f"✅ Collected {n_auth_samples} authentic samples")
        
        with col2:
            st.subheader("Collect Imposter Samples")
            n_imp_samples = st.number_input("Number of imposter samples", min_value=5, max_value=20, value=10)
            
            if st.button("🎤 Record Imposter Samples"):
                with st.spinner("Recording..."):
                    paths = []
                    for i in range(n_imp_samples):
                        st.write(f"Recording sample {i+1}/{n_imp_samples}")
                        audio = auth_system.record_audio(duration=3)
                        path = f"./temp_imp_{i}.wav"
                        sf.write(path, audio, auth_system.sample_rate)
                        paths.append(path)
                    
                    auth_system.process_samples(paths, labels=[0] * n_imp_samples)
                    st.success(f"✅ Collected {n_imp_samples} imposter samples")
        
        st.markdown("---")
        
        if st.button("🚀 Train Initial Model", type="primary"):
            if len(auth_system.labeled_features) < 10:
                st.error("Need at least 10 labeled samples to train")
            else:
                with st.spinner("Training model..."):
                    auth_system.train(epochs=50, batch_size=8)
                st.success("✅ Model trained successfully!")
    
    with tab2:
        st.header("Active Learning Iteration")
        
        st.markdown("""
        Active learning selects the most informative samples for labeling,
        reducing the number of labels needed by up to 50%.
        """)
        
        n_select = st.slider("Samples to select", min_value=1, max_value=10, value=5)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Select Most Informative Samples"):
                if len(auth_system.unlabeled_features) == 0:
                    st.warning("No unlabeled samples. Please add some first.")
                else:
                    with st.spinner("Calculating BALD scores..."):
                        indices, scores = auth_system.active_learning_iteration(n_samples=n_select)
                        
                        st.session_state.selected_indices = indices
                        st.session_state.bald_scores = scores
                        
                        st.success(f"✅ Selected {len(indices)} samples")
                        
                        # Display scores
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.bar(range(len(scores)), scores)
                        ax.axhline(y=scores[indices[0]] if len(indices) > 0 else 0, 
                                  color='r', linestyle='--', label='Selection threshold')
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel("BALD Score (Uncertainty)")
                        ax.set_title("Active Learning: BALD Acquisition Scores")
                        ax.legend()
                        st.pyplot(fig)
        
        with col2:
            st.subheader("Label Selected Samples")
            
            if hasattr(st.session_state, 'selected_indices'):
                for idx in st.session_state.selected_indices[:3]:  # Show first 3
                    st.write(f"Sample {idx}")
                    label = st.radio(f"Label for sample {idx}", 
                                   ["Authentic", "Imposter"], 
                                   key=f"label_{idx}")
                
                if st.button("✅ Submit Labels"):
                    labels = []
                    for idx in st.session_state.selected_indices:
                        label = st.session_state.get(f"label_{idx}", "Authentic")
                        labels.append(1 if label == "Authentic" else 0)
                    
                    auth_system.label_samples(st.session_state.selected_indices, labels)
                    
                    # Retrain
                    with st.spinner("Retraining model..."):
                        auth_system.train(epochs=30, batch_size=8)
                    
                    st.success("✅ Labels submitted and model retrained!")
    
    with tab3:
        st.header("Voice Authentication")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Record Voice")
            
            if st.button("🎤 Record for Authentication"):
                with st.spinner("Recording..."):
                    audio = auth_system.record_audio(duration=3)
                    st.session_state.test_audio = audio
                    st.success("✅ Recording complete")
            
            if hasattr(st.session_state, 'test_audio'):
                if st.button("🔐 Authenticate"):
                    with st.spinner("Authenticating..."):
                        result = auth_system.authenticate(audio_array=st.session_state.test_audio)
                        st.session_state.auth_result = result
        
        with col2:
            if hasattr(st.session_state, 'auth_result'):
                result = st.session_state.auth_result
                
                if result['is_authentic']:
                    st.success("✅ AUTHENTICATED")
                else:
                    st.error("❌ REJECTED")
                
                st.metric("Confidence", f"{result['confidence']:.2%}")
                st.metric("Uncertainty", f"{result['uncertainty']:.4f}")
                
                # Show probabilities
                st.bar_chart({
                    'Imposter': result['raw_probabilities'][0],
                    'Authentic': result['raw_probabilities'][1]
                })
    
    with tab4:
        st.header("System Analytics")
        
        if auth_system.training_history:
            # Training history
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Loss")
                losses = [h['loss'] for h in auth_system.training_history]
                st.line_chart(losses)
            
            with col2:
                st.subheader("Training Accuracy")
                accs = [h['accuracy'] for h in auth_system.training_history]
                st.line_chart(accs)
            
            # Show final stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Final Accuracy", f"{auth_system.accuracy_history[-1]:.2f}%")
            
            with col2:
                reduction = (1 - auth_system.samples_labeled / max(len(auth_system.labeled_features), 1)) * 100
                st.metric("Label Reduction", f"{reduction:.1f}%")
            
            with col3:
                st.metric("Total Epochs", len(auth_system.training_history))

if __name__ == "__main__":
    main()
