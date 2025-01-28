# src/tree_of_life/gan_navigator/latent_explorer.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=64):  # Changed output dimension
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            # Initial dense layer
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64):  # Match Generator output
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, pattern):
        return self.model(pattern)


class NavigationSystem:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.latent_dim = 100
        self.output_dim = 64
        
        # Initialize models
        self.generator = Generator(latent_dim=self.latent_dim, output_dim=self.output_dim).to(device)
        self.discriminator = Discriminator(input_dim=self.output_dim).to(device)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            
        self.movement_history = []
        self.generated_cache = {}
        self.last_position = None

    def train(self, embeddings, n_epochs=200, batch_size=64, save_path=None):
        """Train GAN on embeddings data"""
        self.logger.info("Starting GAN training...")

        # Ensure minimum batch size
        batch_size = max(32, min(batch_size, len(embeddings)))

        # Preprocess embeddings
        embeddings_min = embeddings.min(axis=0)
        embeddings_max = embeddings.max(axis=0)
        embeddings_normalized = (
            2 * (embeddings - embeddings_min) / (embeddings_max - embeddings_min) - 1
        )

        # Reshape embeddings to match discriminator input size
        if embeddings_normalized.shape[1] != self.output_dim:
            # Pad or truncate to match required dimension
            if embeddings_normalized.shape[1] < self.output_dim:
                pad_width = self.output_dim - embeddings_normalized.shape[1]
                embeddings_normalized = np.pad(
                    embeddings_normalized, ((0, 0), (0, pad_width)), mode="constant"
                )
            else:
                embeddings_normalized = embeddings_normalized[:, : self.output_dim]

        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(embeddings_normalized).to(self.device)
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Optimizers
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Loss function
        adversarial_loss = nn.BCELoss()

        for epoch in range(n_epochs):
            total_g_loss = 0
            total_d_loss = 0
            batches = 0

            for i, (real_patterns,) in enumerate(dataloader):
                batch_size = real_patterns.size(0)
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                gen_patterns = self.generator(z)
                g_loss = adversarial_loss(self.discriminator(gen_patterns), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(self.discriminator(real_patterns), valid)
                fake_loss = adversarial_loss(
                    self.discriminator(gen_patterns.detach()), fake
                )
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                batches += 1

            if (epoch + 1) % 20 == 0:
                self.logger.info(
                    f"[Epoch {epoch+1}/{n_epochs}] "
                    f"[D loss: {total_d_loss/batches:.4f}] "
                    f"[G loss: {total_g_loss/batches:.4f}]"
                )

        if save_path:
            self.save_model(save_path)

        self.logger.info("Training completed!")

    def save_model(self, path):
        """Save generator and discriminator models"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, path)
        
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load saved models"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.generator.eval()
            self.discriminator.eval()
            self.logger.info("Successfully loaded model")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _position_to_latent(self, position):
        """Convert 3D position to latent vector"""
        # Normalize position to [-1, 1] range
        pos = np.array(position[:2])  # Take only x, y coordinates
        if np.any(pos):
            pos = pos / (np.abs(pos).max() + 1e-8)  # Normalize with small epsilon
        
        # Create base latent vector with normal distribution
        z = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Use position to influence first dimensions
        z[0, 0] = pos[0] * 2  # Scale to use full range
        z[0, 1] = pos[1] * 2
        
        # Add some consistency based on last position
        if self.last_position is not None:
            z[0, 2:4] = z[0, :2] * 0.5  # Smooth transition
        
        self.last_position = pos
        return torch.FloatTensor(z).to(self.device)
    
    def move_to(self, position):
        """Generate new pattern based on position"""
        # Convert position to latent vector
        z = self._position_to_latent(position)
        
        # Generate pattern
        with torch.no_grad():
            pattern = self.generator(z)
            pattern = pattern.cpu().numpy()[0]
            
            # Reshape pattern to 8x8
            pattern = pattern.reshape(8, 8)
            
            # Add variations based on position
            x, y = position[:2]
            pattern = pattern * (1 + 0.2 * np.sin(5 * x)) + 0.2 * np.cos(5 * y)
            
            # Ensure pattern is properly normalized
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
        
        # Cache the result
        self.generated_cache[position] = pattern
        self.movement_history.append(position)
        
        return pattern

    def get_transition(self, start_pos, end_pos, steps=10):
        """Generate smooth transition between positions"""
        patterns = []
        for t in np.linspace(0, 1, steps):
            current_pos = tuple(a + (b - a) * t for a, b in zip(start_pos, end_pos))
            patterns.append(self.move_to(current_pos))
        return patterns
