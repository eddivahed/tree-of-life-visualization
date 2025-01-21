# src/tree_of_life/gan_navigator/latent_explorer.py

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=2):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Simpler architecture without batch normalization
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class NavigationSystem:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.latent_dim = 100
        
        # Initialize models
        self.generator = Generator(latent_dim=self.latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            
        self.movement_history = []
        self.generated_cache = {}

    def train(self, embeddings, n_epochs=200, batch_size=64, save_path=None):
        """Train the GAN on embeddings data"""
        self.logger.info("Starting GAN training...")
        
        # Ensure minimum batch size
        batch_size = max(32, min(batch_size, len(embeddings)))
        
        # Normalize embeddings to [-1, 1] range
        embeddings_min = embeddings.min(axis=0)
        embeddings_max = embeddings.max(axis=0)
        embeddings_normalized = 2 * (embeddings - embeddings_min) / (embeddings_max - embeddings_min) - 1
        
        # Convert embeddings to torch dataset
        embeddings_tensor = torch.FloatTensor(embeddings_normalized).to(self.device)
        dataloader = DataLoader(
            TensorDataset(embeddings_tensor), 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True  # Prevent single-sample batches
        )
        
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        
        # Optimizers with adjusted learning rates
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        
        for epoch in range(n_epochs):
            total_g_loss = 0
            total_d_loss = 0
            batches = 0
            
            for i, (real_points,) in enumerate(dataloader):
                batch_size = real_points.size(0)
                
                # Ground truths
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                gen_points = self.generator(z)
                g_loss = adversarial_loss(self.discriminator(gen_points), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(self.discriminator(real_points), valid)
                fake_loss = adversarial_loss(self.discriminator(gen_points.detach()), fake)
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
        """Save generator model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, path)
        
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load saved model"""
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

    def move_to(self, position):
        """Generate new position based on input position"""
        # Convert position to latent vector
        z = self._position_to_latent(position)
        
        # Generate new position
        with torch.no_grad():
            generated = self.generator(z)
            new_pos = generated.cpu().numpy()[0]
        
        # Cache the result
        self.generated_cache[position] = new_pos
        self.movement_history.append(position)
        
        return new_pos

    def _position_to_latent(self, position):
        """Convert 2D/3D position to latent vector"""
        # Normalize position
        pos = np.array(position[:2]) / np.max(np.abs(position)) if np.any(position) else np.zeros(2)
        
        # Create base latent vector
        z = np.random.normal(0, 1, (1, self.latent_dim))
        
        # Modify first 2 dimensions based on position
        z[0, :2] = pos
        
        return torch.FloatTensor(z).to(self.device)

    def get_transition(self, start_pos, end_pos, steps=10):
        """Generate smooth transition between positions"""
        images = []
        for t in np.linspace(0, 1, steps):
            current_pos = tuple(a + (b - a) * t for a, b in zip(start_pos, end_pos))
            images.append(self.move_to(current_pos))
        return images