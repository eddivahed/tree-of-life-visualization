# Tree of Life Visualization

An interactive visualization system that combines deep learning and dimensionality reduction to create an explorable space of natural images with real-time generative art capabilities.

## 🌟 Features

- **Dual-View Interface**
  - 3D visualization of image relationships
  - Real-time GAN-generated patterns
  - Interactive navigation system

- **Advanced ML Pipeline**
  - UMAP dimensionality reduction
  - GAN-based pattern generation
  - Feature extraction system

- **Interactive Controls**
  - Intuitive camera navigation
  - Real-time pattern updates
  - Smooth transitions

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Poetry for dependency management
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tree-of-life-visualization.git
cd tree-of-life-visualization
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Running the Application

1. Download and process images:
```bash
python -m tree_of_life.data.downloader
```

2. Train the GAN model:
```bash
python -m tree_of_life.gan_navigator.trainer
```

3. Launch the visualization:
```bash
python -m tree_of_life.visualization.main
```

## 🎮 Usage

### Navigation Controls
- **Arrow Keys**: Move camera
- **Mouse Wheel**: Zoom in/out
- **Mouse Drag**: Pan view
- **R**: Reset camera
- **+/-**: Adjust image size

### GAN View
- Real-time pattern generation
- Position-based visualization
- Smooth transitions between states

## 🏗️ Project Structure

```
tree-of-life-project/
├── pyproject.toml      
├── src/
│   └── tree_of_life/
│       ├── data/           # Data handling
│       ├── clustering/     # ML processing
│       ├── gan_navigator/  # GAN systems
│       └── visualization/  # Display
├── data/
│   ├── images/        # Image storage
│   ├── models/        # Trained models
│   └── csv/           # Metadata
└── notebooks/         # Development
```

## 🛠️ Technical Details

### Core Components

1. **Data Pipeline**
   - Image collection from Open Images Dataset
   - Feature extraction (color, texture)
   - Data preprocessing and normalization

2. **Clustering System**
   - UMAP dimensionality reduction
   - Feature normalization
   - Cluster relationship mapping

3. **GAN Integration**
   - Position-based latent space navigation
   - Real-time pattern generation
   - Smooth state transitions

4. **Visualization**
   - Split-view interface
   - Interactive controls
   - Real-time updates

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Development Guide](docs/development.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Open Images Dataset for the training data
- UMAP algorithm by Leland McInnes
- All contributors and supporters

## 🎥 Related Video

Check out the project explanation video on my YouTube channel: [Fellowship of eddie](https://www.youtube.com/your-channel)
