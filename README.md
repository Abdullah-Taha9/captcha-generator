# CAPTCHA Generator with YAML Configuration

This project provides a flexible CAPTCHA generator that can be configured using a YAML file, allowing for easy customization of CAPTCHA generation parameters. It follows the exact directory structure and format specified in the project requirements.

## Features

- Generate CAPTCHAs with varying levels of difficulty
- Support for multiple parts: part2 (standard), part3 (harder), and part4 (hardest)
- Structured output following the required directory format
- JSON annotations in the required format for training and validation
- Customizable via YAML configuration file
- Extensive control over CAPTCHA appearance and distortions
- Command-line interface for easy usage

## Requirements

- Python 3.6+
- Required Python packages:
  - numpy
  - Pillow (PIL)
  - PyYAML

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd captcha-generator
   ```

2. Install required packages:
   ```
   pip install numpy Pillow PyYAML
   ```

## Usage

### Basic Usage

```
python captcha_generator.py --config config.yaml --part part2
```

### Command-line Arguments

- `--config`: Path to the YAML configuration file (default: 'config.yaml')
- `--part`: Which part to generate ('part2', 'part3', or 'part4') (default: 'part2')
- `--train_samples`: Number of training samples to generate (default: 100)
- `--val_samples`: Number of validation samples to generate (default: 20)
- `--test_samples`: Number of test samples to generate (default: 20)
- `--output_dir`: Base output directory (default: 'output')

### Using as a Library

```python
from captcha_generator import CAPTCHAGenerator, load_config

# Initialize with a config file
generator = CAPTCHAGenerator(config_path='config.yaml')

# Generate a single CAPTCHA
img, text, bboxes = generator.generate_captcha()

# Generate a dataset
dataset = generator.generate_dataset(num_samples=50, mode='part2')

# Export to the required directory structure
generator.export_to_original_format(
    dataset_metadata=dataset,
    output_dir='output',
    part='part2',
    mode='train'
)
```

## Configuration File

The `config.yaml` file allows you to configure all aspects of CAPTCHA generation:

### Basic Settings

```yaml
width: 640                          # CAPTCHA image width
height: 160                         # CAPTCHA image height
charset: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Characters to use
captcha_length_range: [3, 7]        # Min and max length
```

### Font and Background Configuration

```yaml
font_paths:                         # Paths to font files
  - "/path/to/font1.ttf"
  - "/path/to/font2.ttf"

background_colors:                  # RGB color tuples
  - [255, 255, 255]                 # White
  - [240, 240, 240]                 # Light gray
```

### Mode-specific Configurations

The config file includes separate sections for part2, part3, and part4 modes:

```yaml
mode_configs:
  part2:
    mode: 'part2'
    rotation_range: [-15, 15]
    font_size_range: [40, 60]
    color_variation: true
    challenging_fonts: false
    complex_background: false
    
  part3:
    mode: 'part3'
    large_rotation_range: [-55, 55]
    line_distractors: 2
    noise_level: 0.12
    complex_background: true
    
  part4:
    mode: 'part4'
    line_distractors: 3
    circular_distractors: 1
    non_ascii_distractors: 2
    blur_level: 1.2
    character_overlap: true
```

## Output Directory Structure

The generator creates the following directory structure:

```
output/
├── part2/                      # Root directory for part2
│   ├── train/                  # Training data split
│   │   ├── images/             # Images folder
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── labels.json         # Labels file in required format
│   ├── val/                    # Validation data split
│   │   ├── images/
│   │   │   └── ...
│   │   └── labels.json
│   └── test/                   # Test data split
│       └── images/
│           └── ...
├── part3/
│   └── ...
└── part4/
    └── ...
```

## Example Configuration

See the included `config.yaml` file for a comprehensive example of all available configuration options.
