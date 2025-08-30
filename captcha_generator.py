# Mode-specific configurations matching project requirements
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
import os
from typing import List, Tuple, Dict, Optional, Any
import json
import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}

class CAPTCHAGenerator:
    """
    A comprehensive CAPTCHA generator that can create both normal and degraded CAPTCHA images
    for training computer vision models. Supports various degradation types for robustness testing.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 width: int = 640, 
                 height: int = 160,
                 font_paths: Optional[List[str]] = None,
                 background_colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        Initialize the CAPTCHA generator.
        
        Args:
            config_path: Path to the YAML configuration file
            width: Image width in pixels (overridden by config if provided)
            height: Image height in pixels (overridden by config if provided)
            font_paths: List of paths to font files (TTF/OTF) (overridden by config if provided)
            background_colors: List of RGB tuples for background colors (overridden by config if provided)
        """
        # Load configuration from YAML if provided
        self.config = {}
        if config_path:
            self.config = load_config(config_path)
        
        # Set basic parameters with priority: config > explicit args > defaults
        self.width = self.config.get('width', width)
        self.height = self.config.get('height', height)
        
        # Character set from config or default
        charset_config = self.config.get('charset', "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.charset = charset_config
        
        # Non-ASCII distractors from config or default
        distractors_config = self.config.get('non_ascii_distractors', 
                                            "αβγδεζηθικλμνξοπρστυφχψω№§¶†‡•…‰′″‹›€™")
        self.non_ascii_distractors = distractors_config
        
        # Font paths from config, args, or defaults
        config_fonts = self.config.get('font_paths', [])
        self.font_paths = config_fonts if config_fonts else (font_paths or self._get_default_fonts())
        
        # Background colors from config, args, or defaults
        config_bg_colors = self.config.get('background_colors', [])
        if config_bg_colors:
            # Convert from config format to tuples if needed
            self.background_colors = [tuple(color) if isinstance(color, list) else color 
                                     for color in config_bg_colors]
        else:
            self.background_colors = background_colors or [
                (255, 255, 255),  # White
                (240, 240, 240),  # Light gray
                (255, 248, 220),  # Cornsilk
                (245, 245, 220),  # Beige
                (230, 230, 250),  # Lavender
            ]
        
    def _get_default_fonts(self) -> List[str]:
        """Get diverse system fonts including decorative ones. Modify paths based on your system."""
        possible_fonts = [
            # Standard fonts
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/Times.ttc",  # macOS
            "/System/Library/Fonts/Courier.ttc",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "C:/Windows/Fonts/calibri.ttf",  # Windows
            "C:/Windows/Fonts/times.ttf",  # Windows
            "C:/Windows/Fonts/cour.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
            
            # Decorative and stylized fonts
            "/System/Library/Fonts/Impact.ttf",  # macOS
            "/System/Library/Fonts/Chalkduster.ttc",  # macOS
            "/System/Library/Fonts/Marker Felt.ttc",  # macOS
            "/System/Library/Fonts/Papyrus.ttc",  # macOS
            "/System/Library/Fonts/Brush Script.ttf",  # macOS
            "C:/Windows/Fonts/impact.ttf",  # Windows
            "C:/Windows/Fonts/comic.ttf",  # Windows Comic Sans
            "C:/Windows/Fonts/BRADHITC.TTF",  # Windows Bradley Hand
            "C:/Windows/Fonts/BRUSHSCI.TTF",  # Windows Brush Script
            "C:/Windows/Fonts/FORTE.TTF",  # Windows Forte
            "C:/Windows/Fonts/SNAP__.TTF",  # Windows Snap ITC
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux Bold
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",  # Linux Serif
            
            # Additional common decorative fonts (if available)
            "C:/Windows/Fonts/STENCIL.TTF",
            "C:/Windows/Fonts/SHOWG.TTF",
            "C:/Windows/Fonts/JOKERMAN.TTF",
            "C:/Windows/Fonts/CURLZ___.TTF",
            "C:/Windows/Fonts/RAVIE.TTF",
        ]
        available_fonts = [font for font in possible_fonts if os.path.exists(font)]
        
        # If no system fonts found, we'll use PIL's default and create synthetic variations
        if not available_fonts:
            print("Warning: No system fonts found. Using default font with variations.")
        
        return available_fonts
    
    def generate_captcha(self, 
                        text: Optional[str] = None,
                        captcha_length: Optional[int] = None,
                        config: Optional[Dict] = None) -> Tuple[Image.Image, str, List[Dict]]:
        """
        Generate a CAPTCHA image with specified configuration.
        
        Args:
            text: Specific text to generate (if None, random text is generated)
            captcha_length: Length of CAPTCHA (if None, random between 3-6)
            config: Configuration dictionary for degradations (overrides both defaults and config file)
            
        Returns:
            Tuple of (PIL Image, ground truth text, bounding boxes list)
        """
        # Default configuration
        default_config = {
            'mode': 'normal',  # 'normal', 'part3', 'part4'
            'rotation_range': (-15, 15),
            'shear_range': (-0.2, 0.2),
            'font_size_range': (40, 60),
            'color_variation': True,
            'background_texture': False,
            
            # Part 3 degradations
            'large_rotation_range': (-45, 45),
            'line_distractors': 0,
            'noise_level': 0.0,
            'complex_background': False,
            
            # Part 4 degradations
            'circular_distractors': 0,
            'non_ascii_distractors': 0,
            'challenging_fonts': False,
            'blur_level': 0.0,
            'character_overlap': False,
        }
        
        # Apply configurations with priority: explicit config param > yaml config > defaults
        captcha_config = default_config.copy()
        
        # Get config from YAML if it exists
        yaml_captcha_config = self.config.get('captcha_config', {})
        if yaml_captcha_config:
            captcha_config.update(yaml_captcha_config)
        
        # Finally apply the explicit config parameter if provided
        if config:
            captcha_config.update(config)
        
        # Generate text if not provided
        if text is None:
            if captcha_length is None:
                # Check if there's a captcha length range in the config
                length_range = self.config.get('captcha_length_range', (3, 7))
                captcha_length = random.randint(length_range[0], length_range[1])
            text = ''.join(random.choices(self.charset, k=captcha_length))
        
        # Create base image
        if captcha_config['complex_background'] and captcha_config['mode'] in ['part3', 'part4']:
            img = self._create_complex_background()
        else:
            bg_color = random.choice(self.background_colors)
            img = Image.new('RGB', (self.width, self.height), bg_color)
        
        draw = ImageDraw.Draw(img)
        bboxes = []
        
        # Calculate character spacing
        char_spacing = self.width // (len(text) + 1)
        
        # Draw each character
        for i, char in enumerate(text):
            bbox = self._draw_character(
                img, draw, char, i, char_spacing, captcha_config, len(text)
            )
            bboxes.append(bbox)
        
        # Apply degradations based on mode
        if captcha_config['mode'] == 'part3':
            img = self._apply_part3_degradations(img, draw, captcha_config)
        elif captcha_config['mode'] == 'part4':
            img = self._apply_part4_degradations(img, draw, captcha_config, text)
        
        # Apply noise if specified
        if captcha_config['noise_level'] > 0:
            img = self._add_noise(img, captcha_config['noise_level'])
            
        # Apply blur if specified
        if captcha_config['blur_level'] > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=captcha_config['blur_level']))
        
        return img, text, bboxes
    
    def _create_complex_background(self) -> Image.Image:
        """Create a more complex and varied background with multiple effects."""
        img = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(img)
        
        # Choose background style randomly
        bg_style = random.choice(['gradient', 'noise_pattern', 'geometric', 'textured'])
        
        if bg_style == 'gradient':
            # Multi-directional gradient
            direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
            
            if direction == 'radial':
                # Radial gradient from center
                center_x, center_y = self.width // 2, self.height // 2
                max_radius = math.sqrt(center_x**2 + center_y**2)
                
                for y in range(self.height):
                    for x in range(self.width):
                        distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                        ratio = distance / max_radius
                        
                        r = int(220 + 35 * math.sin(ratio * math.pi))
                        g = int(230 + 25 * math.cos(ratio * math.pi * 1.5))
                        b = int(240 + 15 * math.sin(ratio * math.pi * 2))
                        
                        color = (max(180, min(255, r)), max(180, min(255, g)), max(180, min(255, b)))
                        draw.point((x, y), color)
                        
                        # Add some random noise
                        if random.random() < 0.02:
                            noise_color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in color)
                            draw.point((x, y), noise_color)
            
            elif direction == 'diagonal':
                for y in range(self.height):
                    for x in range(self.width):
                        ratio = (x + y) / (self.width + self.height)
                        r = int(200 + 55 * math.sin(ratio * math.pi * 2))
                        g = int(210 + 45 * math.cos(ratio * math.pi * 3))
                        b = int(220 + 35 * math.sin(ratio * math.pi * 1.5))
                        color = (max(180, min(255, r)), max(180, min(255, g)), max(180, min(255, b)))
                        draw.point((x, y), color)
            
            else:  # horizontal or vertical
                for y in range(self.height):
                    ratio = y / self.height if direction == 'vertical' else 0.5
                    for x in range(self.width):
                        if direction == 'horizontal':
                            ratio = x / self.width
                        
                        r = int(210 + 45 * math.sin(ratio * math.pi * 2 + random.uniform(-0.1, 0.1)))
                        g = int(220 + 35 * math.cos(ratio * math.pi * 3 + random.uniform(-0.1, 0.1)))
                        b = int(230 + 25 * math.sin(ratio * math.pi * 1.8 + random.uniform(-0.1, 0.1)))
                        color = (max(180, min(255, r)), max(180, min(255, g)), max(180, min(255, b)))
                        draw.point((x, y), color)
        
        elif bg_style == 'noise_pattern':
            # Start with base color
            base_color = random.choice([(240, 240, 240), (235, 245, 235), (245, 235, 235), (235, 235, 245)])
            img = Image.new('RGB', (self.width, self.height), base_color)
            
            # Add noise pattern
            for _ in range(self.width * self.height // 10):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                noise_intensity = random.randint(-30, 30)
                pixel_color = tuple(max(0, min(255, c + noise_intensity)) for c in base_color)
                draw.point((x, y), pixel_color)
        
        elif bg_style == 'geometric':
            # Geometric pattern background
            base_colors = [(230, 230, 230), (225, 235, 225), (235, 225, 225), (225, 225, 235)]
            img = Image.new('RGB', (self.width, self.height), random.choice(base_colors))
            draw = ImageDraw.Draw(img)
            
            # Add geometric shapes
            for _ in range(random.randint(15, 30)):
                shape_type = random.choice(['rectangle', 'ellipse', 'line'])
                x1 = random.randint(0, self.width)
                y1 = random.randint(0, self.height)
                x2 = x1 + random.randint(20, 100)
                y2 = y1 + random.randint(20, 60)
                
                alpha = random.randint(10, 40)
                color = tuple(random.randint(200, 250) for _ in range(3))
                
                # Create semi-transparent overlay
                overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                if shape_type == 'rectangle':
                    overlay_draw.rectangle([x1, y1, x2, y2], fill=(*color, alpha))
                elif shape_type == 'ellipse':
                    overlay_draw.ellipse([x1, y1, x2, y2], fill=(*color, alpha))
                else:  # line
                    overlay_draw.line([x1, y1, x2, y2], fill=(*color, alpha), width=random.randint(1, 4))
                
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        else:  # textured
            # Create textured background with multiple layers
            base_color = random.choice([(220, 220, 220), (215, 225, 215), (225, 215, 215)])
            img = Image.new('RGB', (self.width, self.height), base_color)
            
            # Add texture with various patterns
            for layer in range(3):
                overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                pattern_type = random.choice(['dots', 'lines', 'waves'])
                
                if pattern_type == 'dots':
                    for _ in range(random.randint(50, 150)):
                        x = random.randint(0, self.width)
                        y = random.randint(0, self.height)
                        radius = random.randint(1, 5)
                        color = tuple(random.randint(180, 240) for _ in range(3))
                        alpha = random.randint(20, 60)
                        overlay_draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                                           fill=(*color, alpha))
                
                elif pattern_type == 'lines':
                    for _ in range(random.randint(20, 50)):
                        x1 = random.randint(0, self.width)
                        y1 = random.randint(0, self.height)
                        x2 = x1 + random.randint(-50, 50)
                        y2 = y1 + random.randint(-30, 30)
                        color = tuple(random.randint(180, 240) for _ in range(3))
                        alpha = random.randint(20, 50)
                        overlay_draw.line([x1, y1, x2, y2], fill=(*color, alpha), 
                                        width=random.randint(1, 3))
                
                else:  # waves
                    wave_amplitude = random.randint(10, 30)
                    wave_frequency = random.uniform(0.01, 0.03)
                    for y in range(0, self.height, 5):
                        points = []
                        for x in range(0, self.width, 10):
                            wave_y = y + wave_amplitude * math.sin(x * wave_frequency)
                            points.append((x, wave_y))
                        
                        if len(points) > 1:
                            color = tuple(random.randint(180, 240) for _ in range(3))
                            alpha = random.randint(15, 40)
                            for i in range(len(points) - 1):
                                overlay_draw.line([points[i], points[i+1]], 
                                                fill=(*color, alpha), width=2)
                
                img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        return img
    
    def _draw_character(self, img: Image.Image, draw: ImageDraw.Draw, 
                       char: str, char_index: int, char_spacing: int, 
                       config: Dict, total_chars: int) -> Dict:
        """Draw a single character with enhanced distortions and decorative fonts."""
        
        # Font selection strictly based on config
        if config.get('challenging_fonts') and config['mode'] == 'part4':
            # Use more decorative/challenging fonts for Part 4
            font_path = random.choice(self.font_paths) if self.font_paths else None
        else:
            font_path = random.choice(self.font_paths) if self.font_paths else None
        
        # Font sizes strictly from config without additional variation
        base_min, base_max = config['font_size_range']
        font_size = random.randint(base_min, base_max)
        
        # Create font with potential synthetic effects
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Basic character positioning
        x_base = char_spacing * (char_index + 1)
        y_base = self.height // 2
        
        # Small offset for readability
        x_offset = random.randint(-10, 10)
        y_offset = random.randint(-10, 10)
        x = x_base + x_offset
        y = y_base + y_offset
        
        # Rotation strictly based on config
        if config['mode'] == 'part4' or config['mode'] == 'part3':
            # Use large_rotation_range from config
            min_rot, max_rot = config.get('large_rotation_range', (-45, 45))
            rotation = random.uniform(min_rot, max_rot)
        else:
            # Normal mode uses rotation_range from config
            min_rot, max_rot = config.get('rotation_range', (-15, 15))
            rotation = random.uniform(min_rot, max_rot)
        
        # Enhanced character colors with more variety
        if config['color_variation']:
            color_style = random.choice(['dark', 'colored', 'high_contrast'])
            if color_style == 'dark':
                color = (
                    random.randint(0, 60),
                    random.randint(0, 60),
                    random.randint(0, 60)
                )
            elif color_style == 'colored':
                # More vibrant colors
                base_color = random.choice([
                    (random.randint(100, 200), 0, 0),  # Red tones
                    (0, random.randint(100, 200), 0),  # Green tones
                    (0, 0, random.randint(100, 200)),  # Blue tones
                    (random.randint(80, 150), random.randint(80, 150), 0),  # Yellow/brown
                    (random.randint(80, 150), 0, random.randint(80, 150)),  # Purple
                ])
                color = base_color
            else:  # high_contrast
                color = (0, 0, 0)  # Pure black
        else:
            color = (0, 0, 0)
        
        # Get character dimensions for distortion calculations
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        
        # Character overlap based strictly on config
        if config.get('character_overlap') and char_index > 0:
            # Fixed 30% overlap or use overlap_amount from config if provided
            overlap_amount = config.get('overlap_amount', 0.3)
            x -= int(char_width * overlap_amount)
        
        # Store original position for bounding box
        original_x = x
        original_y = y
        
        # Enhanced distortions: shear, scale, and perspective
        temp_size = max(char_width, char_height) + 60
        char_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
        char_draw = ImageDraw.Draw(char_img)
        
        # Draw character in center
        temp_x = (temp_size - char_width) // 2
        temp_y = (temp_size - char_height) // 2
        char_draw.text((temp_x, temp_y), char, font=font, fill=(*color, 255))
        
        # Apply distortions based on strict config parameters
        
        # 1. Rotation (always applied)
        if abs(rotation) > 0.5:
            char_img = char_img.rotate(rotation, expand=True)
        
        # 2. Shear distortion (based on config)
        min_shear, max_shear = config.get('shear_range', (-0.2, 0.2))
        shear_x = random.uniform(min_shear, max_shear)
        shear_y = random.uniform(min_shear, max_shear)
        char_img = self._apply_shear(char_img, shear_x, shear_y)
        
        # 3. Scale distortion (if enabled in config)
        if config.get('scale_distortion', False):
            scale_x = random.uniform(0.8, 1.2)
            scale_y = random.uniform(0.8, 1.2)
            new_width = int(char_img.width * scale_x)
            new_height = int(char_img.height * scale_y)
            char_img = char_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 4. Perspective distortion (only for part3/part4 if enabled in config)
        if config['mode'] in ['part3', 'part4'] and config.get('perspective_distortion', False):
            char_img = self._apply_perspective_distortion(char_img)
        
        # 5. Outline/stroke effect (if enabled in config)
        if config.get('character_outline', False):
            char_img = self._add_character_outline(char_img, color)
        
        # Paste the distorted character
        paste_x = max(0, min(self.width - char_img.width, x - char_img.width // 2))
        paste_y = max(0, min(self.height - char_img.height, y - char_img.height // 2))
        
        try:
            img.paste(char_img, (paste_x, paste_y), char_img)
        except:
            # Fallback to simple text drawing if paste fails
            draw.text((original_x, original_y), char, font=font, fill=color)
        
        # Calculate bounding box in original dataset format
        bbox_dict = {
            'character': char,
            'x_center': original_x / self.width,
            'y_center': original_y / self.height,
            'width': char_width / self.width,
            'height': char_height / self.height,
            'rotation': rotation
        }
        
        return bbox_dict
    
    def _apply_shear(self, img: Image.Image, shear_x: float, shear_y: float) -> Image.Image:
        """Apply shear transformation to image."""
        try:
            # Shear transformation matrix
            transform_matrix = (
                1, shear_x, 0,
                shear_y, 1, 0
            )
            return img.transform(img.size, Image.AFFINE, transform_matrix, 
                               resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
        except:
            return img
    
    def _apply_perspective_distortion(self, img: Image.Image) -> Image.Image:
        """Apply subtle perspective distortion to character."""
        try:
            width, height = img.size
            # Define perspective transformation
            distortion = random.uniform(0.05, 0.15)
            
            # Random perspective direction
            perspective_type = random.choice(['left', 'right', 'top', 'bottom'])
            
            if perspective_type == 'left':
                transform = [
                    0, int(height * distortion),
                    width, 0,
                    width, height,
                    0, int(height * (1 - distortion))
                ]
            elif perspective_type == 'right':
                transform = [
                    0, 0,
                    width, int(height * distortion),
                    width, int(height * (1 - distortion)),
                    0, height
                ]
            elif perspective_type == 'top':
                transform = [
                    int(width * distortion), 0,
                    int(width * (1 - distortion)), 0,
                    width, height,
                    0, height
                ]
            else:  # bottom
                transform = [
                    0, 0,
                    width, 0,
                    int(width * (1 - distortion)), height,
                    int(width * distortion), height
                ]
            
            return img.transform(img.size, Image.QUAD, transform, 
                               resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
        except:
            return img
    
    def _add_character_outline(self, img: Image.Image, base_color: tuple) -> Image.Image:
        """Add outline effect to character."""
        try:
            # Create outline by drawing the same text with offset in a different color
            outline_color = tuple(min(255, c + 100) for c in base_color)  # Lighter outline
            
            # Create new image with outline
            outlined = Image.new('RGBA', img.size, (0, 0, 0, 0))
            
            # Draw outline by pasting original image with small offsets
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # Create offset version
                    offset_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    try:
                        offset_img.paste(img, (dx, dy))
                        outlined = Image.alpha_composite(outlined, offset_img)
                    except:
                        pass
            
            # Paste original on top
            outlined = Image.alpha_composite(outlined, img)
            return outlined
        except:
            return img
    
    def _apply_part3_degradations(self, img: Image.Image, draw: ImageDraw.Draw, config: Dict) -> Image.Image:
        """Apply Part 3 specific degradations."""
        # Line distractors - use exact number from config
        line_count = config.get('line_distractors', 0)
        line_width = config.get('line_width', 2)
        
        for _ in range(line_count):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            # Use a consistent color instead of random
            color = (100, 100, 100)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        
        return img
    
    def _apply_part4_degradations(self, img: Image.Image, draw: ImageDraw.Draw, 
                                 config: Dict, text: str) -> Image.Image:
        """Apply Part 4 specific degradations."""
        # Circular distractors - use exact number from config
        circle_count = config.get('circular_distractors', 0)
        circle_width = config.get('circle_width', 2)
        
        for _ in range(circle_count):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            radius = config.get('circle_radius', 20)
            # Use a consistent color instead of random
            color = (100, 100, 100)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        outline=color, width=circle_width)
        
        # Non-ASCII character distractors - use exact number from config
        non_ascii_count = config.get('non_ascii_distractors', 0)
        non_ascii_font_size = config.get('non_ascii_font_size', 40)
        
        for _ in range(non_ascii_count):
            distractor_char = random.choice(self.non_ascii_distractors)
            x = random.randint(0, self.width - 50)
            y = random.randint(0, self.height - 50)
            # Use a consistent color instead of random
            color = (100, 100, 100)
            
            try:
                if self.font_paths:
                    font = ImageFont.truetype(self.font_paths[0], non_ascii_font_size)
                else:
                    font = ImageFont.load_default()
                draw.text((x, y), distractor_char, font=font, fill=color)
            except:
                pass  # Skip if font doesn't support the character
        
        return img
    
    def _add_noise(self, img: Image.Image, noise_level: float) -> Image.Image:
        """Add random noise to the image."""
        img_array = np.array(img)
        noise = np.random.randint(-int(255 * noise_level), 
                                 int(255 * noise_level), 
                                 img_array.shape, dtype=np.int16)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    def generate_dataset(self, 
                        num_samples: int = None,
                        mode: str = None,
                        output_dir: str = None,
                        custom_config: Optional[Dict] = None,
                        save_annotations: bool = None) -> List[Dict]:
        """
        Generate a dataset of CAPTCHA images matching original dataset format.
        
        Args:
            num_samples: Number of samples to generate (overrides config)
            mode: Generation mode ('normal', 'part3', 'part4') (overrides config)
            output_dir: Directory to save images (overrides config)
            custom_config: Custom configuration overrides (highest priority)
            save_annotations: Whether to save bounding box annotations (overrides config)
            
        Returns:
            List of dictionaries containing metadata for each generated sample
        """
        # Get configuration from YAML or use defaults
        dataset_config = self.config.get('dataset_generation', {})
        
        # Use parameters with priority: explicit args > yaml config > defaults
        num_samples = num_samples if num_samples is not None else dataset_config.get('num_samples', 100)
        mode = mode if mode is not None else dataset_config.get('mode', 'normal')
        output_dir = output_dir if output_dir is not None else dataset_config.get('output_dir', 'generated_captchas')
        save_annotations = save_annotations if save_annotations is not None else dataset_config.get('save_annotations', True)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhanced mode-specific configurations with more challenging distortions
        mode_configs = self.config.get('mode_configs', {})
        
        # If no mode configs in YAML, use these defaults
        if not mode_configs:
            mode_configs = {
                'normal': {
                    'mode': 'normal',
                    'rotation_range': (-25, 25),  # Increased rotation for normal mode
                    'font_size_range': (40, 70),  # Wider font size range
                    'color_variation': True,
                    'challenging_fonts': True,
                    'complex_background': False,
                },
                'part3': {
                    'mode': 'part3',
                    'large_rotation_range': (-55, 55),  # More extreme rotations
                    'line_distractors': 2,  # Will be randomized per sample
                    'noise_level': 0.12,  # Will be randomized per sample
                    'complex_background': True,
                    'font_size_range': (35, 75),  # Wider range for more diversity
                    'challenging_fonts': True,  # Use decorative fonts even in part3
                    'color_variation': True,
                },
                'part4': {
                    'mode': 'part4',
                    'large_rotation_range': (-65, 65),  # Even more extreme rotations
                    'line_distractors': 3,  # Will be randomized per sample
                    'circular_distractors': 1,  # Will be randomized per sample
                    'non_ascii_distractors': 2,  # Will be randomized per sample
                    'challenging_fonts': True,
                    'blur_level': 1.2,  # Will be randomized per sample
                    'character_overlap': True,  # Will be randomized per sample
                    'noise_level': 0.18,  # Will be randomized per sample
                    'complex_background': True,
                    'font_size_range': (30, 80),  # Widest range for maximum challenge
                    'color_variation': True,
                }
        }
        
        config = mode_configs.get(mode, {})
        if custom_config:
            config.update(custom_config)
        
        dataset_metadata = []
        
        for i in range(num_samples):
            # Generate sample-specific config variations for maximum diversity
            sample_config = config.copy()
            
            # No per-sample randomization - use exact config values
            
            # Get captcha length range from config or use sensible defaults if not specified
            length_range = self.config.get('captcha_length_range', [3, 7])
            captcha_length = random.randint(length_range[0], length_range[1])
            img, text, bboxes = self.generate_captcha(
                captcha_length=captcha_length,
                config=sample_config
            )
            
            # Save image in PNG format (lossless)
            filename = f"{mode}_{i:06d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath, 'PNG')
            
            # Create metadata matching original dataset format
            metadata = {
                'filename': filename,
                'filepath': filepath,  # Store the absolute filepath
                'text': text,  # Ground truth CAPTCHA string
                'length': len(text),  # CAPTCHA length
                'resolution': f"{self.width}x{self.height}",
                'mode': mode,
            }
            
            # Add bounding boxes if annotations should be saved
            if save_annotations:
                # Format bounding boxes as (X-center, Y-center, Width, Height)
                formatted_bboxes = []
                for bbox in bboxes:
                    formatted_bboxes.append({
                        'character': bbox['character'],
                        'x_center': bbox['x_center'],
                        'y_center': bbox['y_center'],
                        'width': bbox['width'],
                        'height': bbox['height'],
                    })
                metadata['bboxes'] = formatted_bboxes
            
            dataset_metadata.append(metadata)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples for {mode} mode")
        
        # Save metadata in JSON format
        metadata_file = os.path.join(output_dir, f"{mode}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Save summary statistics
        summary = {
            'total_samples': num_samples,
            'mode': mode,
            'resolution': f"{self.width}x{self.height}",
            'character_set': self.charset,
            'captcha_lengths': list(set(len(item['text']) for item in dataset_metadata)),
            'avg_length': sum(len(item['text']) for item in dataset_metadata) / num_samples,
        }
        
        summary_file = os.path.join(output_dir, f"{mode}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Mode: {mode}")
        print(f"Samples: {num_samples}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Average CAPTCHA length: {summary['avg_length']:.1f}")
        print(f"Files saved to: {output_dir}")
        
        return dataset_metadata
    
    def export_to_original_format(self, dataset_metadata: List[Dict], 
                                 output_dir: str, part: str = 'part2', mode: str = 'train') -> None:
        """
        Export generated dataset to match the original dataset format as described in documentation.
        Creates directory structure and JSON files according to the expected format.
        
        Args:
            dataset_metadata: List of metadata from generate_dataset
            output_dir: Base directory to save formatted data
            part: Part number ('part2', 'part3', 'part4')
            mode: Dataset split name ('train', 'val', 'test')
        """
        # Create directory structure like part2/train/images/
        part_dir = os.path.join(output_dir, part)
        mode_dir = os.path.join(part_dir, mode)
        images_dir = os.path.join(mode_dir, 'images')
        
        os.makedirs(images_dir, exist_ok=True)
        
        # Prepare labels data in the expected format
        labels_data = []
        
        for i, item in enumerate(dataset_metadata):
            # Generate a 6-digit image ID with leading zeros
            image_id = f"{i+1:06d}"
            
            # Get the source image path from the metadata
            if 'filepath' in item and os.path.exists(item['filepath']):
                # Use the absolute filepath if available
                old_path = item['filepath']
            else:
                # Fallback to constructed path (but this likely won't work)
                old_path = os.path.join(output_dir, item['filename'])
                
            # Set destination path
            new_filename = f"{image_id}.png"
            new_path = os.path.join(images_dir, new_filename)
            
            # Copy the file
            try:
                import shutil
                if os.path.exists(old_path):
                    shutil.copy2(old_path, new_path)
                    # Print success for the first few files to confirm it's working
                    if i < 3:  
                        print(f"Copied: {old_path} -> {new_path}")
                else:
                    print(f"Warning: Source image not found at {old_path}")
            except Exception as e:
                print(f"Error copying image: {e} (from {old_path} to {new_path})")
            
            # Create entry for this image
            entry = {
                'height': self.height,
                'width': self.width,
                'image_id': image_id,
                'captcha_string': item['text'],
                'annotations': []
            }
            
            # Process bounding boxes if available (train and val only)
            if 'bboxes' in item and mode in ['train', 'val']:
                for bbox in item['bboxes']:
                    char = bbox['character']
                    
                    # Convert normalized coordinates to absolute pixel values
                    x_center = bbox['x_center'] * self.width
                    y_center = bbox['y_center'] * self.height
                    width = bbox['width'] * self.width
                    height = bbox['height'] * self.height
                    
                    # Calculate the absolute bbox coordinates [x1, y1, x2, y2]
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)
                    
                    # Convert character to category_id (0-9 for digits, 10-35 for A-Z)
                    category_id = None
                    if char.isdigit():
                        category_id = int(char)
                    elif char.isalpha():
                        category_id = ord(char.upper()) - ord('A') + 10
                    
                    if category_id is not None:
                        # Create a simple oriented_bbox (just a rotated rectangle)
                        # In a real implementation, this would need proper rotation calculation
                        # For simplicity, we're generating a basic approximation
                        rotation = bbox.get('rotation', 0)
                        cx, cy = x_center, y_center
                        w, h = width, height
                        
                        # Create oriented bbox points based on rotation
                        import math
                        angle_rad = math.radians(rotation)
                        cos_val = math.cos(angle_rad)
                        sin_val = math.sin(angle_rad)
                        
                        # Calculate the four corner points of the rotated rectangle
                        half_w = w / 2
                        half_h = h / 2
                        
                        # Order: top-left, top-right, bottom-right, bottom-left
                        points = [
                            (-half_w, -half_h),  # top-left
                            (half_w, -half_h),   # top-right
                            (half_w, half_h),    # bottom-right
                            (-half_w, half_h),   # bottom-left
                        ]
                        
                        oriented_bbox = []
                        for px, py in points:
                            # Rotate point
                            rx = px * cos_val - py * sin_val
                            ry = px * sin_val + py * cos_val
                            # Translate to center
                            rx += cx
                            ry += cy
                            oriented_bbox.extend([rx, ry])
                        
                        # Create annotation
                        annotation = {
                            'bbox': [x1, y1, x2, y2],
                            'oriented_bbox': oriented_bbox,
                            'category_id': category_id
                        }
                        entry['annotations'].append(annotation)
            
            labels_data.append(entry)
        
        # Save labels file for all splits (including test)
        labels_file = os.path.join(mode_dir, 'labels.json')
        with open(labels_file, 'w') as f:
            json.dump(labels_data, f, indent=2)
        
        print(f"\nExported to original format:")
        print(f"- Directory: {part}/{mode}/")
        print(f"- Images: {images_dir}")
        print(f"- Labels: {os.path.join(mode_dir, 'labels.json')}")
        print(f"- Total samples: {len(dataset_metadata)}")

# Example usage and configuration presets
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Generator with YAML config support')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--part', type=str, default='part2', choices=['part2', 'part3', 'part4'], help='Part number')
    parser.add_argument('--train_samples', type=int, help='Override number of training samples from config')
    parser.add_argument('--val_samples', type=int, help='Override number of validation samples from config')
    parser.add_argument('--test_samples', type=int, help='Override number of test samples from config')
    parser.add_argument('--output_dir', type=str, help='Override base output directory from config')
    args = parser.parse_args()

    # Initialize generator with config file
    generator = CAPTCHAGenerator(config_path=args.config)

    # Set mode based on part
    mode = args.part
    
    # Get configuration from YAML file
    dataset_config = generator.config.get('dataset_generation', {})
    
    # Set values with priority: command line args > yaml config > defaults
    train_samples = args.train_samples if args.train_samples is not None else dataset_config.get('train_samples', 100)
    val_samples = args.val_samples if args.val_samples is not None else dataset_config.get('val_samples', 20)
    test_samples = args.test_samples if args.test_samples is not None else dataset_config.get('test_samples', 20)
    base_dir = args.output_dir if args.output_dir is not None else dataset_config.get('output_dir', 'output')
    
    # Create all three splits
    print(f"Generating {mode} dataset...")
    print(f"Using configuration from: {args.config}")
    print(f"Samples: train={train_samples}, val={val_samples}, test={test_samples}")
    
    # Generate training set
    if train_samples > 0:
        print(f"Generating training set ({train_samples} samples)...")
        train_data = generator.generate_dataset(
            num_samples=train_samples,
            mode=mode,
            output_dir=os.path.join(base_dir, 'temp_train')
        )
        generator.export_to_original_format(
            train_data,
            output_dir=base_dir,
            part=mode,
            mode='train'
        )
    
    # Generate validation set
    if val_samples > 0:
        print(f"Generating validation set ({val_samples} samples)...")
        val_data = generator.generate_dataset(
            num_samples=val_samples,
            mode=mode,
            output_dir=os.path.join(base_dir, 'temp_val')
        )
        generator.export_to_original_format(
            val_data,
            output_dir=base_dir,
            part=mode,
            mode='val'
        )
    
    # Generate test set
    if test_samples > 0:
        print(f"Generating test set ({test_samples} samples)...")
        test_data = generator.generate_dataset(
            num_samples=test_samples,
            mode=mode,
            output_dir=os.path.join(base_dir, 'temp_test')
        )
        generator.export_to_original_format(
            test_data,
            output_dir=base_dir,
            part=mode,
            mode='test'
        )
        
    # Clean up temporary directories
    import shutil
    
    temp_dirs = [
        os.path.join(base_dir, 'temp_train'),
        os.path.join(base_dir, 'temp_val'),
        os.path.join(base_dir, 'temp_test')
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    
    print(f"\nDataset generation complete! Files saved to: {base_dir}/{mode}/")