# Mode-specific configurations matching project requirements
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
import os
import sys
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
        
        # Print font configuration summary
        if self.font_paths:
            print(f"\nCAPTCHA Generator initialized with {len(self.font_paths)} fonts:")
            print(f"  - First few fonts: {', '.join([os.path.basename(f) for f in self.font_paths[:3]])}")
            if len(self.font_paths) > 3:
                print(f"  - ...and {len(self.font_paths) - 3} more")
        else:
            print("CAPTCHA Generator initialized with default font only (no custom fonts available)")
        
        # Background colors from config, args, or defaults
        config_bg_colors = self.config.get('background_colors', [])
        if config_bg_colors:
            # Convert from config format to tuples if needed
            self.background_colors = [tuple(color) if isinstance(color, list) else color 
                                     for color in config_bg_colors]
        else:
            # self.background_colors = background_colors or [
            #     (255, 255, 255),  # White
            #     (240, 240, 240),  # Light gray
            #     (255, 248, 220),  # Cornsilk
            #     (245, 245, 220),  # Beige
            #     (230, 230, 250),  # Lavender
            # ]
            raise ValueError("No background colors detected")

    def _get_default_fonts(self) -> List[str]:
        """Get available DejaVu fonts which are known to be reliable on this system."""
        
        # Keep only DejaVu fonts which are reliable on the system
        possible_fonts = [
            # Linux DejaVu fonts (reliable on Linux systems)
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-BoldOblique.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansCondensed-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansCondensed-Oblique.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono-BoldOblique.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono-Oblique.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansCondensed.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansCondensed-BoldOblique.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-ExtraLight.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-Regular.otf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-Oblique.otf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-Bold.otf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-BoldOblique.otf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-Light.ttf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-LightItalic.ttf",
            "/usr/share/fonts/abattis-cantarell/Cantarell-Medium.ttf",
        ]
        available_fonts = [font for font in possible_fonts if os.path.exists(font)]
        
        # If no DejaVu fonts found, we'll just use PIL's default font later
        if not available_fonts:
            print("Warning: No DejaVu fonts found. Will use PIL's default font.")
        else:
            print(f"Found {len(available_fonts)} DejaVu fonts:")
            for font in available_fonts:
                print(f"  - {os.path.basename(font)}")
        
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
        # Get configuration from YAML - no default values, everything must be in the config.yaml
        captcha_config = self.config.get('captcha_config', {})
        
        # Get mode-specific config if applicable
        mode = config.get('mode') if config else None
        if mode and mode in self.config.get('mode_configs', {}):
            mode_specific_config = self.config.get('mode_configs', {}).get(mode, {})
            # Mode-specific config overrides base config
            mode_config_copy = captcha_config.copy()
            mode_config_copy.update(mode_specific_config)
            captcha_config = mode_config_copy
        
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
        if captcha_config.get('complex_background') and captcha_config.get('mode') in ['part3', 'part4']:
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
        
        # Print a summary of font usage for this CAPTCHA
        # print(f"\nGenerated CAPTCHA with text: '{text}' using {len(bboxes)} characters")
        
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
        
        # Initialize font tracking if not already done
        if not hasattr(self, '_last_used_fonts'):
            self._last_used_fonts = []
            self._synthetic_variations = []
        
        # Determine font type based on mode
        if config.get('challenging_fonts') and config.get('mode') == 'part4':
            font_type = "decorative"
        else:
            font_type = "standard"
        
        # Font sizes from config
        base_min, base_max = config['font_size_range']
        font_size = random.randint(base_min, base_max)
        
        # Font selection - prefer DejaVu fonts if available
        font = None
        font_path = None
        
        if self.font_paths:
            # Simple rotation through available fonts
            font_path = random.choice(self.font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
                # print(f"Using {font_type} font: {os.path.basename(font_path)} (size: {font_size})")
                
                # Track usage
                if len(self.font_paths) > 1:
                    self._last_used_fonts.append(font_path)
                    # Keep only the last 5 fonts in history
                    self._last_used_fonts = self._last_used_fonts[-5:]
            except Exception as e:
                print(f"Error loading font {os.path.basename(font_path) if font_path else 'unknown'}: {e}")
                font_path = None  # Reset if font loading failed
        
        # If no font loaded, use PIL's default with synthetic variations for diversity
        if not font:
            font = ImageFont.load_default()
            
            # Create a synthetic variation for variety
            variation_id = random.randint(0, 999)
            variation_type = random.choice(["normal", "bold", "italic", "condensed"])
            variation_key = f"{font_type}_{font_size}_{variation_type}_{variation_id}"
            
            if variation_key not in self._synthetic_variations:
                self._synthetic_variations.append(variation_key)
                if len(self._synthetic_variations) > 15:  # Keep more variations
                    self._synthetic_variations.pop(0)
            
            # We'll simulate different font styles through rendering options later
            # print(f"Using PIL default font (size: {font_size}, style: {variation_type}, id: {variation_id})")
        
        # Basic character positioning
        x_base = char_spacing * (char_index + 1)
        y_base = self.height // 2
        
        # Small offset for readability
        x_offset = random.randint(-10, 10)
        y_offset = random.randint(-10, 10)
        x = x_base + x_offset
        y = y_base + y_offset
        
        # Determine if this is a default font that needs extra variety
        is_default_font = font_path is None
        
        # Set variation type for default fonts (will be used for color selection)
        variation_type = "normal"
        if is_default_font:
            variation_type = random.choice(["normal", "bold", "italic", "condensed"])
            
        # Rotation based on config, with extra randomness for default fonts
        if config.get('mode') == 'part4' or config.get('mode') == 'part3':
            # Use large_rotation_range from config
            large_rotation_range = config.get('large_rotation_range')
            if large_rotation_range:
                min_rot, max_rot = large_rotation_range
                # Add more randomness for default fonts to create variety
                if is_default_font:
                    min_rot = min_rot * 1.2
                    max_rot = max_rot * 1.2
                rotation = random.uniform(min_rot, max_rot)
            else:
                rotation = 0 if not is_default_font else random.uniform(-30, 30)
        else:
            # Normal mode uses rotation_range from config
            rotation_range = config.get('rotation_range')
            if rotation_range:
                min_rot, max_rot = rotation_range
                # Add more randomness for default fonts to create variety
                if is_default_font:
                    min_rot = min_rot * 1.2
                    max_rot = max_rot * 1.2
                rotation = random.uniform(min_rot, max_rot)
            else:
                rotation = 0 if not is_default_font else random.uniform(-20, 20)
        
        # Character colors based on config and font type
        if config.get('color_variation'):
            # For default fonts, use more distinctive colors and styles
            if is_default_font:
                if variation_type == "bold":
                    color_style = "high_contrast"
                elif variation_type == "italic":
                    color_style = "colored"
                else:
                    color_style = random.choice(['dark', 'colored', 'high_contrast'])
            else:
                color_style = random.choice(['dark', 'colored', 'high_contrast'])
                
            # Apply the selected color style
            if color_style == 'dark':
                color = (
                    random.randint(0, 60),
                    random.randint(0, 60),
                    random.randint(0, 60)
                )
            elif color_style == 'colored':
                # More vibrant colors for default fonts
                if is_default_font:
                    base_color = random.choice([
                        (random.randint(50, 150), 0, 0),  # Red tones
                        (0, random.randint(50, 150), 0),  # Green tones
                        (0, 0, random.randint(50, 150)),  # Blue tones
                        (random.randint(50, 120), random.randint(50, 120), 0),  # Yellow/brown
                        (random.randint(50, 120), 0, random.randint(50, 120)),  # Purple
                        (0, random.randint(50, 120), random.randint(50, 120)),  # Cyan
                    ])
                else:
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
            color = (0, 0, 0)  # Default to black if color variation disabled
        
        # Get character dimensions for distortion calculations
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        
        # Character overlap based strictly on config
        if config.get('character_overlap') and char_index > 0:
            # Use overlap_amount from config 
            overlap_amount = config.get('overlap_amount')
            if overlap_amount:
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
        shear_range = config.get('shear_range')
        if shear_range:
            min_shear, max_shear = shear_range
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
        # Line distractors from config
        line_count = config.get('line_distractors', 0)  # Default to 0 for safety
        line_width = config.get('line_width')
        
        for _ in range(line_count):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            # Use a consistent color instead of random
            color = (100, 100, 100)
            # Use line_width if specified, otherwise default to 1
            width = line_width if line_width is not None else 1
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        
        return img
    
    def _apply_part4_degradations(self, img: Image.Image, draw: ImageDraw.Draw, 
                                 config: Dict, text: str) -> Image.Image:
        """Apply Part 4 specific degradations."""
        # Circular distractors from config
        circle_count = config.get('circular_distractors', 0)  # Default to 0 for safety
        circle_width = config.get('circle_width')
        circle_radius = config.get('circle_radius')
        
        for _ in range(circle_count):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            radius = circle_radius if circle_radius is not None else 15
            # Use a consistent color instead of random
            color = (100, 100, 100)
            width = circle_width if circle_width is not None else 1
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        outline=color, width=width)
        
        # Non-ASCII character distractors from config
        non_ascii_count = config.get('non_ascii_distractors', 0)  # Default to 0 for safety
        non_ascii_font_size = config.get('non_ascii_font_size')
        
        for _ in range(non_ascii_count):
            distractor_char = random.choice(self.non_ascii_distractors)
            x = random.randint(0, self.width - 50)
            y = random.randint(0, self.height - 50)
            # Use a consistent color instead of random
            color = (100, 100, 100)
            # Use font size from config or default to reasonable size
            font_size = non_ascii_font_size if non_ascii_font_size is not None else 30
            
            try:
                if self.font_paths:
                    font_path = self.font_paths[0]
                    font = ImageFont.truetype(font_path, font_size)
                    # print(f"Using font for non-ASCII distractor: {os.path.basename(font_path)} (size: {font_size})")
                else:
                    font = ImageFont.load_default()
                    # print(f"Using default font for non-ASCII distractor (size: {font_size})")
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
        
        # Mode configs must be defined in the config.yaml file
        if not mode_configs:
            print("Warning: No mode configurations found in the config file. Please define them in config.yaml.")
            mode_configs = {}
        
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
        
        # print(f"\nDataset generation complete!")
        # print(f"Mode: {mode}")
        # print(f"Samples: {num_samples}")
        # print(f"Resolution: {self.width}x{self.height}")
        # print(f"Average CAPTCHA length: {summary['avg_length']:.1f}")
        # print(f"Files saved to: {output_dir}")
        
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
                    # if i < 3:  
                        # print(f"Copied: {old_path} -> {new_path}")
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
        
        # print(f"\nExported to original format:")
        # print(f"- Directory: {part}/{mode}/")
        # print(f"- Images: {images_dir}")
        # print(f"- Labels: {os.path.join(mode_dir, 'labels.json')}")
        # print(f"- Total samples: {len(dataset_metadata)}")

# Example usage and configuration presets
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CAPTCHA Generator with YAML config support')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--part', type=str, nargs='+', default=['part2'], 
                      help='Part number(s) to generate. Multiple parts can be specified, e.g. --part part2 part3 part4')
    parser.add_argument('--train_samples', type=int, help='Override number of training samples from config')
    parser.add_argument('--val_samples', type=int, help='Override number of validation samples from config')
    parser.add_argument('--test_samples', type=int, help='Override number of test samples from config')
    parser.add_argument('--output_dir', type=str, help='Override base output directory from config')
    args = parser.parse_args()

    # Initialize generator with config file
    generator = CAPTCHAGenerator(config_path=args.config)

    # Get configuration from YAML file
    dataset_config = generator.config.get('dataset_generation', {})
    
    # Set values with priority: command line args > yaml config > defaults
    train_samples = args.train_samples if args.train_samples is not None else dataset_config.get('train_samples', 100)
    val_samples = args.val_samples if args.val_samples is not None else dataset_config.get('val_samples', 20)
    test_samples = args.test_samples if args.test_samples is not None else dataset_config.get('test_samples', 20)
    base_dir = args.output_dir if args.output_dir is not None else dataset_config.get('output_dir', 'output')
    
    # Validate parts
    valid_parts = ['part2', 'part3', 'part4']
    parts_to_generate = [part for part in args.part if part in valid_parts]
    if not parts_to_generate:
        print(f"Error: No valid parts specified. Valid parts are: {', '.join(valid_parts)}")
        sys.exit(1)
    
    # Process each part
    print(f"Using configuration from: {args.config}")
    print(f"Samples: train={train_samples}, val={val_samples}, test={test_samples}")
    print(f"Generating datasets for: {', '.join(parts_to_generate)}")
    
    for part in parts_to_generate:
        print(f"\n{'='*50}")
        print(f"Generating {part} dataset...")
        print(f"{'='*50}")
        
        # Generate training set
        if train_samples > 0:
            print(f"Generating {part} training set ({train_samples} samples)...")
            train_data = generator.generate_dataset(
                num_samples=train_samples,
                mode=part,
                output_dir=os.path.join(base_dir, f'temp_{part}_train')
            )
            generator.export_to_original_format(
                train_data,
                output_dir=base_dir,
                part=part,
                mode='train'
            )
        
        # Generate validation set
        if val_samples > 0:
            print(f"Generating {part} validation set ({val_samples} samples)...")
            val_data = generator.generate_dataset(
                num_samples=val_samples,
                mode=part,
                output_dir=os.path.join(base_dir, f'temp_{part}_val')
            )
            generator.export_to_original_format(
                val_data,
                output_dir=base_dir,
                part=part,
                mode='val'
            )
        
        # Generate test set
        if test_samples > 0:
            print(f"Generating {part} test set ({test_samples} samples)...")
            test_data = generator.generate_dataset(
                num_samples=test_samples,
                mode=part,
                output_dir=os.path.join(base_dir, f'temp_{part}_test')
            )
            generator.export_to_original_format(
                test_data,
                output_dir=base_dir,
                part=part,
                mode='test'
            )
            
        # Clean up temporary directories for this part
        import shutil
        
        temp_dirs = [
            os.path.join(base_dir, f'temp_{part}_train'),
            os.path.join(base_dir, f'temp_{part}_val'),
            os.path.join(base_dir, f'temp_{part}_test')
        ]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                # print(f"Cleaned up temporary directory: {temp_dir}")
        
        print(f"Dataset generation for {part} complete! Files saved to: {base_dir}/{part}/")
    
    print(f"\nAll datasets generated successfully!")
