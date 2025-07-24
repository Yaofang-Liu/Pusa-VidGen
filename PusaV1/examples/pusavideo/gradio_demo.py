import gradio as gr
import torch
import os
import sys
import datetime
import shutil
from PIL import Image
import cv2
import numpy as np
from diffsynth import ModelManager, PusaMultiFramesPipeline, PusaV2VPipeline, WanVideoPusaPipeline, save_video
import tempfile

class PusaVideoDemo:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_manager = None
        self.multi_frames_pipe = None
        self.v2v_pipe = None
        self.t2v_pipe = None
        self.base_dir = "model_zoo/PusaV1/Wan2.1-T2V-14B"
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_models(self):
        """Load all models once for efficiency"""
        if self.model_manager is None:
            print("Loading models...")
            self.model_manager = ModelManager(device="cpu")
            
            model_files = sorted([os.path.join(self.base_dir, f) for f in os.listdir(self.base_dir) if f.endswith('.safetensors')])
            
            self.model_manager.load_models(
                [
                    model_files,
                    os.path.join(self.base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                    os.path.join(self.base_dir, "Wan2.1_VAE.pth"),
                ],
                torch_dtype=torch.bfloat16,
            )
            print("Models loaded successfully!")
        
    def load_lora_and_get_pipe(self, pipe_type, lora_path, lora_alpha):
        """Load LoRA and return appropriate pipeline"""

        if self.model_manager is None:
            self.load_models()
            self.model_manager.load_lora(lora_path, lora_alpha=lora_alpha)
        
        if pipe_type == "multi_frames":
            pipe = PusaMultiFramesPipeline.from_model_manager(self.model_manager, torch_dtype=torch.bfloat16, device=self.device)
            pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
        elif pipe_type == "v2v":
            pipe = PusaV2VPipeline.from_model_manager(self.model_manager, torch_dtype=torch.bfloat16, device=self.device)
            pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9)
        elif pipe_type == "t2v":
            pipe = WanVideoPusaPipeline.from_model_manager(self.model_manager, torch_dtype=torch.bfloat16, device=self.device)
            pipe.enable_vram_management(num_persistent_param_in_dit=None)
        
        return pipe

    def process_video_frames(self, video_path):
        """Process video frames for V2V pipeline"""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get original video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate scaling and cropping parameters
        target_width = 1280
        target_height = 720
        target_ratio = target_width / target_height
        original_ratio = width / height
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize maintaining aspect ratio
            if original_ratio > target_ratio:
                # Video is wider than target
                new_width = int(height * target_ratio)
                # Crop width from center
                start_x = (width - new_width) // 2
                frame = frame[:, start_x:start_x + new_width]
            else:
                # Video is taller than target
                new_height = int(width / target_ratio)
                # Crop height from center
                start_y = (height - new_height) // 2
                frame = frame[start_y:start_y + new_height]
            
            # Resize to target dimensions
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames

    def generate_i2v_video(self, image_path, prompt, noise_multiplier, 
                          lora_alpha, num_inference_steps, negative_prompt, progress=gr.Progress()):
        """Generate video from single image (I2V)"""
        try:
            progress(0.1, desc="Loading models...")
            lora_path = "./model_zoo/PusaV1/pusa_v1.pt"
            pipe = self.load_lora_and_get_pipe("multi_frames", lora_path, lora_alpha)
            
            progress(0.2, desc="Processing input image...")
            
            # Process single image for I2V
            if image_path is None:
                raise ValueError("No image provided")
            
            # Handle image path - Gradio with type="filepath" returns the path directly
            img = Image.open(image_path)
            processed_image = img.convert("RGB").resize((1280, 720), Image.LANCZOS)
            
            # I2V always uses position 0 (first frame)
            multi_frame_images = {0: (processed_image, float(noise_multiplier))}
            
            progress(0.4, desc="Generating video...")
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                multi_frame_images=multi_frame_images,
                num_inference_steps=num_inference_steps,
                height=720, width=1280, num_frames=81,
                seed=0, tiled=True
            )
            
            progress(0.9, desc="Saving video...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(self.output_dir, f"i2v_output_{timestamp}_noise_{noise_multiplier}_alpha_{lora_alpha}.mp4")
            save_video(video, video_filename, fps=25, quality=5)
            
            progress(1.0, desc="Complete!")
            return video_filename, f"Video generated successfully! Saved to {video_filename}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    def generate_multi_frames_video(self, image1, image2, image3, num_imgs, prompt, cond_position, noise_multipliers, 
                                   lora_alpha, num_inference_steps, negative_prompt, progress=gr.Progress()):
        """Generate video from multiple frames (Start-End, Multi-frame)"""
        try:
            progress(0.1, desc="Loading models...")
            lora_path = "./model_zoo/PusaV1/pusa_v1.pt"
            pipe = self.load_lora_and_get_pipe("multi_frames", lora_path, lora_alpha)
            
            progress(0.2, desc="Processing input images...")
            
            # Parse conditioning positions and noise multipliers
            cond_pos_list = [int(x.strip()) for x in cond_position.split(',')]
            noise_mult_list = [float(x.strip()) for x in noise_multipliers.split(',')]
            
            # Collect images based on num_imgs
            image_paths = [image1, image2]
            if num_imgs == "3" and image3 is not None:
                image_paths.append(image3)
            
            # Filter out None values
            image_paths = [path for path in image_paths if path is not None]
            
            if len(image_paths) != len(cond_pos_list) or len(image_paths) != len(noise_mult_list):
                raise ValueError("The number of images, conditioning positions, and noise multipliers must be the same.")
            
            # Process images
            processed_images = []
            for img_path in image_paths:
                img = Image.open(img_path)
                processed_images.append(img.convert("RGB").resize((1280, 720), Image.LANCZOS))
            
            multi_frame_images = {
                cond_pos: (img, noise_mult) 
                for cond_pos, img, noise_mult in zip(cond_pos_list, processed_images, noise_mult_list)
            }
            
            progress(0.4, desc="Generating video...")
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                multi_frame_images=multi_frame_images,
                num_inference_steps=num_inference_steps,
                height=720, width=1280, num_frames=81,
                seed=0, tiled=True
            )
            
            progress(0.9, desc="Saving video...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(self.output_dir, f"multi_frame_output_{timestamp}.mp4")
            save_video(video, video_filename, fps=25, quality=5)
            
            progress(1.0, desc="Complete!")
            return video_filename, f"Video generated successfully! Saved to {video_filename}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    def generate_v2v_video(self, video_path, prompt, cond_position, noise_multipliers,
                          lora_alpha, num_inference_steps, negative_prompt, progress=gr.Progress()):
        """Generate video from video (V2V completion, extension)"""
        try:
            progress(0.1, desc="Loading models...")
            lora_path = "./model_zoo/PusaV1/pusa_v1.pt"
            pipe = self.load_lora_and_get_pipe("v2v", lora_path, lora_alpha)
            
            progress(0.2, desc="Processing input video...")
            
            # Parse conditioning positions and noise multipliers
            cond_pos_list = [int(x.strip()) for x in cond_position.split(',')]
            noise_mult_list = [float(x.strip()) for x in noise_multipliers.split(',')]
            
            # Process video
            conditioning_video = self.process_video_frames(video_path)
            
            progress(0.4, desc="Generating video...")
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                conditioning_video=conditioning_video,
                conditioning_indices=cond_pos_list,
                conditioning_noise_multipliers=noise_mult_list,
                num_inference_steps=num_inference_steps,
                height=720, width=1280, num_frames=81,
                seed=0, tiled=True
            )
            
            progress(0.9, desc="Saving video...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.basename(video_path).split('.')[0]
            video_filename = os.path.join(self.output_dir, f"v2v_{output_filename}_{timestamp}.mp4")
            save_video(video, video_filename, fps=25, quality=5)
            
            progress(1.0, desc="Complete!")
            return video_filename, f"Video generated successfully! Saved to {video_filename}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    def generate_t2v_video(self, prompt, lora_alpha, num_inference_steps, 
                          negative_prompt, progress=gr.Progress()):
        """Generate video from text prompt"""
        try:
            progress(0.1, desc="Loading models...")
            lora_path = "./model_zoo/PusaV1/pusa_v1.pt"
            pipe = self.load_lora_and_get_pipe("t2v", lora_path, lora_alpha)
            
            progress(0.3, desc="Generating video...")
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=720, width=1280, num_frames=81,
                seed=0, tiled=True
            )
            
            progress(0.9, desc="Saving video...")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = os.path.join(self.output_dir, f"t2v_output_{timestamp}.mp4")
            save_video(video, video_filename, fps=25, quality=5)
            
            progress(1.0, desc="Complete!")
            return video_filename, f"Video generated successfully! Saved to {video_filename}"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

def create_demo():
    demo_instance = PusaVideoDemo()
    
    # Set custom cache directory to avoid permission issues
    import tempfile
    import os
    try:
        # Try to use a custom cache directory in the current workspace
        cache_dir = os.path.join(os.getcwd(), "gradio_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = cache_dir
    except:
        pass  # Fall back to default if this fails
    
    # Helper function to safely load demo files
    def safe_file_path(file_path):
        """Return file path if it exists, None otherwise"""
        try:
            if os.path.exists(file_path):
                return file_path
        except:
            pass
        return None
    
    # Custom CSS for fancy black design
    css = """
    /* === Main Theme: "Cosmic Flow" === */
    :root {
        --color-primary: #22d3ee; /* Cosmic Cyan */
        --color-secondary: #ec4899; /* Galactic Pink */
        --color-accent: #a78bfa; /* Astral Violet */
        --color-background-dark: #0f172a; /* Midnight Slate */
        --color-background-light: #1e293b; /* Twilight Slate */
        --color-surface: rgba(30, 41, 59, 0.6); /* Glassy Slate */
        --color-surface-hover: rgba(30, 41, 59, 0.9);
        --color-text-light: #f1f5f9; /* Starlight White */
        --color-text-medium: #94a3b8; /* Nebula Gray */
        --color-text-dark: #64748b; /* Meteor Gray */
        --font-main: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        --radius-lg: 20px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    /* === Global Styles === */
    .gradio-container {
        font-family: var(--font-main) !important;
        background: linear-gradient(135deg, var(--color-background-dark) 0%, var(--color-background-light) 100%) !important;
        color: var(--color-text-light) !important;
    }
    
    * {
        color: var(--color-text-light);
        border-color: rgba(148, 163, 184, 0.1); /* slate-400/10% */
    }

    /* === Glassmorphism Containers === */
    .gr-panel, .gr-box, .gr-group, .gr-column, .gr-tabitem, .gr-accordion {
        background: var(--color-surface) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }

    .gr-panel:hover, .gr-box:hover, .gr-group:hover, .gr-column:hover {
        background: var(--color-surface-hover) !important;
        border-color: rgba(148, 163, 184, 0.2) !important;
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3) !important;
    }

    /* === Header (Static Nebula) === */
    .fancy-header {
        text-align: center !important;
        background-color: var(--color-background-dark) !important;
        padding: 40px !important;
        border-radius: var(--radius-lg) !important;
        margin-bottom: 40px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.5) !important;
    }
    .fancy-header::before {
        content: '' !important;
        position: absolute !important;
        top: -150px; left: -150px; right: -150px; bottom: -150px;
        background: 
            radial-gradient(ellipse at 20% 25%, var(--color-primary), transparent 40%),
            radial-gradient(ellipse at 80% 30%, var(--color-accent), transparent 40%),
            radial-gradient(ellipse at 50% 90%, var(--color-secondary), transparent 45%) !important;
        opacity: 0.2 !important;
        filter: blur(80px) !important;
        transform: scale(1.2) !important;
        z-index: 0 !important;
    }
    .fancy-header > * { 
        position: relative !important; /* Ensures content is on top of the nebula effect */
        z-index: 1 !important;
    }

    /* === Tabs === */
    .gr-tabs { background: transparent !important; }
    .gr-tab-nav {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: var(--radius-lg) !important;
        padding: 6px !important;
        border: none !important;
    }
    .gr-tab-nav button {
        background: transparent !important;
        color: var(--color-text-medium) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        padding: 12px 20px !important;
        border: none !important;
    }
    .gr-tab-nav button:hover {
        background: rgba(167, 139, 250, 0.2) !important;
        color: var(--color-text-light) !important;
    }
    .gr-tab-nav button.selected {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%) !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(34, 211, 238, 0.3) !important;
    }
    
    /* === Primary Generate Button === */
    .generate-btn, .primary-btn, button.primary, .gr-button-primary {
        background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%) !important;
        background-size: 250% 250% !important;
        border: 2px solid transparent !important;
        border-radius: var(--radius-lg) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 18px 36px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        transition: all 0.4s ease !important;
        box-shadow: 0 10px 30px rgba(34, 211, 238, 0.2), 0 10px 30px rgba(236, 72, 153, 0.2) !important;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    .generate-btn::before, .primary-btn::before {
        content: '' !important;
        position: absolute !important;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s ease;
        z-index: -1;
    }
    .generate-btn:hover::before, .primary-btn:hover::before {
        left: 100%;
    }
    .generate-btn:hover, .primary-btn:hover {
        transform: translateY(-5px) scale(1.03) !important;
        box-shadow: 0 15px 40px rgba(34, 211, 238, 0.4), 0 15px 40px rgba(236, 72, 153, 0.4) !important;
        background-position: 100% 50% !important;
    }
    
    /* === Secondary & Tertiary Buttons (e.g., "Load Example") === */
    button:not(.primary):not(.selected) {
        background: rgba(148, 163, 184, 0.1) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: var(--color-text-medium) !important;
        border-radius: var(--radius-md) !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    button:not(.primary):not(.selected):hover {
        background: var(--color-accent) !important;
        border-color: var(--color-accent) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(167, 139, 250, 0.3) !important;
    }
    
    /* === Input Fields & Textareas === */
    input, textarea, .gr-textbox, .gr-number {
        background: rgba(15, 23, 42, 0.8) !important; /* Midnight Slate dark */
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: var(--radius-md) !important;
        color: var(--color-text-light) !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    input:focus, textarea:focus, .gr-textbox:focus-within, .gr-number:focus-within {
        border-color: var(--color-primary) !important;
        box-shadow: 0 0 15px rgba(34, 211, 238, 0.2) !important;
        outline: none !important;
    }
    input::placeholder, textarea::placeholder {
        color: var(--color-text-dark) !important;
    }
    
    /* === Sliders === */
    .gr-slider {
        --slider-track-color: rgba(15, 23, 42, 0.9);
        --slider-range-color: linear-gradient(90deg, var(--color-primary) 0%, var(--color-accent) 100%);
        --slider-handle-color: white;
        --slider-handle-shadow: 0 4px 15px rgba(34, 211, 238, 0.4);
    }
    .gradio-container .gr-slider .gr-slider-track { background: var(--slider-track-color) !important; }
    .gradio-container .gr-slider .gr-slider-range { background: var(--slider-range-color) !important; }
    .gradio-container .gr-slider .gr-slider-handle {
        background: var(--slider-handle-color) !important;
        border: 2px solid var(--color-primary) !important;
        box-shadow: var(--slider-handle-shadow) !important;
    }
    
    /* === File Upload === */
    .gr-file, .gr-upload {
        background: rgba(15, 23, 42, 0.7) !important;
        border: 2px dashed var(--color-text-dark) !important;
        border-radius: var(--radius-lg) !important;
        transition: all 0.3s ease !important;
    }
    .gr-file:hover, .gr-upload:hover {
        border-color: var(--color-primary) !important;
        background: rgba(34, 211, 238, 0.1) !important;
    }
    .gr-file *, .gr-upload * { color: var(--color-text-medium) !important; background: transparent !important; }
    
    /* === Markdown & Text === */
    .gr-markdown { color: var(--color-text-light) !important; }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        background: linear-gradient(90deg, var(--color-primary) 0%, var(--color-secondary) 100%);
        -webkit-background-clip: text;
        -moz-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .gr-markdown a {
        color: var(--color-primary) !important;
        text-decoration: none !important;
        transition: all 0.2s ease;
    }
    .gr-markdown a:hover {
        color: var(--color-secondary) !important;
        text-decoration: underline !important;
    }
    label {
        color: var(--color-text-medium) !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    .gr-info {
        color: var(--color-text-dark) !important;
        font-style: italic;
    }
    
    /* === Progress Bar === */
    .gr-progress {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: var(--radius-sm) !important;
    }
    .gr-progress-bar {
        background: linear-gradient(90deg, var(--color-primary) 0%, var(--color-accent) 100%) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* === Scrollbar === */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: var(--color-background-light); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--color-accent), var(--color-primary));
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(var(--color-primary), var(--color-secondary));
    }
    
    /* === Final cleanup & overrides === */
    .gradio-container .prose {
        color: var(--color-text-light) !important;
    }
    .gradio-container .gr-button * {
        color: inherit !important;
    }
    """
    
    with gr.Blocks(css=css, title="✨ Pusa V1.0 - Revolutionary AI Video Generation ✨", theme=gr.themes.Default(primary_hue="purple", neutral_hue="gray").set(
        body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        background_fill_primary="#1e293b",
        background_fill_secondary="#0f172a",
        border_color_primary="rgba(148, 163, 184, 0.1)"
    )) as demo:
        
        # Header
        gr.HTML("""
        <div class="fancy-header">
            <div style="position: relative; z-index: 1;">
                <h1 style="font-size: 3.5em; margin-bottom: 20px; text-shadow: 0 4px 15px rgba(0,0,0,0.4); background: none !important; color: white !important;">
                ✨ PUSA V1.0 ✨
            </h1>
            <h2 style="font-size: 1.4em; margin-bottom: 15px; opacity: 0.95; background: none !important; color: white !important;">
                🎬 Revolutionary Video Generation with Vectorized Timestep Adaptation
            </h2>
            <p style="font-size: 1.2em; margin-bottom: 10px; background: none !important; color: white !important;">
                🔥 <strong>BREAKTHROUGH PERFORMANCE:</strong> Surpassing Wan-I2V on Vbench-I2V with only $500 training cost! 🔥
            </p>
            <p style="font-size: 1.1em; opacity: 0.9; background: none !important; color: white !important;">
                🚀 <strong>4 Powerful Modes:</strong> I2V • Multi-Frame • V2V • T2V 🚀
            </p>
            <div style="margin-top: 20px; font-size: 0.9em; opacity: 0.8; background: none !important; color: white !important;">
                💎 State-of-the-Art • ⚡ Lightning Fast • 🎯 Precision Control • 🌟 Professional Quality
                </div>
            </div>
        </div>
        """)
        
        # Set default LoRA path (hidden from users)
        lora_path = "./model_zoo/PusaV1/pusa_v1.pt"
        
        # Tabs for different functionalities
        with gr.Tabs():
            
            # Tab 1: Image-to-Video (I2V)
            with gr.TabItem("🎨 Image-to-Video"):
                gr.Markdown("""
                ### Image-to-Video Generation (I2V)
                Generate videos from a single starting image. Perfect for bringing static images to life with natural motion and animation.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Input Image")
                        image_input = gr.Image(
                            label="Upload Single Image",
                            type="filepath",  # This returns the file path directly
                            height=300
                        )
                        
                        gr.Markdown("#### ⚙️ Generation Parameters")
                        with gr.Group():
                            noise_multiplier_i2v = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.2, step=0.1,
                                label="Noise Multiplier",
                                info="Controls how faithful the generation is to the input image (0=faithful, 1=creative)"
                            )
                            lora_alpha_i2v = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.4, step=0.1,
                                label="LoRA Alpha",
                                info="Controls temporal consistency (1-2 recommended)"
                            )
                            steps_i2v = gr.Slider(
                                minimum=10, maximum=50, value=10, step=5,
                                label="Inference Steps"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Text Prompts")
                        prompt_i2v = gr.Textbox(
                            lines=4,
                            label="Prompt",
                            placeholder="Describe the motion and animation you want to see in the video..."
                        )
                        negative_prompt_i2v = gr.Textbox(
                            lines=3,
                            value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                            label="Negative Prompt"
                        )
                        
                        generate_i2v_btn = gr.Button("🎬 Generate I2V Video", variant="primary", size="lg", elem_classes=["generate-btn", "primary-btn"])
                        
                        gr.Markdown("#### 📹 Output")
                        video_output_i2v = gr.Video(label="Generated Video")
                        status_i2v = gr.Textbox(label="Status", interactive=False)
                
                # Demo examples for I2V
                gr.Markdown("### 🎭 Demo Examples")
                with gr.Accordion("Example 1: Monk Meditation", open=False):
                    gr.Markdown("""
                    **Prompt:** "A wide-angle shot shows a serene monk meditating with gentle swaying and peaceful movement..."
                    - **Noise Multiplier:** 0.2
                    - **LoRA Alpha:** 1.4
                    """)
                    gr.Button("Load Example 1").click(
                        lambda: (0.2, 1.4, "A wide-angle shot shows a serene monk meditating perched atop a pile of weathered rocks that spell out 'ZEN'. The scene is bathed in warm sunrise light with gentle swaying movement."),
                        outputs=[noise_multiplier_i2v, lora_alpha_i2v, prompt_i2v]
                    )
                
                with gr.Accordion("Example 2: Space Adventure", open=False):
                    gr.Markdown("""
                    **Prompt:** "A female climber rock climbing on an asteroid in deep space with dynamic movement..."
                    - **Noise Multiplier:** 0.3
                    - **LoRA Alpha:** 1.2
                    """)
                    gr.Button("Load Example 2").click(
                        lambda: (0.3, 1.2, "A low-angle, long exposure shot of a lone female climber, wearing shorts and tank top rock climbing on a massive asteroid in deep space. The climber moves methodically with focused determination."),
                        outputs=[noise_multiplier_i2v, lora_alpha_i2v, prompt_i2v]
                    )
            
            # Tab 2: Multi-Frames to Video
            with gr.TabItem("🖼️ Multi-Frames to Video"):
                gr.Markdown("""
                ### Multi-Frames to Video Generation
                Generate videos using multiple conditioning frames for advanced control:
                - **Start-End Frames**: Create smooth transitions between two frames
                - **Multi-frame Conditioning**: Use multiple frames for complex scenarios
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📷 Input Images")
                        # Replace gr.Files with multiple gr.Image components for better display
                        with gr.Row():
                            image1_input = gr.Image(label="Image 1", type="filepath", height=200)
                            image2_input = gr.Image(label="Image 2", type="filepath", height=200)
                            image3_input = gr.Image(label="Image 3 (Optional)", type="filepath", height=200)

                        # Add a textbox to specify how many images are being used
                        num_images = gr.Dropdown(
                            choices=["2", "3"], 
                            value="2", 
                            label="Number of Images"
                        )
                        
                        gr.Markdown("#### 🎯 Conditioning Parameters")
                        with gr.Group():
                            cond_position_multi = gr.Textbox(
                                value="0,20",
                                label="Conditioning Positions",
                                info="Comma-separated frame indices (0-20). E.g., '0,20' for start-end, '0,10,20' for multi-frame"
                            )
                            noise_multipliers_multi = gr.Textbox(
                                value="0.2,0.5",
                                label="Noise Multipliers",
                                info="Comma-separated values (0-1). Controls noise for each frame. E.g., '0.2,0.5' for start-end"
                            )
                        
                        gr.Markdown("#### ⚙️ Generation Parameters")
                        with gr.Group():
                            lora_alpha_multi = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.4, step=0.1,
                                label="LoRA Alpha",
                                info="Controls temporal consistency (1-2 recommended)"
                            )
                            steps_multi = gr.Slider(
                                minimum=10, maximum=50, value=10, step=5,
                                label="Inference Steps"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Text Prompts")
                        prompt_multi = gr.Textbox(
                            lines=4,
                            label="Prompt",
                            placeholder="Describe the transition or sequence you want to generate..."
                        )
                        negative_prompt_multi = gr.Textbox(
                            lines=3,
                            value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                            label="Negative Prompt"
                        )
                        
                        generate_multi_btn = gr.Button("🎬 Generate Multi-Frame Video", variant="primary", size="lg", elem_classes=["generate-btn", "primary-btn"])
                        
                        gr.Markdown("#### 📹 Output")
                        video_output_multi = gr.Video(label="Generated Video")
                        status_multi = gr.Textbox(label="Status", interactive=False)
                
                # Demo examples for Multi-Frame
                gr.Markdown("### 🎭 Demo Examples")
                with gr.Accordion("Example 1: Start-End Transition", open=False):
                    gr.Markdown("""
                    **Prompt:** "Plastic injection machine opens releasing a soft inflatable figure..."
                    - **Conditioning Position:** 0,20 (first and last frame)
                    - **Noise Multiplier:** 0.2,0.5
                    - **LoRA Alpha:** 1.4
                    """)
                    gr.Button("Load Example 1").click(
                        lambda: ("0,20", "0.2,0.5", 1.4, "Plastic injection machine opens releasing a soft inflatable foamy morphing sticky figure over a hand. Isometric. Low light. Dramatic light. Macro shot. Real footage"),
                        outputs=[cond_position_multi, noise_multipliers_multi, lora_alpha_multi, prompt_multi]
                    )
                
                with gr.Accordion("Example 2: Multi-Frame Sequence", open=False):
                    gr.Markdown("""
                    **Prompt:** "Smooth transformation sequence with gradual changes..."
                    - **Conditioning Position:** 0,10,20 (beginning, middle, end)
                    - **Noise Multiplier:** 0.2,0.4,0.6
                    - **LoRA Alpha:** 1.5
                    """)
                    gr.Button("Load Example 2").click(
                        lambda: ("0,10,20", "0.2,0.4,0.6", 1.5, "A smooth transformation sequence showing gradual morphing with consistent lighting and style throughout the video."),
                        outputs=[cond_position_multi, noise_multipliers_multi, lora_alpha_multi, prompt_multi]
                    )
            
            # Tab 3: Video-to-Video
            with gr.TabItem("🎥 Video-to-Video"):
                gr.Markdown("""
                ### Video-to-Video Generation
                Transform existing videos with various conditioning strategies:
                - **Video Completion**: Fill in missing parts using start-end frames
                - **Video Extension**: Extend video duration using initial frames
                - **Video Transition**: Create smooth transitions between scenes
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🎬 Input Video")
                        video_input = gr.File(
                            file_types=["video"],
                            label="Upload Video (minimum 81 frames)"
                        )
                        
                        gr.Markdown("#### 🎯 Conditioning Parameters")
                        with gr.Group():
                            cond_position_v2v = gr.Textbox(
                                value="0,20",
                                label="Conditioning Positions",
                                info="Frame indices for conditioning. E.g., '0,20' for completion, '0,1,2,3' for extension"
                            )
                            noise_multipliers_v2v = gr.Textbox(
                                value="0.3,0.3",
                                label="Noise Multipliers",
                                info="Noise levels for each conditioning frame"
                            )
                        
                        gr.Markdown("#### ⚙️ Generation Parameters")
                        with gr.Group():
                            lora_alpha_v2v = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.4, step=0.1,
                                label="LoRA Alpha"
                            )
                            steps_v2v = gr.Slider(
                                minimum=10, maximum=50, value=10, step=5,
                                label="Inference Steps"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Text Prompts")
                        prompt_v2v = gr.Textbox(
                            lines=4,
                            label="Prompt",
                            placeholder="Describe how you want to transform the video..."
                        )
                        negative_prompt_v2v = gr.Textbox(
                            lines=3,
                            value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                            label="Negative Prompt"
                        )
                        
                        generate_v2v_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_classes=["generate-btn", "primary-btn"])
                        
                        gr.Markdown("#### 📹 Output")
                        video_output_v2v = gr.Video(label="Generated Video")
                        status_v2v = gr.Textbox(label="Status", interactive=False)
                
                # Demo examples for V2V
                gr.Markdown("### 🎭 Demo Examples")
                with gr.Accordion("Example 1: Video Completion", open=False):
                    gr.Markdown("""
                    **Prompt:** "Piggy bank surfing a tube in Teahupoo wave at dusk..."
                    - **Conditioning Position:** 0,20 (start and end frames)
                    - **Noise Multiplier:** 0.3,0.3
                    """)
                    gr.Button("Load Example 1").click(
                        lambda: ("0,20", "0.3,0.3", "Piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film"),
                        outputs=[cond_position_v2v, noise_multipliers_v2v, prompt_v2v]
                    )
                
                with gr.Accordion("Example 2: Video Extension", open=False):
                    gr.Markdown("""
                    **Prompt:** "Piggy bank surfing a tube in Teahupoo wave at dusk..."
                    - **Conditioning Position:** 0,1,2,3 (first 4 latent frames)
                    - **Noise Multiplier:** 0.0,0.3,0.4,0.5
                    """)
                    gr.Button("Load Example 2").click(
                        lambda: ("0,1,2,3", "0.0,0.3,0.4,0.5", "Piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film"),
                        outputs=[cond_position_v2v, noise_multipliers_v2v, prompt_v2v]
                    )
            
            # Tab 4: Text-to-Video
            with gr.TabItem("📝 Text-to-Video"):
                gr.Markdown("""
                ### Text-to-Video Generation
                Generate videos directly from text descriptions. Create entirely new video content from your imagination!
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Text Prompts")
                        prompt_t2v = gr.Textbox(
                            lines=6,
                            label="Prompt",
                            placeholder="Describe the video you want to create in detail...",
                            value="A person is enjoying a meal of spaghetti with a fork in a cozy, dimly lit Italian restaurant."
                        )
                        negative_prompt_t2v = gr.Textbox(
                            lines=4,
                            value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                            label="Negative Prompt"
                        )
                        
                        gr.Markdown("#### ⚙️ Generation Parameters")
                        with gr.Group():
                            lora_alpha_t2v = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.4, step=0.1,
                                label="LoRA Alpha",
                                info="Controls generation quality and consistency"
                            )
                            steps_t2v = gr.Slider(
                                minimum=10, maximum=50, value=10, step=5,
                                label="Inference Steps"
                            )
                    
                    with gr.Column(scale=1):
                        generate_t2v_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg", elem_classes=["generate-btn", "primary-btn"])
                        
                        gr.Markdown("#### 📹 Output")
                        video_output_t2v = gr.Video(label="Generated Video")
                        status_t2v = gr.Textbox(label="Status", interactive=False)
                
                # Demo examples for T2V
                gr.Markdown("### 🎭 Demo Examples")
                with gr.Accordion("Example 1: Restaurant Scene", open=True):
                    gr.Markdown("""
                    **Prompt:** "A person enjoying spaghetti in a cozy Italian restaurant..."
                    """)
                    gr.Button("Load Example 1").click(
                        lambda: "A person is enjoying a meal of spaghetti with a fork in a cozy, dimly lit Italian restaurant. The person has warm, friendly features and is dressed casually but stylishly in jeans and a colorful sweater. They are sitting at a small, round table, leaning slightly forward as they eat with enthusiasm. The spaghetti is piled high on their plate, with some strands hanging over the edge. The background shows soft lighting from nearby candles and a few other diners in the corner, creating a warm and inviting atmosphere. The scene captures a close-up view of the person's face and hands as they take a bite of spaghetti, with subtle movements of their mouth and fork. The overall style is realistic with a touch of warmth and authenticity, reflecting the comfort of a genuine dining experience.",
                        outputs=[prompt_t2v]
                    )
                
                with gr.Accordion("Example 2: Space Adventure", open=False):
                    gr.Markdown("""
                    **Prompt:** "A female climber rock climbing on an asteroid in deep space..."
                    """)
                    gr.Button("Load Example 2").click(
                        lambda: "A low-angle, long exposure shot of a lone female climber, wearing shorts and tank top rock climbing on a massive asteroid in deep space. The climber is suspended against a star-filled void. Dramatic shadows across the asteroid's rugged surface, emphasizing the climber's isolation and the scale of the space rock. Dust particles float in the light beams, catching the light. The climber moves methodically, with focused determination.",
                        outputs=[prompt_t2v]
                    )
        
        # Demo Gallery Section
        with gr.Group():
            gr.HTML("""
            <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, rgba(34, 211, 238, 0.1) 0%, rgba(167, 139, 250, 0.1) 100%); border-radius: 20px; margin: 20px 0; border: 1px solid rgba(34, 211, 238, 0.2);">
                <h2 style="background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-accent) 100%); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; font-size: 2.2em;">
                    🎬 Demo Gallery - See Pusa V1.0 in Action!
                </h2>
                <p style="font-size: 1.2em; line-height: 1.6; margin-bottom: 15px; color: var(--color-text-light);">
                    Explore real examples showcasing the power and versatility of Pusa V1.0 across different generation modes.
                </p>
                <p style="font-size: 1.0em; margin-bottom: 10px; color: var(--color-text-medium); font-style: italic;">
                    📂 Note: Demo files should be placed in ./demos/ and ./assets/ directories to display properly.
                </p>
            </div>
            """)
            
            with gr.Tabs():
                # Image-to-Video Demo
                with gr.TabItem("🎨 I2V Demo Results"):
                    gr.Markdown("### 📷➡️🎬 Image-to-Video Generation Example")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🖼️ Input Image")
                            demo_input_image = gr.Image(
                                value=safe_file_path("./demos/input_image.jpg"),
                                label="Monk Meditation Scene", 
                                interactive=False
                            )
                            gr.Markdown("""
                            **Settings Used:**
                            - **Prompt:** "A wide-angle shot shows a serene monk meditating perched a top of the letter E of a pile of weathered rocks that vertically spell out 'ZEN'. The rock formation is perched atop a misty mountain peak at sunrise..."
                            - **Conditioning Position:** 0 (first frame)
                            - **Noise Multiplier:** 0.2
                            - **LoRA Alpha:** 1.4
                            - **Inference Steps:** 30
                            - **File Path:** ./demos/input_image.jpg
                            """)
                        
                        with gr.Column():
                            gr.Markdown("#### 🎥 Generated Video")
                            demo_i2v_video = gr.Video(
                                value=safe_file_path("./assets/multi_frame_output_cond_0_noise_0p2.mp4"),
                                label="I2V Result - Single Image Animation",
                                height=400
                            )
                
                # Multi-Frame Demo  
                with gr.TabItem("🖼️ Multi-Frame Demo Results"):
                    gr.Markdown("### 🎯 Start-End Frame Generation Example")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🖼️ Input Frames")
                            with gr.Row():
                                start_frame = gr.Image(
                                    value=safe_file_path("./demos/start_frame.jpg"),
                                    label="Start Frame (Position 0)", 
                                    interactive=False
                                )
                                end_frame = gr.Image(
                                    value=safe_file_path("./demos/end_frame.jpg"),
                                    label="End Frame (Position 20)", 
                                    interactive=False
                                )
                            gr.Markdown("""
                            **Settings Used:**
                            - **Prompt:** "plastic injection machine opens releasing a soft inflatable foamy morphing sticky figure over a hand. isometric. low light. dramatic light. macro shot. real footage"
                            - **Conditioning Positions:** 0,20 (start and end frames)
                            - **Noise Multipliers:** 0.2,0.5
                            - **LoRA Alpha:** 1.4
                            - **Inference Steps:** 30
                            - **File Paths:** ./demos/start_frame.jpg, ./demos/end_frame.jpg
                            """)
                        
                        with gr.Column():
                            gr.Markdown("#### 🎥 Generated Video")
                            demo_multi_video = gr.Video(
                                value=safe_file_path("./assets/multi_frame_output_cond_0_20_noise_0p2_0p5.mp4"),
                                label="Start-End Frame Transition",
                                height=400
                            )
                
                # Video-to-Video Demo
                with gr.TabItem("🎥 V2V Demo Results"):
                    gr.Markdown("### 🎬➡️🎬 Video Extension Example")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📹 Input Video")
                            demo_input_video = gr.Video(
                                value=safe_file_path("./demos/input_video.mp4"),
                                label="Original Video (Input for Extension)",
                                height=300
                            )
                            gr.Markdown("""
                            **Settings Used:**
                            - **Prompt:** "piggy bank surfing a tube in teahupo'o wave dusk light cinematic shot shot in 35mm film"
                            - **Conditioning Positions:** 0,1,2,3 (first 4 latent frames)
                            - **Noise Multipliers:** 0.0,0.3,0.4,0.5
                            - **LoRA Alpha:** 1.4
                            - **Inference Steps:** 30
                            - **Task:** Video Extension (using first 13 frames as conditioning)
                            - **File Path:** ./demos/input_video.mp4
                            """)
                        
                        with gr.Column():
                            gr.Markdown("#### 🎥 Extended Video")
                            demo_v2v_video = gr.Video(
                                value=safe_file_path("./assets/v2v_input_video_cond_0_1_2_3_noise_0p0_0p3_0p4_0p5.mp4"),
                                label="V2V Extension Result (81 frames total)",
                                height=400
                            )
                
                # Text-to-Video Demo
                with gr.TabItem("📝 T2V Demo Results"):
                    gr.Markdown("### 📝➡️🎬 Text-to-Video Generation Example")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📝 Text Prompt")
                            gr.Textbox(
                                value="A person is enjoying a meal of spaghetti with a fork in a cozy, dimly lit Italian restaurant. The person has warm, friendly features and is dressed casually but stylishly in jeans and a colorful sweater. They are sitting at a small, round table, leaning slightly forward as they eat with enthusiasm. The spaghetti is piled high on their plate, with some strands hanging over the edge. The background shows soft lighting from nearby candles and a few other diners in the corner, creating a warm and inviting atmosphere. The scene captures a close-up view of the person's face and hands as they take a bite of spaghetti, with subtle movements of their mouth and fork. The overall style is realistic with a touch of warmth and authenticity, reflecting the comfort of a genuine dining experience.",
                                label="Input Prompt",
                                lines=8,
                                interactive=False
                            )
                            gr.Markdown("""
                            **Settings Used:**
                            - **LoRA Alpha:** 1.4
                            - **Inference Steps:** 30
                            - **Negative Prompt:** "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality..."
                            - **Task:** Pure Text-to-Video Generation (81 frames)
                            - **File Path:** ./assets/t2v_output.mp4
                            """)
                        
                        with gr.Column():
                            gr.Markdown("#### 🎥 Generated Video")
                            demo_t2v_video = gr.Video(
                                value=safe_file_path("./assets/t2v_output.mp4"),
                                label="T2V Result - Generated from Text Only",
                                height=400
                            )
                
                # Comparison Section
                with gr.TabItem("📊 Method Comparison"):
                    gr.Markdown("### 🆚 Pusa V1.0 vs Other Methods")
                    
                    with gr.Group():
                        gr.Markdown("""
                        #### 🏆 Performance Highlights
                        
                        **Pusa V1.0 achieves breakthrough efficiency:**
                        - 💰 **Training Cost:** Only $500 vs $10,000+ for comparable methods
                        - 📊 **Data Efficiency:** 4K training samples vs 100K+ typically required
                        - 🎯 **Performance:** Surpasses Wan-I2V on Vbench-I2V metrics
                        - 🔧 **Versatility:** 4 generation modes in one unified model
                        """)
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("""
                                #### ⚡ Technical Innovation
                                - **Vectorized Timestep Adaptation (VTA)** for fine-grained temporal control
                                - **LoRA with large rank (512)** for efficient approximation of full fine-tuning
                                - **Multi-task capabilities** without task-specific training
                                - **Preserved T2V abilities** while gaining new I2V/V2V capabilities
                                """)
                            
                            with gr.Column():
                                gr.Markdown("""
                                #### 🎮 Usage Modes
                                1. **Image-to-Video (I2V):** Single image → 81-frame video
                                2. **Multi-Frame:** Start-end frames → smooth transition
                                3. **Video-to-Video (V2V):** Completion, extension, editing
                                4. **Text-to-Video (T2V):** Pure text prompt → video
                                """)
                    
                    gr.HTML("""
                    <div style="text-align: center; padding: 20px; background: rgba(34, 211, 238, 0.1); border-radius: 15px; margin: 20px 0;">
                        <h3 style="color: var(--color-primary); margin-bottom: 15px;">
                            🔬 Research Impact
                        </h3>
                        <p style="font-size: 1.1em; line-height: 1.6;">
                            Pusa V1.0 demonstrates that <strong>high-quality video generation doesn't require massive computational resources</strong>. 
                            Our vectorized timestep adaptation approach opens new possibilities for democratizing video AI research and applications.
                        </p>
                    </div>
                    """)

        # Information section
        with gr.Group():
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: rgba(30, 41, 59, 0.6); border-radius: 15px; margin: 20px 0; backdrop-filter: blur(12px);">
                <h2 style="background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-secondary) 100%); background-clip: text; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 15px;">
                    📖 About Pusa V1.0
                </h2>
                <p style="font-size: 1.1em; line-height: 1.6; margin-bottom: 20px; color: var(--color-text-light);">
                    <strong>Pusa V1.0</strong> leverages <span style="color: var(--color-primary);">vectorized timestep adaptation (VTA)</span> for fine-grained temporal control 
                    within a unified video diffusion framework. The model achieves unprecedented efficiency, surpassing Wan-I2V on Vbench-I2V with only <span style="color: var(--color-secondary);">$500 training cost</span> and 4k data.
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 💡 Pro Tips for Best Results
                    
                    🎚️ **LoRA Alpha**: Use values between 1-2 for optimal balance between quality and consistency
                    
                    🔊 **Noise Multipliers**: Lower values (0.0-0.3) for faithful conditioning, higher values (0.4-1.0) for more variation
                    
                    📍 **Conditioning Positions**: Frame 0 is first frame, frame 20 is last frame in the 21-frame latent space
                    
                    ✍️ **Prompts**: Be descriptive and specific for better results
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### 🔗 Important Links
                    
                    🌐 **[Project Page](https://yaofang-liu.github.io/Pusa_Web/)** - Official project website
                    
                    📄 **[Technical Report](https://arxiv.org/abs/2507.16116)** - Detailed research paper
                    
                    🤗 **[Model on HuggingFace](https://huggingface.co/RaphaelLiu/PusaV1)** - Download models
                    
                    📚 **[Training Dataset](https://huggingface.co/datasets/RaphaelLiu/PusaV1_training)** - Training data
                    """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 30px; margin-top: 40px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">
            <p style="font-size: 1.2em; margin-bottom: 10px;">
                <strong>✨ Made with ❤️ for the AI Community ✨</strong>
            </p>
            <p style="opacity: 0.8;">
                Experience the future of video generation with Pusa V1.0 🚀
            </p>
        </div>
        """)
        
        # Event handlers
        generate_i2v_btn.click(
            fn=demo_instance.generate_i2v_video,
            inputs=[image_input, prompt_i2v, noise_multiplier_i2v,
                   lora_alpha_i2v, steps_i2v, negative_prompt_i2v],
            outputs=[video_output_i2v, status_i2v]
        )
        
        generate_multi_btn.click(
            fn=demo_instance.generate_multi_frames_video,
            inputs=[image1_input, image2_input, image3_input, num_images, prompt_multi, cond_position_multi, noise_multipliers_multi, 
                   lora_alpha_multi, steps_multi, negative_prompt_multi],
            outputs=[video_output_multi, status_multi]
        )
        
        generate_v2v_btn.click(
            fn=demo_instance.generate_v2v_video,
            inputs=[video_input, prompt_v2v, cond_position_v2v, noise_multipliers_v2v,
                   lora_alpha_v2v, steps_v2v, negative_prompt_v2v],
            outputs=[video_output_v2v, status_v2v]
        )
        
        generate_t2v_btn.click(
            fn=demo_instance.generate_t2v_video,
            inputs=[prompt_t2v, lora_alpha_t2v, steps_t2v, negative_prompt_t2v],
            outputs=[video_output_t2v, status_t2v]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    ) 
