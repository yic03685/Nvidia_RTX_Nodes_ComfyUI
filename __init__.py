import torch

try:
    import nvvfx
    NVVFX_AVAILABLE = True
except ImportError:
    NVVFX_AVAILABLE = False
    print("[RTXVideoSuperResolution] nvidia-vfx not available — node disabled")


class RTXVideoSuperResolution:
    """NVIDIA RTX Video Super Resolution using nvvfx SDK.

    Legacy ComfyUI node format for compatibility with v0.17.x.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_type": (["scale by multiplier", "target dimensions"],),
            },
            "optional": {
                "resize_type.scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.01}),
                "width": ("INT", {"default": 1920, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1080, "min": 64, "max": 8192, "step": 8}),
                "quality": (["LOW", "MEDIUM", "HIGH", "ULTRA"], {"default": "HIGH"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_images",)
    FUNCTION = "execute"
    CATEGORY = "image/upscaling"

    def execute(self, images, resize_type, quality="HIGH", **kwargs):
        scale = kwargs.get("resize_type.scale", 2.0)
        width = kwargs.get("width", 1920)
        height = kwargs.get("height", 1080)

        b, h, w, c = images.shape

        if resize_type == "scale by multiplier":
            output_width = int(w * scale)
            output_height = int(h * scale)
        else:
            output_width = width
            output_height = height

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16
        out_pixels = output_width * output_height
        batch_size = max(1, MAX_PIXELS // out_pixels)

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            out_tensor = torch.empty((images.shape[0], output_height, output_width, c), device=images.device, dtype=images.dtype)
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]
                batch_cuda = batch.cuda().permute(0, 3, 1, 2).float().contiguous()

                for j in range(batch_cuda.shape[0]):
                    input_frame = batch_cuda[j]
                    dlpack_out = sr.run(input_frame).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)

        return (out_tensor,)


if NVVFX_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "RTXVideoSuperResolution": RTXVideoSuperResolution,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "RTXVideoSuperResolution": "RTX Video Super Resolution",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
