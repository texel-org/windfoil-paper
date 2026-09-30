#!/usr/bin/env python3
import argparse
import asyncio
import json

print("Loading CLIP runtime...", flush=True)

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
_MODELS = {}
CLIPAG_URL = "https://zenodo.org/records/10446026/files/CLIPAG_ViTB32.pt"


def resolve_pretrained(pretrained):
    if pretrained != "clipag":
        return pretrained
    import os
    import urllib.request

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLIPAG_ViTB32.pt")
    if not os.path.exists(path):
        raw = path + ".download"
        if not os.path.exists(raw):
            print(f"Downloading CLIPAG weights from {CLIPAG_URL}...", flush=True)
            urllib.request.urlretrieve(CLIPAG_URL, raw)
        # The published checkpoint pickles numpy scalars, which torch>=2.6
        # rejects under weights_only; re-save as a tensor-only state dict.
        checkpoint = torch.load(raw, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        torch.save(state, path)
        os.remove(raw)
    return path


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class UnfoldPatchEmbed(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.out_channels = conv.out_channels
        self.kernel = conv.kernel_size[0]
        self.stride = conv.stride[0]

    def forward(self, image):
        patches = F.unfold(image, kernel_size=self.kernel, stride=self.stride)
        result = self.weight.reshape(self.out_channels, -1) @ patches
        if self.bias is not None:
            result = result + self.bias[:, None]
        height = (image.shape[-2] - self.kernel) // self.stride + 1
        width = (image.shape[-1] - self.kernel) // self.stride + 1
        return result.reshape(image.shape[0], self.out_channels, height, width)


def patch_mps_stem(model, device):
    if device.type != "mps":
        return
    for name, module in list(model.named_modules()):
        if not (isinstance(module, torch.nn.Conv2d)
                and module.kernel_size == module.stride
                and module.padding == (0, 0)
                and module.groups == 1):
            continue
        parent = model
        *path, leaf = name.split(".")
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, UnfoldPatchEmbed(module))


def load_model(config, device):
    name = config.get("model", "ViT-B-32-quickgelu")
    pretrained = config.get("pretrained", "openai")
    key = (name, str(pretrained), str(device))
    if key not in _MODELS:
        print(f"Loading CLIP {name} ({pretrained})...", flush=True)
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            name, pretrained=resolve_pretrained(pretrained)
        )
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        patch_mps_stem(model, device)
        _MODELS[key] = (model, open_clip.get_tokenizer(name))
        print("CLIP ready.", flush=True)
    return _MODELS[key]


class ClipLoss:
    def __init__(self, config, device):
        self.device = device
        self.model, tokenizer = load_model(config, device)
        prompts = config["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        # A prompt of the form "embed:<npz-path>:<key>" targets a precomputed
        # unit embedding (e.g. a PCA-grid cell) instead of encoded text.
        if len(prompts) == 1 and prompts[0].startswith("embed:"):
            _, path, key = prompts[0].split(":", 2)
            vec = torch.tensor(np.load(path)[key], device=device, dtype=torch.float32)
            self.text = F.normalize(vec.view(1, -1), dim=-1)
        else:
            with torch.no_grad():
                text = F.normalize(self.model.encode_text(tokenizer(prompts).to(device)), dim=-1)
                self.text = F.normalize(text.mean(dim=0, keepdim=True), dim=-1)
        image_size = getattr(self.model.visual, "image_size", 224)
        self.image_size = int(image_size if isinstance(image_size, int) else image_size[0])
        mean = getattr(self.model.visual, "image_mean", None) or (0.48145466, 0.4578275, 0.40821073)
        std = getattr(self.model.visual, "image_std", None) or (0.26862954, 0.26130258, 0.27577711)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.augs = int(config.get("augs", 4))
        self.seed = config.get("seed")
        self.interpolation = config.get("interp", "bilinear")
        if self.augs:
            import torchvision.transforms as transforms
            # Perspective padding is "outside the canvas": the client sends the
            # display-space background so crops never contain values the render
            # itself could not produce (literal 1.0 is unreachable under a
            # saturating tonemap).
            fill = config.get("fill", 1)
            fill = [float(v) for v in fill] if isinstance(fill, (list, tuple)) else float(fill)
            self.augment = transforms.Compose([
                transforms.RandomPerspective(fill=fill, p=1, distortion_scale=0.5),
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 0.9), antialias=True),
            ])

    def embed(self, images):
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode=self.interpolation,
                align_corners=False,
            )
        return F.normalize(self.model.encode_image((images - self.mean) / self.std), dim=-1)

    def loss_and_grad(self, image, step):
        if self.seed is not None:
            seed = (int(self.seed) + int(step or 0)) & 0x7FFFFFFF
            torch.manual_seed(seed)
            if self.device.type == "mps":
                torch.mps.manual_seed(seed)
        image = image.requires_grad_(True)
        # Clamp INSIDE the autograd graph: an out-of-range display pixel gets
        # the true clamp subgradient (zero) instead of a gradient evaluated at
        # the boundary but applied to the unclamped value by the host chain.
        display = image.clamp(0, 1)
        batch = (
            torch.cat([self.augment(display) for _ in range(self.augs)])
            if self.augs
            else display
        )
        loss = (1 - self.embed(batch) @ self.text.T).mean()
        gradient, = torch.autograd.grad(loss, image)
        return float(loss.detach().cpu()), gradient.detach()


def rgba_tensor(data, width, height, device):
    rgba = np.frombuffer(data, dtype=np.float32).reshape(height, width, 4)
    rgb = torch.from_numpy(np.ascontiguousarray(rgba[..., :3])).to(device)
    # No clamp here: loss_and_grad clamps inside the autograd graph so the
    # returned gradient is consistent with the pixel values the host holds.
    return rgb.permute(2, 0, 1).unsqueeze(0)


def rgba_gradient(gradient, width, height):
    result = np.zeros((height, width, 4), dtype=np.float32)
    result[..., :3] = gradient.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return result.tobytes()


async def handle(socket, device):
    objective = None
    pending = None
    async for message in socket:
        if isinstance(message, bytes):
            if not pending or objective is None:
                continue
            header, pending = pending, None
            image = rgba_tensor(message, header["w"], header["h"], device)
            loss, gradient = objective.loss_and_grad(image, header.get("step"))
            await socket.send(json.dumps({"type": "result", "id": header["id"], "loss": loss}))
            await socket.send(rgba_gradient(gradient, header["w"], header["h"]))
            continue
        data = json.loads(message)
        if data["type"] == "config":
            try:
                objective = ClipLoss(data, device)
                await socket.send(json.dumps({"type": "ready", "device": str(device)}))
            except Exception as error:
                await socket.send(json.dumps({"type": "error", "message": str(error)}))
        elif data["type"] == "grad":
            pending = data


async def serve(args):
    import websockets

    device = pick_device(args.device)
    print(f"CLIP loss server ws://{args.host}:{args.port} ({device})", flush=True)

    async def connection(socket):
        try:
            await handle(socket, device)
        except websockets.exceptions.ConnectionClosed:
            pass

    async with websockets.serve(connection, args.host, args.port, max_size=256 * 1024 * 1024):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    try:
        asyncio.run(serve(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
