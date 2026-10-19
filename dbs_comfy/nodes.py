"""ComfyUI nodes for Dynamic Band Splitting (DBS) speech / music / sound-effects separation. Nothing is downloaded at run time."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

DBSModel = io.Custom("DBS_MODEL")
DEVICES = ("auto", "cpu")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


def _device(choice: str):
    import comfy.model_management as mm
    import torch

    if choice == "cpu":
        return torch.device("cpu")
    device = mm.get_torch_device()
    return device if device.type == "cuda" else torch.device("cpu")  # the complex-STFT models are run on CPU or CUDA only


class DBSLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DBSLoader",
            display_name="DBS Loader",
            category="loaders/dbs",
            description=("Loads a Dynamic Band Splitting separator (DnR v2: music, sound effects, speech) from models/dbs/<model>/ "
                         "(config.json and model.safetensors)."),
            inputs=[
                io.Combo.Input("model", options=list(runtime.MODELS), default="DBS-TIGER-DnR",
                               tooltip="DBS-TIGER-DnR (1.0M parameters, better: 10.1 dB SI-SDR on DnR v2) or DBS-BSRNN-DnR (4.7M, 9.1 dB, faster on CPU)"),
                io.Combo.Input("device", options=list(DEVICES), default="auto",
                               tooltip="auto: ComfyUI's GPU when it is a CUDA device, otherwise the CPU"),
            ],
            outputs=[DBSModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, device) -> io.NodeOutput:
        if model not in runtime.MODELS or device not in DEVICES:
            raise ValueError(f"unknown model {model!r} or device {device!r}")
        folder = paths.model_dir(model, runtime.FILES)
        dev = _device(device)
        runtime.load(folder, model, dev)  # fail here, not later, when a file is missing or broken
        return io.NodeOutput({"folder": folder, "name": model, "device": str(dev)})


class DBSSeparate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DBSSeparate",
            display_name="DBS Separate (music / sfx / speech)",
            category="audio/dbs",
            description=("Separates a soundtrack into music, sound effects and speech with the official overlap-add windows "
                         "(DBS-TIGER 6 s every 3 s, DBS-BSRNN 3 s every 1.5 s). The models work on 44.1 kHz mono; other rates "
                         "are resampled with ComfyUI's resampler. The three outputs add up to the (mono or per-channel) input."),
            inputs=[
                DBSModel.Input("model"),
                io.Audio.Input("audio"),
                io.Combo.Input("channels", options=list(runtime.CHANNEL_MODES), default="mono",
                               tooltip="mono: the channel mean, as the official usage does. each channel: every channel separated on its own (keeps stereo; takes longer)"),
                io.Combo.Input("output_sample_rate", options=list(runtime.OUTPUT_RATES), default="44100",
                               tooltip="44100: the model's rate. input: resampled back to the input's rate"),
            ],
            outputs=[io.Audio.Output(display_name="music"), io.Audio.Output(display_name="sfx"),
                     io.Audio.Output(display_name="speech"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, audio, channels, output_sample_rate) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        net = runtime.load(model["folder"], model["name"], model["device"])
        total = runtime.total_windows(model["name"], audio, channels)
        music, sfx, speech, report = runtime.separate_audio(net, model["name"], audio, channels, output_sample_rate,
                                                            progress=_progress(total))
        return io.NodeOutput(music, sfx, speech, json.dumps(report, indent=1))


class DBSExtension(ComfyExtension):
    async def get_node_list(self):
        return [DBSLoader, DBSSeparate]


async def comfy_entrypoint() -> DBSExtension:
    paths.register()
    return DBSExtension()
