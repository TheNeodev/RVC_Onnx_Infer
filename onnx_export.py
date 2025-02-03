import os
import sys
import onnx
import json
import torch
import warnings

sys.path.append(os.getcwd())

from lib.synthesizers import SynthesizerONNX

warnings.filterwarnings("ignore")

def onnx_exporter(input_path, output_path, device="cpu"):
    cpt = (torch.load(input_path, map_location="cpu") if os.path.isfile(input_path) else None)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

    model_name, model_author, epochs, steps, version, f0, model_hash, vocoder, creation_date = cpt.get("model_name", None), cpt.get("author", None), cpt.get("epoch", None), cpt.get("step", None), cpt.get("version", "v1"), cpt.get("f0", 1), cpt.get("model_hash", None), cpt.get("vocoder", "Default"), cpt.get("creation_date", None)
    text_enc_hidden_dim = 768 if version == "v2" else 256
    tgt_sr = cpt["config"][-1]

    net_g = SynthesizerONNX(*cpt["config"], use_f0=f0, text_enc_hidden_dim=text_enc_hidden_dim, vocoder=vocoder, checkpointing=False)
    net_g.load_state_dict(cpt["weight"], strict=False)

    if f0:
        args = (torch.rand(1, 200, text_enc_hidden_dim).to(device), torch.tensor([200]).long().to(device), torch.LongTensor([0]).to(device), torch.rand(1, 192, 200).to(device), torch.randint(size=(1, 200), low=5, high=255).to(device), torch.rand(1, 200).to(device))
        input_names = ["phone", "phone_lengths", "ds", "rnd", "pitch", "pitchf"]
        dynamic_axes = {"phone": [1], "rnd": [2], "pitch": [1], "pitchf": [1]}
    else:
        args = (torch.rand(1, 200, text_enc_hidden_dim).to(device), torch.tensor([200]).long().to(device), torch.LongTensor([0]).to(device), torch.rand(1, 192, 200).to(device))
        input_names = ["phone", "phone_lengths", "ds", "rnd"]
        dynamic_axes = {"phone": [1], "rnd": [2]}

    torch.onnx.export(net_g, args, output_path, do_constant_folding=False, opset_version=17, verbose=False, input_names=input_names, output_names=["audio"], dynamic_axes=dynamic_axes)

    model = onnx.load(output_path)
    model.metadata_props.append(onnx.StringStringEntryProto(key="model_info", value=json.dumps({"model_name": model_name, "author": model_author, "epoch": epochs, "step": steps, "version": version, "sr": tgt_sr, "f0": f0, "model_hash": model_hash, "creation_date": creation_date, "vocoder": vocoder, "text_enc_hidden_dim": text_enc_hidden_dim})))

    onnx.save(model, output_path)
    return output_path

if __name__ == "__main__":
    input_model = "model.pth"
    output_model = "model.onnx"
    onnx_exporter(input_model, output_model)