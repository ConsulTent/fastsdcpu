import backend.lcm_text_to_image as m
from backend.models.lcmdiffusion_setting import LCMDiffusionSetting as S

class Reached(Exception): pass
def stop(name):
    def f(*a, **k): raise Reached(name)
    return f
for attr, name in [("get_lcm_model_pipeline", "lcm"), ("get_lcm_lora_pipeline", "lcm_lora"),
                   ("GGUFDiffusion", "gguf"), ("get_ov_diffusion_pipeline", "ov_diffusion"),
                   ("get_ov_text_to_image_pipeline", "ov_sd"), ("OvHcLatentConsistency", "hetero")]:
    setattr(m, attr, stop(name))

def load(**kw):
    t = m.LCMTextToImage()
    try:
        t.init(lcm_diffusion_setting=S(**kw))
    except Reached as r:
        return t, str(r)
    raise AssertionError("no loader reached")

KLEIN = "rupeshs/flux2-klein-4b-int4-ov"
FLUX3 = "rupeshs/flux3-image-klein-int4-ov"
SQUARE = "rupeshs/sd15-lcm-square-openvino-int8"
FLUX1 = "OpenVINO/FLUX.1-schnell-int4-ov"

def klein_openvino_rejects_taesd():
    for mid in (KLEIN, FLUX3):
        try:
            load(openvino_lcm_model_id=mid, use_openvino=True, use_tiny_auto_encoder=True)
        except ValueError as e:
            assert "Flux Klein" in str(e), e
            continue
        raise AssertionError(f"{mid} accepted TAESD in OpenVINO mode")

def stale_ov_id_ignored_outside_openvino():
    for mode, want in (({}, "lcm"), ({"use_lcm_lora": True}, "lcm_lora"), ({"use_gguf_model": True}, "gguf")):
        for mid in (KLEIN, SQUARE):
            t, hit = load(openvino_lcm_model_id=mid, use_tiny_auto_encoder=True,
                          negative_prompt="blurry", **mode)
            assert (hit, t.ov_model_id) == (want, ""), (mode, mid, hit, t.ov_model_id)

def square_still_hetero_in_openvino():
    _, hit = load(openvino_lcm_model_id=SQUARE, use_openvino=True)
    assert hit == "hetero", hit

def flux1_uses_ov_diffusion_loader():
    _, hit = load(openvino_lcm_model_id=FLUX1, use_openvino=True)
    assert hit == "ov_diffusion", hit

failed = 0
for check in (klein_openvino_rejects_taesd, stale_ov_id_ignored_outside_openvino,
              square_still_hetero_in_openvino, flux1_uses_ov_diffusion_loader):
    try:
        check(); print("PASS", check.__name__)
    except Exception as e:
        failed += 1; print("FAIL", check.__name__, type(e).__name__, e)
raise SystemExit(failed)
