# Runs the real CLI (src/app.py) but also saves the returned images, which the one-shot CLI drops.
import os, runpy, sys
sys.path.insert(0, "/app/src")
import context
orig = context.Context.generate_text_to_image
def wrapped(self, *a, **k):
    imgs = orig(self, *a, **k)
    for i, im in enumerate(imgs or []):
        im.save(f"/out/{os.environ['CASE']}_{i}.png")
    return imgs
context.Context.generate_text_to_image = wrapped
sys.argv = ["/app/src/app.py"] + sys.argv[1:]
runpy.run_path("/app/src/app.py", run_name="__main__")
