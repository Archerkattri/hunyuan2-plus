# Code


We designed a diffusers-like API to use our shape generation model - Hunyuan3D-DiT and texture synthesis model -
Hunyuan3D-Paint.

You could assess **Hunyuan3D-DiT** via:

```python
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
image = Image.open('assets/demo.png').convert('RGBA')
mesh = pipeline(image=image)[0]
```

The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other
format) file.

For **Hunyuan3D-Paint**, do the following:

```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
image = Image.open('assets/demo.png').convert('RGBA')
mesh = pipeline(image=image)[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image=image)
```

Please visit [examples](examples) folder for more advanced usage, such as **multiview image to 3D generation** and *
*texture generation
for handcrafted mesh**.
