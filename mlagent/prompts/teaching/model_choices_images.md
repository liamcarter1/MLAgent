# Choosing a model for images

A model for images is a **neural network**: a stack of layers that each look at small
patches of the picture and pass on what they found. The early layers learn edges and
colours, the later ones learn shapes and, eventually, whole objects. Three choices, from
cheapest to strongest.

<!--level:beginner,intermediate-->
## Tiny CNN

Two convolutional layers and nothing else. It trains in seconds even on the CPU, which
makes it the right first run: it tells you whether your data and labels line up before
you spend time on anything bigger. Expect modest accuracy on anything harder than clear
shapes on a plain background.

## Small CNN

Three convolutional blocks with batch normalisation and dropout. Still trains from
scratch, still fine on the CPU for a few thousand small images, and much better at real
photographs than the tiny one. This is the default, and the one to stay with while you
are learning what the tuning loop does.

## Pretrained ResNet-18 (transfer learning)

ResNet-18 has already been trained on a million photographs, so it arrives knowing what
edges, textures and object parts look like. **Transfer learning** means keeping that
knowledge and only teaching it your classes. It is by far the most accurate option on
real photographs, especially when you have only a few hundred images per class -- but it
is roughly ten times slower per epoch than the small CNN, so switch to the GPU runtime
before you pick it, and prefer 64px over 128px images on the CPU.

`freeze_backbone` decides how much of it you retrain: `yes` trains only the final layer
(fast, and enough when your images look like ordinary photographs), `no` retrains
everything (slower, better when your images look nothing like everyday photos -- X-rays,
say, or satellite tiles).
<!--/level-->

<!--level:expert-->
- `tiny_cnn`: 2 conv blocks, ~5k parameters. Sanity check.
- `small_cnn`: 3 conv blocks with BN and dropout. Default; CPU-viable to a few thousand
  images.
- `resnet18`: torchvision ResNet-18. `pretrained=imagenet` downloads weights (needs the
  network); `freeze_backbone=yes` trains the head only. ~10x the small CNN's cost per
  epoch; use the GPU runtime, and prefer 64px on CPU.
<!--/level-->

Whatever you pick, you are not stuck with it: the tuning step can switch families, and
*Redo a stage* regenerates the whole training project.
