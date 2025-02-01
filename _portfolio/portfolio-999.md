---
title: "ControlNet + LoRA Line-Art Colorization Workflow"
excerpt: "This is a course project. [Slide](https://docs.google.com/presentation/d/1--OoNGXd8Wfc12vTyD4rxgOoaCXto4GL/edit?usp=sharing&ouid=110097611116200166954&rtpof=true&sd=true) & [Paper](https://drive.google.com/file/d/1jjJ6YeQM4jzgP4o8op2zyBr9xubnGool/view?usp=sharing)<br/><img src='/images/workflow.png'>"
collection: portfolio
---

<img src='/images/workflow.png'>

Diffusion models have revolutionized image generation, yet they consistently struggle with precise character depiction, particularly when attempting to reproduce nuanced, character-specific visual features. Current generative approaches often fail to maintain the delicate balance between preserving structural integrity and capturing unique character attributes, resulting in images that either lack detailed authenticity or deviate significantly from the intended input.

The motivation for this research stems from the growing demand for more sophisticated and controllable image generation techniques in domains such as digital art, character design, and creative media. As artists, designers, and content creators increasingly rely on AI-assisted tools, there is a critical need for methods that can accurately translate conceptual designs into visually faithful representations. The ability to generate images that not only capture the essence of a character but also respect the original artistic intent represents a significant technological and creative challenge.

Previous approaches to addressing this limitation have explored various techniques. Prompt-guided image-to-image methods attempted to use text descriptions to guide generation, but often failed to maintain line art contours. ControlNet demonstrated promising results in the preservation of structural characteristics, while low rank adaptation (LoRA) models showed potential in learning character-specific characteristics. However, each of these individual approaches failed to provide a comprehensive solution, with limitations ranging from structural inaccuracies to incomplete character representation.

The key contributions of this work are:

Developing a hybrid workflow that integrates ControlNet and LoRA into a unified diffusion model approach

Demonstrating the effectiveness of this method through comprehensive baseline comparisons

Establishing a novel framework for character-specific image generation that maintains both structural fidelity and unique character attributes
