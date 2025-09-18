---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am Xingahao Chen, a master's student at the [University of Washington](https://www.washington.edu/). 

My research interests focus on computer vision and generative AI, and I have worked on projects involving 3D Gaussian splatting (3DGS), diffusion models, LLM agents, image segmentation, and reinforcement learning.

<span style="color: #cc3333;"><strong>I am actively looking for PhD positions (Spring & Fall 2026)! If you are interested in working together or have potential PhD opportunities, please feel free to contact me. [Edu email](cxh42@uw.edu)</strong></span>

News
=====
* [04.2025] Started the Developing Immersive Experiences for AR/VR course, where I am building interactive 3DGS experiences for VR headsets.
* [01.2025] Joined the [TACO Group](https://taco-group.github.io/) at Texas A&M University as a remote research assistant under [Dr. Zhengzhong Tu](https://vztu.github.io/).
* [12.2024] Started working on the "Machine Learning for Community-Driven Coastal Erosion Monitoring and Management" project at the UW Applied Physics Laboratory with advisers [Dr. Roxanne Carini](https://www.apl.washington.edu/people/profile.php?last_name=Carini&first_name=Roxanne), [Dr. Morteza Derakhti](https://www.ce.washington.edu/facultyfinder/morteza-derakhti) and [Dr. Arindam Das](https://www.ece.uw.edu/people/arindam-das/).
* [12.2024] Enrolled in the Large Language Models course taught by [Dr. Karthik Mohan](https://www.linkedin.com/in/karthik-mohan-72a4b323/). [Course web page](https://bytesizeml.github.io/llm2025/).
* [09.2024] Enrolled in the Computer Vision course taught by [Dr. Stan Birchfield](https://research.nvidia.com/person/stan-birchfield).
* [09.2024] Began the M.S. program in Electrical and Computer Engineering at the University of Washington.

Ongoing Research
=====
### RL-guided image restoration agent
Developing a reinforcement learning controller that orchestrates denoising, deblurring, deraining, low-light enhancement, super-resolution, colorization, and face refinement modules to recover images suffering coupled degradations. The policy fuses NR-IQA scores (NIQE, BRISQUE, MUSIQ, ImageReward), visual embeddings, and action history, and is trained with PPO to learn minimal toolchains that trade quality gains against inference cost.

Evaluation spans synthetic CleanBench-style mixes and real captures to benchmark robustness against LLM/VLM schedulers such as AgenticIR and JarvisIR, with emphasis on reducing hallucinations and over-processing.

### Open-source NuRec-style scene reconstruction
Re-creating NVIDIA's recent Omniverse NuRec + 3DGUT workflow with an open toolchain so multi-sensor logs (RGB, LiDAR, IMU) can be converted into photorealistic 3D Gaussian scenes for Isaac Sim and CARLA. The roadmap covers Gaussian-based reconstruction, USD export, and lightweight viewers that mirror NuRec features like Physically Accurate Dataset streaming and Sensor RTX-quality ray tracing.

The study leans on NVIDIA's August 2025 developer demo and Newsroom briefing-where NuRec shipped with CARLA integrations, Foretellix toolchain support, and Voxel51 FiftyOne dataset hooks-to map required components before swapping in open kernels (e.g., gsplat, Nerfstudio, differentiable rendering).

Education
=====
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="/images/uwlogo.png" width="100px" style="margin-right: 20px;">
    <div>
        <p style="margin: 0;">University of Washington</p>
        <p style="margin: 0;">September 2024 ~ January 2026 (Expected)</p>
        <p style="margin: 0;">M.S. Electrical & Computer Engineering</p>
    </div>
</div>

<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <img src="/images/henulogo.png" width="100px" style="margin-right: 20px;">
    <div>
        <p style="margin: 0;">Henan University</p>
        <p style="margin: 0;">September 2020 ~ June 2024</p>
        <p style="margin: 0;">B.E. Automation</p>
    </div>
</div>

Contact
=====
**Personal email:** [xhc42@outlook.com](xhc42@outlook.com)

**Edu email:** [cxh42@uw.edu](cxh42@uw.edu)

**WeChat:** ICXH42


