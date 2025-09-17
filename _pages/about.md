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

<span style="color: #cc3333;"><strong>I am actively looking for PhD positions (Spring & Fall 2026)! If you are interested in working together or have potential PhD opportunities, please feel free to contact me.</strong></span>


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
<div class="ongoing-research-wrapper">
  <div class="paper-box ongoing-research-card">
    <div class="paper-box-image">
      <div>
        <div class="badge">In Progress</div>
        <img src="/images/workflow.png" alt="CoastalVision agent mockup" width="100%">
      </div>
    </div>
    <div class="paper-box-text" markdown="1">

<a href="https://github.com/cxh42/coastalvision-agent" class="paper-title">CoastalVision Agent: Multimodal Monitoring for Shoreline Change</a>

**Team:** Xingahao Chen, Applied Physics Laboratory at UW, TACO Group collaborators<br>
**Focus:** Build a coastal monitoring agent that fuses drone video, satellite imagery, and in-situ sensors for proactive erosion alerts.

**What we are building**

- Train a diffusion-based enhancer to stabilize drone video captured in harsh weather.
- Use 3D Gaussian splatting to maintain a live shoreline reconstruction that supports VR inspection.
- Deploy a retrieval-augmented policy that recommends field inspections when erosion risk spikes.

**Latest milestone**

- 2025-05: Integrated NOAA wave forecasts into the agent and validated forecasts on Westport Beach survey data.

<a href="https://github.com/cxh42/coastalvision-agent" class="paper-link">Repository</a> | <a href="https://www.notion.so" class="paper-link">Lab notebook</a>

    </div>
  </div>
  <div class="paper-box ongoing-research-card">
    <div class="paper-box-image">
      <div>
        <div class="badge">Prototype</div>
        <img src="/images/3dgs-12.png" alt="NeuroGaussian Studio workspace" width="100%">
      </div>
    </div>
    <div class="paper-box-text" markdown="1">

<a href="https://github.com/cxh42/neurogaussian-studio" class="paper-title">NeuroGaussian Studio: Real-time 3DGS Authoring Toolkit</a>

**Team:** Xingahao Chen, UW Immersive Experiences cohort<br>
**Focus:** Author VR-ready environments from handheld captures with interactive relighting and agentic editing.

**What we are building**

- A streaming 3DGS renderer that compresses splats for standalone VR headsets.
- A co-pilot interface where language prompts drive material edits and object insertions.
- An evaluation harness that compares user comfort across retiming strategies.

**Latest milestone**

- 2025-04: Completed the first end-to-end capture-to-VR pipeline demo on Quest 3 hardware.

<a href="https://github.com/cxh42/neurogaussian-studio" class="paper-link">Repository</a> | <a href="https://uw.edu" class="paper-link">Course project page</a>

    </div>
  </div>
</div>

<style>
.ongoing-research-wrapper {
  display: grid;
  gap: 1.75rem;
}
.paper-box {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  border-radius: 18px;
  padding: 1.5rem;
  background: #ffffff;
  border: 1px solid #e0e7ff;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.paper-box.ongoing-research-card {
  border-left: 4px solid #1d4ed8;
  background: linear-gradient(135deg, rgba(219, 234, 254, 0.35), rgba(255, 255, 255, 0.9));
}
.paper-box:hover {
  transform: translateY(-6px);
  box-shadow: 0 18px 36px rgba(15, 23, 42, 0.12);
}
.paper-box-image {
  flex: 1 1 220px;
  max-width: 360px;
  display: flex;
  justify-content: center;
  margin-bottom: 1rem;
}
.paper-box-image > div {
  width: 100%;
}
.paper-box-image img {
  width: 100%;
  border-radius: 14px;
  object-fit: cover;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.15);
}
.paper-box-text {
  flex: 1 1 320px;
}
@media (min-width: 768px) {
  .paper-box {
    flex-wrap: nowrap;
  }
  .paper-box-image {
    margin-bottom: 0;
    margin-right: 1.75rem;
  }
}
.badge {
  display: inline-block;
  background: #1d4ed8;
  color: #ffffff;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-radius: 999px;
  padding: 0.25rem 0.75rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 4px 12px rgba(29, 78, 216, 0.35);
}
.paper-title {
  font-size: 1.2rem;
  font-weight: 600;
  color: #111827;
  text-decoration: none;
  border-bottom: none;
}
.paper-title:hover {
  color: #1d4ed8;
  text-decoration: none;
}
.paper-link {
  color: #1d4ed8;
  font-weight: 600;
  text-decoration: none;
  border-bottom: none;
}
.paper-link:hover {
  text-decoration: underline;
}
.ongoing-research-card p {
  margin-bottom: 0.75rem;
}
.ongoing-research-card ul {
  margin-bottom: 0.75rem;
  padding-left: 1.1rem;
}
.ongoing-research-card li {
  margin-bottom: 0.4rem;
}
</style>
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
**Email:** xhc42@outlook.com

**WeChat:** ICXH42


