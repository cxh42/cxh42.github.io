---
title: "Unity VR 3DGS Viewer · Meta Quest 2"
excerpt: "A Unity-based Meta Quest 2 VR app for browsing 3D Gaussian Splatting scenes.<br/><img src='/images/3dgs-9.png'>"
collection: portfolio
---

<br/><img src='/images/3dgs-9.png'>

Project Overview
=====
I built a Meta Quest 2 VR viewer in Unity to explore 3D Gaussian Splatting (3DGS) reconstructions. The app focuses on natural, comfortable viewing in-headset while balancing fidelity and performance on mobile VR.

Experience & Features
-----
- Map-style Home: Start from a campus map; select scenes via controller trigger.
- Quick Return: A “Main Page” button follows the user near the lower-right of view for easy back navigation.
- Info Panels: Each 3DGS scene includes an info panel with background and highlights.
- Free Roam: Natural walking and head-look. For smaller scenes, flying is intentionally omitted to maintain comfort.
- In Plan: Skybox switching (e.g., starry/daytime) and a linear guided tour mode (parallel to the map mode).

<br/><img src='/images/3dgs-8.png'>

Technical Notes
-----
- 3DGS Import: Collected data, reconstructed scenes, and imported 3DGS into Unity.
- VR UI & Interaction: Controller-trigger interactions; refined layout for the home panel and the follow-along return button to reduce “where is the button?” moments.
- Quest 2 Performance: Reduced splat counts for heavy models (e.g., sculptures), tuned UI readability and layout, and pursued a quality–framerate balance.

<br/><img src='/images/3dgs-10.png'>

Progress
-----
- Completed: Unity environment setup and required plugins; 3DGS file import; VR nav prototype; reconstructions for the central square and several key buildings; MVP demo and peer feedback.
- Current Content: Around 8 scenes (sculptures and buildings) accessible from the home map for free exploration.
- Known Challenge: Native 3DGS rendering on Quest 2 pushes performance; mitigations include model simplification and UI/interaction optimizations. Continued exploration of better render/load strategies.

<br/><img src='/images/3dgs-11.png'>

My Role
-----
- End-to-End: Interaction flow, scene organization, UI and info panels.
- 3DGS Pipeline: Data collection, reconstruction, import, and precise placement/scale tuning.
- Performance & UX: Model simplification, UI polish, readability and accessibility adjustments.
- Build & Delivery: Quest 2 APK builds and live demos.

Next
-----
- Expand 3DGS scene coverage and enrich info panels.
- Add skybox switching and a linear guided tour mode.
- Keep optimizing performance without sacrificing visual quality.

References
-----
- graphdeco-inria/gaussian-splatting: https://github.com/graphdeco-inria/gaussian-splatting
- aras-p/UnityGaussianSplatting: https://github.com/aras-p/UnityGaussianSplatting

<br/><img src='/images/3dgs-13.png'>
