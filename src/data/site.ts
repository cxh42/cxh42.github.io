// Every fact on the page lives here. PRODUCT.md lists what is confirmed and what must stay off the site.

export const person = {
  name: 'Xinghao Chen',
  given: 'Xinghao',
  family: 'Chen',
  cjk: '陈星昊',
  role: 'Generative AI researcher, currently working on video generation.',
  email: 'cxh@tamu.edu',
};

export const links = {
  openreview: 'https://openreview.net/profile?id=~Xinghao_Chen4',
  github: 'https://github.com/cxh42',
  linkedin: 'https://www.linkedin.com/in/cxh42',
  tamu: 'https://www.tamu.edu/',
  taco: 'https://taco-group.github.io/',
  advisor: 'https://vztu.github.io/',
};

// Research interests, in the order they appear on the page.
export const interests = [
  { name: 'Video generation', note: 'Current work' },
  { name: 'Large language models & vision-language models', note: '' },
  { name: '3D Gaussian Splatting', note: '' },
];

// Things I built, newest first. Shown after the publications.
export const projects = [
  {
    id: 'coastalseg',
    title: 'CoastalSeg',
    subtitle: 'Image segmentation for coastal erosion monitoring',
    year: '2025',
    context: 'University of Washington Applied Physics Laboratory · capstone project',
    text: 'A system for multi-class segmentation of shoreline photos uploaded by community members, with outlier detection and multi-image perspective correction. The segmentation model, DeepLabV3+ with an EfficientNet-B6 encoder, reaches 0.93 IoU. Built for coastal research at the UW Applied Physics Laboratory and used with MyCoast Washington.',
    team: 'With Zheheng Li, Dylan Scott, Aaryan Shah, Bauka Zhandulla and Sarah Li.',
    tags: ['Semantic segmentation', 'Outlier detection', 'Perspective alignment'],
    links: [
      { label: 'Code', href: 'https://github.com/cxh42/CoastalSeg' },
      { label: 'Demo', href: 'https://huggingface.co/spaces/AveMujica/CoastalSegment' },
    ],
    images: { before: 'coastalseg-photo', after: 'coastalseg-segmentation' },
  },
  {
    id: 'uw-vr',
    title: 'The UW campus in VR',
    subtitle: 'A walkable campus tour for Meta Quest',
    year: '2025',
    context: 'Developing Immersive Experiences for AR/VR, University of Washington',
    text: 'I photographed landmark buildings and sculptures around the University of Washington with a phone, reconstructed them as 3D Gaussian splats, and assembled them into a campus tour you can walk through in a Meta Quest headset.',
    team: '',
    tags: ['3D Gaussian Splatting', 'Meta Quest', 'Phone capture'],
    links: [],
    images: null,
  },
];

// Selected publications: the first-author paper leads as the featured entry, then the rest, newest first. Each one
// shows real before/after pairs from the paper or its project page, played through the sampler; the scene row
// names the edit once (Replace, Insert, Remove) and then lists the examples.
export type Scene = { id: string; from?: string; to: string; alt: string };

export const publications = [
  {
    id: 'vitex',
    year: '2026',
    title: 'ViTeX-Bench: Benchmarking High-Fidelity Video Scene Text Editing',
    short: 'ViTeX-Bench',
    authors: ['Xinghao Chen', 'Xiangbo Gao', 'Jiongze Yu', 'Yuheng Wu', 'Zhengzhong Tu'],
    venue: 'NeurIPS 2026',
    track: 'Evaluations & Datasets Track',
    featured: true,
    role: 'First author',
    // From the paper's abstract (revision of 2026-09-27).
    summary:
      'Video scene text editing replaces the words on signs, boards and labels in a real video while the rest of the scene and its motion stay intact. ViTeX-Bench pairs ViTeX-Dataset, 387 real-world 720p videos, with 13 metrics over text correctness, visual and temporal quality, and edit locality, and releases ViTeX-Edit-14B as an open reference editor.',
    pareto: {
      note: 'There is no single score. Each editor is a point on one primary metric per axis, and the front is the set no other editor beats on all three: across eight baselines from four editing families, accurate text, temporal stability and scene preservation remain hard to get together.',
      leaderboard: 'https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/',
    },
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2609.40356' },
      { label: 'Project page', href: 'https://vitex-bench.github.io/' },
      { label: 'Code', href: 'https://github.com/taco-group/ViTeX-Bench' },
      { label: 'Dataset', href: 'https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset' },
      { label: 'Model', href: 'https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B' },
      { label: 'Leaderboard', href: 'https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/' },
    ],
    // Source (with its text mask) and ViTeX-Edit-14B output, side by side, from the project page showcase.
    stage: {
      base: '/media/vitex',
      kind: 'video',
      verb: 'Replace',
      labels: ['Source, text mask in red', 'ViTeX-Edit-14B'],
      scenes: [
        { id: 'first-last', from: 'First', to: 'Last', alt: 'ViTeX-Edit-14B replacing “First” with “Last” on a chalkboard sign.' },
        { id: 'soc-coc', from: 'SOC', to: 'COC', alt: 'ViTeX-Edit-14B replacing “SOC” with “COC”.' },
        { id: 'only-stop', from: 'ONLY', to: 'STOP', alt: 'ViTeX-Edit-14B replacing “ONLY” with “STOP”.' },
        { id: 'collier-washing', from: 'COLLIER', to: 'WASHING', alt: 'ViTeX-Edit-14B replacing “COLLIER” with “WASHING”.' },
      ] as Scene[],
    },
  },
  {
    id: 'pisco',
    year: '2026',
    title: 'PISCO: Precise Video Instance Insertion with Sparse Control',
    short: 'PISCO',
    authors: ['Xiangbo Gao', 'Renjie Li', 'Xinghao Chen', 'Yuheng Wu', 'Suofei Feng', 'Jie Yang', 'Qing Yin', 'Zhengzhong Tu'],
    venue: 'NeurIPS 2026',
    track: 'Main Track',
    featured: false,
    role: '',
    pareto: null,
    summary:
      'Inserting an object into an existing video from a few keyframes at any timestamps. PISCO carries its appearance, motion, shadows and reflections through the clip while the original scene and its dynamics stay intact.',
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2602.08277' },
      { label: 'Project page', href: 'https://xiangbogaobarry.github.io/PISCO/' },
      { label: 'Code', href: 'https://github.com/taco-group/PISCO' },
      { label: 'Model', href: 'https://huggingface.co/xiangbog/PISCO-14B' },
    ],
    // Original footage and PISCO's insertion, side by side, from the project page's comparisons.
    stage: {
      base: '/media/pisco',
      kind: 'video',
      verb: 'Insert',
      labels: ['Original', 'PISCO'],
      scenes: [
        { id: 'lamp', to: 'Desk lamp', alt: 'PISCO inserting a lit desk lamp beside a marble bust.' },
        { id: 'rowboat', to: 'Rowboat', alt: 'PISCO inserting a toy in a small rowboat on a lake with swans, reflected in the water.' },
        { id: 'bear', to: 'Bear', alt: 'PISCO inserting a large teddy bear riding on a moving motorboat.' },
      ] as Scene[],
    },
  },
  {
    id: 'pvir',
    year: '2026',
    title: 'PVIR-Bench: A Physics-Aware Benchmark for Video Instance Removal',
    short: 'PVIR-Bench',
    authors: ['Zirui Li', 'Xinghao Chen', 'Lingyu Jiang', 'Xiangbo Gao', 'Dengzhe Hou', 'Kazunori Yamada', 'Fangzhou Lin', 'Zhengzhong Tu'],
    venue: 'CVPR 2026 Workshop',
    track: '',
    featured: false,
    role: '',
    pareto: null,
    summary:
      'Removing an object from a video along with the shadows, reflections and other physical effects it leaves behind. It holds 95 real videos with instance masks and removal prompts, split into Simple and Hard and rated by people on instruction following, rendering quality and edit exclusivity.',
    links: [{ label: 'Paper', href: 'https://arxiv.org/abs/2604.05898' }],
    // A benchmark frame and PISCO-Removal's result, from the paper's qualitative comparison (Fig. 1).
    stage: {
      base: '/media/pvir',
      kind: 'image',
      verb: 'Remove',
      labels: ['Source', 'PISCO-Removal'],
      scenes: [
        { id: 'duck', to: 'Duck', alt: 'A duck on a riverbank, and the same frame with the duck removed by PISCO-Removal.' },
        { id: 'kart', to: 'Go-kart', alt: 'A go-kart with two riders on a street, and the same frame with it removed by PISCO-Removal.' },
        { id: 'dancer', to: 'Dancer', alt: 'A dancer in front of an audience, and the same frame with her removed by PISCO-Removal.' },
      ] as Scene[],
    },
  },
];

export const education = [
  {
    id: 'tamu',
    school: 'Texas A&M University',
    abbr: 'Ph.D.',
    years: '2027 –',
    tint: '#500000',
    exposure: 1.45,
    degree: 'Ph.D. in Computer Science',
    dates: '2027 –',
    detail: 'TACO Group · Advisor: Dr. Zhengzhong Tu',
    incoming: true,
  },
  {
    id: 'uw',
    school: 'University of Washington',
    abbr: 'M.S.',
    years: '2024 – 2025',
    tint: '#4b2e83',
    exposure: 1,
    degree: 'M.S. in Electrical & Computer Engineering',
    dates: '2024 – 2025',
    detail: '',
    incoming: false,
  },
  {
    id: 'henu',
    school: 'Henan University',
    abbr: 'B.E.',
    years: '2020 – 2024',
    tint: '#0b4ea2',
    exposure: 1,
    degree: 'B.E. in Automation',
    dates: '2020 – 2024',
    detail: '',
    incoming: false,
  },
] as const;

export const news = [
  {
    date: 'Spring 2027',
    iso: '2027-01',
    upcoming: true,
    text: 'Joining Texas A&M University as a Ph.D. student in Computer Science, in the TACO Group advised by Dr. Zhengzhong Tu.',
  },
  {
    date: 'Sep 24, 2026',
    iso: '2026-09-24',
    upcoming: false,
    text: '[ViTeX-Bench](https://vitex-bench.github.io/), the project I led as first author, was accepted to the NeurIPS 2026 Evaluations & Datasets Track.',
  },
  {
    date: 'Sep 24, 2026',
    iso: '2026-09-24',
    upcoming: false,
    text: '[PISCO](https://xiangbogaobarry.github.io/PISCO/), led by [Xiangbo Gao](https://www.xiangbogao.com/) with me as a co-author, was accepted to the NeurIPS 2026 main track. Congratulations, Xiangbo!',
  },
  {
    date: 'Dec 2025',
    iso: '2025-12',
    upcoming: false,
    text: 'Graduated from the University of Washington with an M.S. in Electrical & Computer Engineering.',
  },
];

export const credits = [
  {
    work: 'Academic Building, Texas A&M University',
    author: 'Alexey Sergeev',
    license: '',
    licenseUrl: '',
    source: 'https://www.asergeev.com/',
  },
  {
    work: 'Suzzallo Library and Red Square',
    author: 'University of Washington',
    license: '',
    licenseUrl: '',
    source: 'https://www.washington.edu/',
  },
  {
    work: 'Henan University Auditorium (河南大学礼堂)',
    author: 'ScareCriterion12',
    license: 'CC BY-SA 4.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/4.0/',
    source: 'https://commons.wikimedia.org/wiki/File:%E6%B2%B3%E5%8D%97%E5%A4%A7%E5%AD%A6%E7%A4%BC%E5%A0%822020.jpg',
  },
];

export const sections = [
  { id: 'research', label: 'Research' },
  { id: 'publications', label: 'Publications' },
  { id: 'projects', label: 'Projects' },
  { id: 'education', label: 'Education' },
  { id: 'news', label: 'News' },
  { id: 'visitors', label: 'Visitors' },
  { id: 'contact', label: 'Contact' },
];
