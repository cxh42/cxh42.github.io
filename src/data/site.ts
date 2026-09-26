// Every fact on the page lives here. PRODUCT.md lists what is confirmed and what must stay off the site.

export const person = {
  name: 'Xinghao Chen',
  given: 'Xinghao',
  family: 'Chen',
  cjk: '陈星昊',
  role: 'Generative AI researcher, currently working on video generation.',
  email: 'cxh4242@gmail.com',
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

// Selected publications, newest first. The ViTeX figure plays real edits from its project page.
export const publications = [
  {
    id: 'vitex',
    year: '2026',
    title: 'ViTeX-Bench: Benchmarking High-Fidelity Video Scene Text Editing',
    short: 'ViTeX-Bench',
    authors: ['Xinghao Chen', 'Xiangbo Gao', 'Jiongze Yu', 'Yuheng Wu', 'Zhengzhong Tu'],
    venue: 'NeurIPS 2026',
    track: 'Evaluations & Datasets Track',
    // Paraphrased from the paper's abstract (revision of 2026-09-26).
    summary:
      'Video scene text editing must replace the characters on signs, boards and labels while the rest of the scene and its motion stay intact. ViTeX-Bench pairs ViTeX-Dataset (387 real-world 720p videos: 230 with reviewed paired edits for training, 157 frozen for evaluation) with 13 metrics over text correctness, visual and temporal quality, and edit locality, compared through one primary metric per axis and a Pareto front instead of a single score. Across eight baselines from four editing families, accurate text, temporal stability and scene preservation remain hard to get together. The open reference editor ViTeX-Edit-14B reaches the highest CharAcc among video-native editors (0.688).',
    pareto: true,
    leaderboard: 'https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/',
    links: [
      { label: 'Project page', href: 'https://vitex-bench.github.io/' },
      { label: 'Dataset', href: 'https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset' },
      { label: 'Benchmark code', href: 'https://github.com/ViTeX-Bench/ViTeX-Bench' },
      { label: 'Model', href: 'https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B' },
      { label: 'Leaderboard', href: 'https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/' },
    ],
    // Source (with text mask) and ViTeX-Edit-14B output, cropped side by side from the project page showcase.
    scenes: [
      { id: 'first-last', from: 'First', to: 'Last' },
      { id: 'soc-coc', from: 'SOC', to: 'COC' },
      { id: 'only-stop', from: 'ONLY', to: 'STOP' },
      { id: 'collier-washing', from: 'COLLIER', to: 'WASHING' },
    ],
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
    text: 'ViTeX-Bench accepted to NeurIPS 2026, Evaluations & Datasets Track.',
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
