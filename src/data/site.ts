// Every fact on the page lives here. PRODUCT.md lists what is confirmed and what must stay off the site.

export const person = {
  name: 'Xinghao Chen',
  given: 'Xinghao',
  family: 'Chen',
  cjk: '陈星昊',
  role: 'Generative AI researcher, focused on video generation.',
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

// Research areas, most recent first. Kept to what the site owner has confirmed.
export const areas = [
  { name: 'Video generation', note: 'Current focus: video scene text editing (ViTeX-Bench, NeurIPS 2026).' },
  { name: 'Vision-language models', note: 'Earlier research.' },
  { name: '3D Gaussian Splatting', note: 'Earlier research: scenes reconstructed from phone captures.' },
  { name: 'Virtual reality', note: 'A walkable UW campus tour for Meta Quest (2025).' },
];

export const projects = [
  {
    title: 'The University of Washington campus in VR',
    year: '2025',
    context: 'Developing Immersive Experiences for AR/VR, University of Washington',
    text: 'I photographed landmark buildings and sculptures around the University of Washington with a phone, reconstructed them as 3D Gaussian splats, and assembled them into a campus tour you can walk through in a Meta Quest headset.',
    tags: ['3D Gaussian Splatting', 'Meta Quest', 'Phone capture'],
  },
];

// Selected publications, newest first. The ViTeX figure plays real edits from its project page.
export const publications = [
  {
    id: 'vitex',
    year: '2026',
    title: 'ViTeX-Bench: Benchmarking High Fidelity Video Scene Text Editing',
    authors: ['Xinghao Chen', 'Xiangbo Gao', 'Jiongze Yu', 'Yuheng Wu', 'Zhengzhong Tu'],
    venue: 'NeurIPS 2026',
    track: 'Evaluations & Datasets Track',
    summary:
      'Video scene text editing replaces the words on signs, boards and labels in a video while the surrounding content, motion and camera stay untouched. ViTeX-Bench pairs 387 real-world 720p videos with text masks and instructions, scores every edit with 13 metrics across text correctness, visual quality and edit locality, and ships ViTeX-Edit-14B, a reference model that reaches 0.688 CharAcc.',
    links: [
      { label: 'Project page', href: 'https://vitex-bench.github.io/' },
      { label: 'Dataset', href: 'https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset' },
      { label: 'Benchmark', href: 'https://huggingface.co/ViTeX-Bench/ViTeX-Bench' },
      { label: 'Model', href: 'https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B' },
      { label: 'Leaderboard', href: 'https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard' },
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
    degree: 'Ph.D. in Computer Science',
    dates: 'Spring 2027 –',
    detail: 'TACO Group · Advisor: Dr. Zhengzhong Tu',
    incoming: true,
  },
  {
    id: 'uw',
    school: 'University of Washington',
    degree: 'M.S. in Electrical & Computer Engineering',
    dates: 'Sep 2024 – Dec 2025',
    detail: '',
    incoming: false,
  },
  {
    id: 'henu',
    school: 'Henan University',
    degree: 'B.E. in Automation',
    dates: 'Sep 2020 – Jun 2024',
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
    work: 'Academic Building',
    author: 'Texas A&M University',
    license: '',
    licenseUrl: '',
    source: 'https://stories.tamu.edu/',
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
  { id: 'education', label: 'Education' },
  { id: 'news', label: 'News' },
  { id: 'visitors', label: 'Visitors' },
  { id: 'contact', label: 'Contact' },
];
