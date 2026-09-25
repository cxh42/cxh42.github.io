// Page chrome: the bar's scrolled state, the wordmark handover after the hero, the phone menu,
// and the schedule rail's marker.

export function initChrome() {
  const root = document.documentElement;
  const name = document.querySelector<HTMLElement>('[data-hero-name]');

  const onScroll = () => root.classList.toggle('scrolled', window.scrollY > 8);
  onScroll();
  window.addEventListener('scroll', onScroll, { passive: true });

  if (name) {
    new IntersectionObserver(
      ([e]) => root.classList.toggle('past-hero', !e.isIntersecting && e.boundingClientRect.top < 0),
      { rootMargin: '-64px 0px 0px 0px' },
    ).observe(name);
  }

  // Phone menu: a disclosure, closed by a link, Escape, or a tap elsewhere.
  const btn = document.querySelector<HTMLButtonElement>('[data-menu-btn]');
  const menu = document.querySelector<HTMLElement>('[data-menu]');
  if (btn && menu) {
    const set = (open: boolean) => {
      btn.setAttribute('aria-expanded', String(open));
      menu.hidden = !open;
    };
    btn.addEventListener('click', () => set(btn.getAttribute('aria-expanded') !== 'true'));
    menu.addEventListener('click', (e) => {
      if ((e.target as HTMLElement).closest('a')) set(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && btn.getAttribute('aria-expanded') === 'true') {
        set(false);
        btn.focus();
      }
    });
    document.addEventListener('click', (e) => {
      if (!menu.hidden && !(e.target as HTMLElement).closest('[data-bar]')) set(false);
    });
  }

  // Rail: the marker glides between section ticks as the reader moves through each section.
  const rail = document.querySelector<HTMLElement>('[data-rail]');
  const marker = document.querySelector<HTMLElement>('[data-rail-marker]');
  const railLinks = Array.from(document.querySelectorAll<HTMLAnchorElement>('[data-rail-link]'));
  const targets = railLinks.map((a) => document.getElementById(a.dataset.railLink!)).filter(Boolean) as HTMLElement[];
  if (!rail || !marker || targets.length !== railLinks.length) return;

  let pending = false;
  let lastActive = -1;
  const update = () => {
    pending = false;
    if (getComputedStyle(rail).display === 'none') return;
    const line = window.innerHeight * 0.4;
    let active = -1;
    let frac = 0;
    for (let i = 0; i < targets.length; i++) {
      const r = targets[i].getBoundingClientRect();
      if (r.top <= line) {
        active = i;
        frac = Math.min(1, Math.max(0, (line - r.top) / Math.max(1, r.height)));
      }
    }
    const railBox = rail.getBoundingClientRect();
    const tickY = (i: number) => {
      const tk = railLinks[i].querySelector('.rail-tick')!.getBoundingClientRect();
      return tk.top + tk.height / 2 - railBox.top;
    };
    let y: number;
    if (active < 0) y = tickY(0) - 14;
    else if (active === targets.length - 1) y = tickY(active);
    else y = tickY(active) + (tickY(active + 1) - tickY(active)) * frac;
    marker.style.setProperty('--marker-y', `${y}px`);
    const band = document.getElementById('education')?.getBoundingClientRect();
    const over = (yy: number) => !!band && yy >= band.top && yy <= band.bottom;
    railLinks.forEach((a, i) => a.classList.toggle('on-dark', over(tickY(i) + railBox.top)));
    marker.classList.toggle('on-dark', over(y + railBox.top));
    if (active !== lastActive) {
      lastActive = active;
      railLinks.forEach((a, i) => a.setAttribute('aria-current', String(i === active)));
    }
  };
  const request = () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(update);
    }
  };
  window.addEventListener('scroll', request, { passive: true });
  window.addEventListener('resize', request);
  request();
}
