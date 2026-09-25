import { isStill, onFrame, watchVisible } from './gl';

// 陈星昊, typed and erased in turn, a different hand each time. Runs on the shared 24 fps clock:
// a character every 6 frames, a 3.5 s hold with the caret blinking at 1 Hz, erased at 3 frames a character.

const HANDS = [
  { family: 'CJK Kai', label: '楷书' },
  { family: 'CJK Xing', label: '行书' },
  { family: 'CJK Song', label: '宋体' },
  { family: 'CJK Cao', label: '草书' },
  { family: 'CJK WenKai', label: '楷体' },
  { family: 'CJK XingKai', label: '行楷' },
];
const NAME = '陈星昊';

export function initTyping() {
  const root = document.documentElement;
  const box = document.querySelector<HTMLElement>('[data-inscription]');
  const text = document.querySelector<HTMLElement>('[data-ins-text]');
  const caret = document.querySelector<HTMLElement>('[data-caret]');
  const label = document.querySelector<HTMLElement>('[data-ins-style]');
  if (!box || !text || !caret || !label || isStill()) return;

  // Warm every hand up front: six subsets of three glyphs, about 12 KB in all.
  const ready = Promise.all(
    HANDS.map((h) => (document.fonts ? document.fonts.load(`48px "${h.family}"`, NAME) : Promise.resolve())),
  ).catch(() => {});

  let hand = 0;
  let shown = NAME.length;
  let phase: 'wait' | 'hold' | 'erase' | 'gap' | 'type' = 'wait';
  let clock = 0;
  let visible = true;
  let fontsOk = false;
  ready.then(() => (fontsOk = true));

  const setHand = (i: number) => {
    text.style.fontFamily = `"${HANDS[i].family}", var(--font-cjk)`;
    label.textContent = HANDS[i].label;
  };
  setHand(0);

  watchVisible(box, (on) => (visible = on), '0px');

  onFrame(() => {
    if (!visible) return;
    clock++;
    switch (phase) {
      case 'wait':
        // The name settles first; the inscription is already written when it appears.
        if (root.classList.contains('hero-done') && fontsOk) {
          root.classList.add('typing');
          phase = 'hold';
          clock = 0;
        }
        break;
      case 'hold':
        caret.classList.toggle('off', Math.floor(clock / 12) % 2 === 1);
        if (clock >= 84) {
          phase = 'erase';
          clock = 0;
          caret.classList.remove('off');
        }
        break;
      case 'erase':
        if (clock % 3 === 0) {
          shown--;
          text.textContent = NAME.slice(0, shown);
          if (shown === 0) {
            phase = 'gap';
            clock = 0;
            label.classList.add('swap');
          }
        }
        break;
      case 'gap':
        caret.classList.toggle('off', Math.floor(clock / 12) % 2 === 1);
        if (clock >= 14) {
          hand = (hand + 1) % HANDS.length;
          setHand(hand);
          label.classList.remove('swap');
          caret.classList.remove('off');
          phase = 'type';
          clock = 0;
        }
        break;
      case 'type':
        if (clock % 6 === 0) {
          shown++;
          text.textContent = NAME.slice(0, shown);
          if (shown === NAME.length) {
            phase = 'hold';
            clock = 0;
          }
        }
        break;
    }
  });
}
