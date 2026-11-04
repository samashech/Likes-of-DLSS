/** The mark: a wedge with a chevron and bar, drawn from the hand sketch. */
export default function Logo() {
  return (
    <svg className="logo" viewBox="0 0 128 64" aria-hidden="true" focusable="false">
      <g fill="none" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round">
        <path className="logo__body" d="M14 18 C 40 9, 90 7, 122 12 L 86 56 L 30 54 C 6 50, 2 28, 14 18 Z" />
        <path d="M62 19 L 16 30 L 76 35 L 40 44" />
        <path d="M56 29 L 94 32 L 86 54" />
      </g>
    </svg>
  );
}
