/**
 * A thumbnail of the whole frame with the inspected region drawn on it.
 * Click (or use the arrow keys) to move the region anywhere in the frame.
 */
export default function RegionPicker({ src, width, height, region, onPick, showBox = true }) {
  const move = (cx, cy) => {
    const x = Math.round(Math.min(Math.max(0, cx - region.w / 2), width - region.w));
    const y = Math.round(Math.min(Math.max(0, cy - region.h / 2), height - region.h));
    onPick({ x, y, w: region.w, h: region.h });
  };

  const onClick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    move(((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height);
  };

  const onKey = (e) => {
    const step = e.shiftKey ? region.w : region.w / 4;
    const cx = region.x + region.w / 2;
    const cy = region.y + region.h / 2;
    const keys = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (keys[e.key]) {
      e.preventDefault();
      move(cx + keys[e.key][0], cy + keys[e.key][1]);
    }
  };

  return (
    <div
      className="picker"
      style={{ aspectRatio: `${width} / ${height}` }}
      onClick={onClick}
      onKeyDown={onKey}
      tabIndex={0}
      role="group"
      aria-label={`Region picker. The region starts at ${region.x}, ${region.y}. Click or use the arrow keys to move it.`}
    >
      <img src={src} alt="" width={width} height={height} />
      {showBox && (
        <div
          className="picker__box"
          style={{
            left: `${(region.x / width) * 100}%`,
            top: `${(region.y / height) * 100}%`,
            width: `${(region.w / width) * 100}%`,
            height: `${(region.h / height) * 100}%`,
          }}
        />
      )}
    </div>
  );
}
