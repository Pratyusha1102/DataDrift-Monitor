// Render a lightweight perspective particle view of a reference-to-current distribution shift.
(function () {
  const canvas = document.getElementById("drift-canvas");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (!canvas || reducedMotion.matches) return;
  const context = canvas.getContext("2d");
  if (!context) { canvas.hidden = true; document.querySelector(".visual-fallback").style.display = "flex"; return; }
  const groups = [{ centerX: .29, centerY: .52, color: "#1e5a89", offset: -.04 }, { centerX: .71, centerY: .45, color: "#0d766e", offset: .04 }];
  const particles = groups.flatMap((group, groupIndex) => Array.from({ length: 34 }, (_, index) => { const angle = index * 2.399; const radius = Math.sqrt((index + 1) / 34) * .17; return { groupIndex, x: group.centerX + Math.cos(angle) * radius, y: group.centerY + Math.sin(angle) * radius * .68, z: ((index * 13) % 29) / 29, radius: 1.8 + ((index * 7) % 5) * .35, offset: group.offset }; }));
  let width = 0, height = 0, animationFrame, phase = 0;
  // Resize for high-density screens while keeping the drawing lightweight.
  function resize() { const bounds = canvas.getBoundingClientRect(); const density = Math.min(window.devicePixelRatio || 1, 2); width = Math.max(1, Math.floor(bounds.width)); height = Math.max(1, Math.floor(bounds.height)); canvas.width = Math.floor(width * density); canvas.height = Math.floor(height * density); context.setTransform(density, 0, 0, density, 0, 0); }
  // Draw distribution clouds with small, restrained motion.
  function draw() { context.clearRect(0, 0, width, height); context.strokeStyle = "#d9e5eb"; context.lineWidth = 1; for (let line = 1; line < 5; line += 1) { const y = (height / 5) * line; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); } context.setLineDash([4, 7]); context.strokeStyle = "#afc2cd"; context.beginPath(); context.moveTo(width * .43, height * .52); context.lineTo(width * .57, height * .48); context.stroke(); context.setLineDash([]); particles.forEach((particle) => { const group = groups[particle.groupIndex]; const pulse = Math.sin(phase + particle.z * 7) * particle.offset; const perspective = .66 + particle.z * .55; context.beginPath(); context.fillStyle = group.color; context.globalAlpha = .28 + particle.z * .62; context.arc((particle.x + pulse) * width, particle.y * height, particle.radius * perspective, 0, Math.PI * 2); context.fill(); }); context.globalAlpha = 1; context.fillStyle = "#526579"; context.font = "600 11px system-ui, sans-serif"; context.fillText("MEASURED SHIFT", width * .43, height * .39); }
  function animate() { phase += .012; draw(); animationFrame = window.requestAnimationFrame(animate); }
  // Fall back to the window resize event on older browsers.
  if ("ResizeObserver" in window) { const observer = new ResizeObserver(resize); observer.observe(canvas); } else { window.addEventListener("resize", resize); }
  resize(); animate();
  reducedMotion.addEventListener("change", (event) => { if (event.matches) { window.cancelAnimationFrame(animationFrame); canvas.hidden = true; document.querySelector(".visual-fallback").style.display = "flex"; } });
}());
// Toggle compact navigation while retaining native links and keyboard support.
(function () { const toggle = document.querySelector(".menu-toggle"); const menu = document.querySelector(".site-menu"); if (!toggle || !menu) return; toggle.addEventListener("click", () => { const open = menu.classList.toggle("is-open"); toggle.setAttribute("aria-expanded", String(open)); }); menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => { menu.classList.remove("is-open"); toggle.setAttribute("aria-expanded", "false"); })); }());
