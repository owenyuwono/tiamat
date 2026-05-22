# Product

## Register

product

## Users

Tech-savvy people interested in fluid simulation: graphics programmers, CS students, and anyone who knows what SPH and FLIP mean. They want to see real-time WebGPU compute in action, tweak physics parameters, and compare solver behavior. The simulation runs in a browser tab on a desktop with a modern GPU.

## Product Purpose

A browser-based real-time fluid sandbox. Drop water into a box, watch it splash, tweak the physics. The simulation itself is the entire product; the UI exists only to let people explore it. Success looks like someone spending five minutes playing with sliders and sharing a screenshot.

## Brand Personality

Playful, minimal, precise.

## Anti-references

- Dense pro-tool UIs (Houdini, Blender parameter panels). Too much chrome for a sandbox.
- SaaS dashboards with card grids and metric tiles. Wrong genre entirely.
- Overly branded creative-tool landing pages. The fluid is the brand, not a logo.

## Design Principles

1. **The simulation is the interface.** Every pixel of UI competes with the fluid for attention. Earn each one.
2. **Invite experimentation.** Labels should make people curious enough to drag a slider, not intimidated by jargon.
3. **Stay out of the way.** Controls recede when not needed. The fullscreen render is the default experience.
4. **Show, don't label.** Where possible, let visual feedback replace text explanations.

## Accessibility & Inclusion

- WCAG AA contrast for all text over translucent panels.
- Reduced-motion preference: disable panel transitions (the simulation itself is inherently animated, no override needed).
- Keyboard navigable controls.
