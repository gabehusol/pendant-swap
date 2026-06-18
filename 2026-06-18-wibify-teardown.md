# Site Teardown: Wibify

**URL:** https://www.wibify.de/en  
**Built by:** Kerim Bilin / Wibify, confirmed from meta tags and page schema  
**Platform:** Next.js / React, inferred from `_next` asset paths and page content  
**Date analyzed:** 2026-06-18

## Tech Stack

| Technology | Evidence | Purpose |
|---|---|---|
| Next.js | `_next/static` CSS and JS chunks | App framework, routing, image optimization |
| React | Page copy references React; serialized React payload in HTML | Component rendering |
| Motion-style animation | JS chunks include Motion references | Reveal and interaction choreography |
| Custom fonts | Preloaded Switzer, Instrument Serif, JetBrains Mono-like font classes | Editorial typography system |
| Structured data | JSON-LD schema in source | SEO, business metadata, FAQ metadata |

## Design System

### Visual Language

Wibify uses a dark, editorial studio aesthetic: a fixed compact nav, full-viewport hero, large serif headline, mono section labels like `[01]`, textured grain overlay, glossy CTAs, dense portfolio/service sections, and scroll-revealed content bands.

### Adapted Palette

| Role | Value |
|---|---|
| Primary cerulean | `#0081A7` |
| Active teal | `#00AFB9` |
| Warm page surface | `#FDFCDC` |
| Soft panel surface | `#FED9B7` |
| Action coral | `#F07167` |

### Effects To Recreate

| Effect | Implementation | Complexity |
|---|---|---|
| Grain overlay | Fixed full-page texture layer with `mix-blend-mode` | Low |
| Sticky capsule nav | Blurred sticky nav with pill CTA | Low |
| Editorial reveal | IntersectionObserver adds a reveal class, CSS animates opacity/transform | Low |
| Shiny CTA | Animated pseudo-element sweep across primary action | Low |
| Numbered sections | Mono eyebrow rows with bracketed counters | Low |

## Build Notes For This Repo

The local app is static HTML/CSS/JS served by FastAPI, so the redesign keeps the existing backend contract and adapts the Wibify style without introducing a React build step. Motion guidance was applied through stagger-friendly reveal patterns, transform/opacity-only animation, touch-safe controls, visible focus states, and a reduced-motion media query.

## Assets Needed

No external Wibify assets are needed. The redesign uses CSS texture, typography, and layout rhythm rather than copying images or brand marks.
