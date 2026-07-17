---
name: Stock Volatility Prediction
description: XGBoost-powered volatility forecasting tool with dark terminal-inspired UI
colors:
  ink: "#0b0f1a"
  surface: "#111827"
  border: "#1e293b"
  stroke: "#334155"
  accent: "#22d3ee"
  accent-muted: "#155e75"
  orange: "#f97316"
  muted: "#9ca3af"
  foreground: "#d1d5db"
  white: "#f8fafc"
  green-bright: "#4ade80"
  red: "#ef4444"
typography:
  display:
    fontFamily: "Syne, sans-serif"
    fontWeight: 700
    fontSize: "clamp(1.75rem, 5vw, 2.5rem)"
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  body:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontWeight: 400
    fontSize: "0.875rem"
    lineHeight: 1.5
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontWeight: 400
    fontSize: "0.75rem"
    letterSpacing: "0.05em"
    textTransform: "uppercase"
  data:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontWeight: 500
    fontSize: "1.5rem"
  chart:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "0.6875rem"
rounded:
  none: "0px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.accent}"
    border: "1px solid {colors.accent}"
    padding: "8px 16px"
    typography: "{typography.label}"
  button-primary-hover:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.ink}"
  card:
    backgroundColor: "{colors.surface}"
    border: "1px solid {colors.border}"
    padding: "{spacing.md}"
  input:
    backgroundColor: "{colors.ink}"
    border: "1px solid {colors.stroke}"
    textColor: "{colors.white}"
    typography: "{typography.body}"
  input-focus:
    borderColor: "{colors.accent}"
  table-cell:
    padding: "{spacing.md}"
    typography: "{typography.body}"
---

# Design System: Stock Volatility Prediction

## 1. Overview

**Creative North Star: "The Trading Terminal"**

A dark, focused data tool for volatility analysis. The design takes visual cues from Bloomberg terminals, developer consoles, and trading dashboards: high-density information, monospaced data, cyan-on-black signaling, and zero decoration. Everything serves the data.

The interface rejects SaaS pastels, rounded cards, gradient accents, and decorative illustration. It is not friendly in the conventional sense — it is efficient, precise, and rewards familiarity.

**Key Characteristics:**
- Dark-on-dark color field with cyan signal color
- Monospaced text throughout labels, data, and UI copy
- Zero border-radius — hard edges consistent with terminal ethos
- Information-dense without clutter
- Sequential multi-ticker workflow with clear progress state
- Straight borders, no shadows, no glass effects

## 2. Colors

A restrained palette built around a near-black ink and cyan signal. The dark field provides high contrast for the accent to punch through.

### Primary
- **Cyan** (`#22d3ee`): Primary actions, interactive elements, loading indicators, chart prediction line. The single voice for "this is interactive."

### Accent
- **Orange** (`#f97316`): Chart actual values, warning states. Deliberately different from cyan to distinguish data sources.

### Semantic
- **Green Bright** (`#4ade80`): Best-score highlighting in comparison tables.
- **Red** (`#ef4444`): Error states and destructive actions.

### Neutral
- **Ink** (`#0b0f1a`): Body background. Nearly black with a blue-slate cast.
- **Surface** (`#111827`): Card, section, and container backgrounds. One step up from ink.
- **Border** (`#1e293b`): Default border color for cards, sections, tables.
- **Stroke** (`#334155`): Input borders, secondary dividers.
- **Muted** (`#9ca3af`): Secondary labels, chart tick text, placeholder text.
- **Foreground** (`#d1d5db`): Body text, data values.
- **White** (`#f8fafc`): Primary text, headings.

### Named Rules
**The Flat By Default Rule.** No shadows, no gradients, no glass effects. Depth is conveyed through tonal layering (Ink → Surface → border), not through elevation. Shadows are prohibited.

## 3. Typography

**Display Font:** Syne (sans-serif, geometric)
**Body/Label Font:** IBM Plex Mono (monospace)

**Character:** The pairing of Syne's precise geometric display with IBM Plex Mono's straightforward monospace creates a technical, authoritative voice. Syne is reserved for the page title only; everything else is monospace, reinforcing the data-terminal feel.

### Hierarchy
- **Display** (Syne 700, clamp 1.75rem–2.5rem, 1.1 line-height, -0.02em letter-spacing): Page title only. `text-wrap: balance`.
- **Data** (IBM Plex Mono 500, 1.5rem): Large metric values (RMSE, R²).
- **Body** (IBM Plex Mono 400, 0.875rem): All UI text, table cells, chart labels.
- **Label** (IBM Plex Mono 400, 0.75rem, 0.05em tracking, uppercase): Section headings, form labels, column headers.
- **Chart** (IBM Plex Mono 400, 0.6875rem): Axis tick labels inside Recharts.

### Named Rules
**The Mono Rule.** UI elements (labels, buttons, badges, tooltips) use IBM Plex Mono at 0.75rem, not the body size. This creates a consistent technical texture distinct from data values.

## 4. Elevation

The system is intentionally flat. Depth is achieved exclusively through tonal layering:
- **Background (Ink):** `#0b0f1a`
- **Surface (Surface):** `#111827` — one step lighter
- **Border:** `#1e293b` — defines boundaries

No box-shadows, no drop-shadows. The border is the only separation mechanism.

## 5. Components

### Buttons
- **Shape:** Hard edges (0px radius). Terminal-like, never rounded.
- **Primary (Analyze, Submit):** Transparent background, 1px cyan solid border, cyan text. 8px vertical / 16px horizontal padding. Label typography. Hover: fills cyan background, inverts text to ink. Transition: 150ms.
- **Ghost (Recent search chips):** Transparent, thin cyan border (30% opacity), cyan text. Hover: 10% cyan background fill.
- **Download CSV:** Same as ghost pattern. No icon dependency — uses text label only.

### Cards / Containers
- **Corner Style:** None (0px radius).
- **Background:** Surface (`#111827`).
- **Border:** 1px solid Border (`#1e293b`).
- **Shadows:** None.
- **Internal Padding:** 16px (spacing md).

### Inputs / Fields
- **Style:** Inset — transparent background (`#0b0f1a`), 1px solid Stroke border (`#334155`).
- **Focus:** Border shifts to accent cyan (`#22d3ee`). No outline, no glow, no ring.
- **Disabled:** 50% opacity.
- **Placeholder:** Muted (`#9ca3af`), same font as body.

### Tables
- **Style:** Full-width, no outer border. Row dividers at 50% opacity.
- **Header:** Label typography, left-aligned (right-aligned for numbers), sortable columns with arrow indicator on active sort.
- **Row:** Hover reveals 30% surface-tone overlay. Best-score cells highlighted in green.
- **Typography:** Body size (0.875rem) throughout.

### Accordion
- **Header:** Surface background, full-width clickable. Hover reveals 50% surface overlay. Shows ticker name (left) and RMSE/R² summary (right) with expand/collapse chevron.
- **Body:** Reveals MetricsPanel + charts + download button. No animation on expand — immediate.

### Loading Indicator
- **Style:** Terminal-style prompt (`$ Analyzing AAPL (1/3)...`). Surface background container. Blinking cursor character.

### Charts (Recharts)
- **Background:** None (transparent on Surface).
- **Grid:** Dashed lines at Border color (`#1e293b`).
- **Axis ticks:** Chart typography, Stroke color.
- **Tooltip:** Ink background, Border border, mono typography.

## 6. Do's and Don'ts

### Do:
- **Do** use cyan (`#22d3ee`) exclusively for interactive elements and the prediction line.
- **Do** use orange (`#f97316`) for actual values in charts.
- **Do** use hard 0px corners throughout — no border-radius anywhere.
- **Do** keep the dark field consistent — no light mode.
- **Do** use monospace for all UI text except the page title.
- **Do** show progress clearly during sequential multi-ticker operations.
- **Do** isolate errors per-ticker so partial results are still useful.

### Don't:
- **Don't** use rounded corners, shadows, gradients, or glass effects.
- **Don't** use light gray text (`#9ca3af`) on dark backgrounds for body copy — bump to `#d1d5db` or brighter.
- **Don't** add decorative motion or page-load animations.
- **Don't** use display fonts in UI labels, buttons, or data.
- **Don't** use card grids with identical icon-headline-body patterns.
- **Don't** put side-stripe borders (colored left/right border > 1px) on cards or items.
- **Don't** use uppercase tracked eyebrows above every section — one or two is voice; all of them is AI grammar.
- **Don't** use numbered section markers (01 / 02 / 03) as default scaffolding.
