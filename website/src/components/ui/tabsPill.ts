/**
 * The pill switcher's class recipe, in ONE place.
 *
 * Two components render this control and they are not interchangeable:
 *
 *   ui/tabs.tsx  — Radix Tabs. Use when each tab owns its own panel, so the
 *                  trigger ⇄ panel `aria-controls` pair is real.
 *   Tablist.tsx  — a tablist and nothing else. Use when the body below is ONE
 *                  shared subtree parameterised by the active tab, where Radix's
 *                  unconditional `aria-controls` would point at nothing.
 *
 * They must look identical, because a user reading a page cannot see which of the
 * two accessibility shapes is underneath. A shared recipe is what makes that
 * structural rather than a promise: there is no second copy to drift.
 *
 * `SegmentedControl` deliberately keeps its own copy of these metrics — it is a
 * FILTER, a different control with the same skin, and folding it in here would
 * couple two independent decisions.
 */

/** The recessed track the segments sit in. */
export const TABS_TRACK_CLASS =
  'inline-flex items-center gap-0.5 rounded-lg border border-border bg-bg-elevated p-0.5'

/**
 * One segment. No border in the base: the sliding indicator carries it, so a
 * selection never shifts the label by a pixel, and the box metrics stay identical
 * to `SegmentedControl`'s.
 *
 * Focus uses the `.focus-ring` utility rather than the global `:focus-visible`
 * outline: that outline is `outline-offset: 2px`, which on a pill inside a 2px
 * track paints a box straddling the track's own border.
 */
export const TABS_SEGMENT_CLASS = [
  'focus-ring group/tab relative flex cursor-pointer items-center gap-1.5 whitespace-nowrap',
  'rounded-md px-2.5 py-1.5 text-[12px] font-medium',
  'text-muted transition-colors hover:text-text',
].join(' ')

/** Applied to the SELECTED segment, on top of `TABS_SEGMENT_CLASS`. */
export const TABS_SEGMENT_ACTIVE_CLASS = 'text-accent'

/** Applied to a segment the surface knows about and cannot serve yet. */
export const TABS_SEGMENT_DISABLED_CLASS = 'cursor-not-allowed text-muted/40 hover:text-muted/40'

/** The `aria-disabled:` form, for the same scanner reason as above. */
export const TABS_SEGMENT_DISABLED_ARIA_CLASS = [
  'aria-disabled:cursor-not-allowed',
  'aria-disabled:text-muted/40',
  'aria-disabled:hover:text-muted/40',
].join(' ')

/** The pill that slides between segments. Absolutely positioned over one segment. */
export const TABS_INDICATOR_CLASS = 'absolute inset-0 rounded-md border border-border bg-card shadow-sm'

/** Spring the indicator travels on. Matches `SegmentedControl`'s. */
export const TABS_INDICATOR_SPRING = { type: 'spring', stiffness: 500, damping: 35 } as const

/**
 * Trailing count, shared part only. The SELECTED colour is deliberately NOT here:
 * the two components learn which segment is selected by different means — Radix
 * from its own `data-state`, `Tablist` from a prop — so each spells its own
 * variant at the point of use. There is nothing common to drift.
 */
export const TABS_COUNT_BASE_CLASS = 'text-[11px]'
