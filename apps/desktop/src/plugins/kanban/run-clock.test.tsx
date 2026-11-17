import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanTask } from './types'
import { RunClock } from './ui'

let disposeLocales: () => void

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-09-16T18:00:00Z'))
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
})

afterEach(() => {
  cleanup()
  disposeLocales()
  vi.useRealTimers()
})

// #99819: a retried task's first start is hours old while the fresh run is
// minutes old — the clock must tick from the current run, not task.started_at.
describe('RunClock', () => {
  it('ticks from the current run start instead of the task first start', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'retried',
      status: 'running',
      started_at: now - 7200, // first start: 2h ago
      current_run_started_at: now - 90 // retry run: 90s ago
    }

    render(<RunClock task={task} />)

    expect(screen.getByText(`${en.working} · 1m`)).toBeTruthy()
  })

  it('falls back to started_at when the backend predates the field', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'legacy',
      status: 'running',
      started_at: now - 7200
    }

    render(<RunClock task={task} />)

    expect(screen.getByText(`${en.working} · 2h`)).toBeTruthy()
  })

  it('renders nothing when neither timestamp is present', () => {
    const { container } = render(<RunClock task={{ id: 't1', title: 'bare', status: 'running' }} />)

    expect(container.textContent).toBe('')
  })

  // The board marks a card whose live run EXISTS but whose start yields no
  // usable age. The engine never reclaims such a run (an unknown age is never
  // over the threshold), so the run's age is UNKNOWN — and the UI must not
  // fall back to the card's FIRST-EVER start, which is never reset on
  // re-dispatch. A card first started 30 days ago would otherwise render
  // "working · 2592000s".
  it('renders nothing when the live run age is unknown', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'unreadable clock',
      status: 'running',
      started_at: now - 2_592_000, // first start: 30 days ago
      current_run_started_at: null, // the live run's start does not parse
      current_run_age_unknown: true
    }

    const { container } = render(<RunClock task={task} />)

    expect(container.textContent).toBe('')
  })

  // Epoch 0 is readable but yields no usable age (no real run started then),
  // so the board marks it unknown too. The clock must not fall back to
  // started_at for it either.
  it('renders nothing when the live run age is unknown at epoch 0', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'epoch zero',
      status: 'running',
      started_at: now - 7200, // first start: 2h ago
      current_run_started_at: 0, // epoch 0: readable, but no usable age
      current_run_age_unknown: true
    }

    const { container } = render(<RunClock task={task} />)

    expect(container.textContent).toBe('')
  })

  // A backend that predates the epoch-0 marker sends 0 UNMARKED. The engine
  // never reclaims an epoch-0 run (its age is not usable), so the desktop
  // must not age it from the first-ever start either — nor render the ~56
  // years that an epoch-0 clock would otherwise read as.
  it('renders nothing for an epoch-0 run clock the backend left unmarked', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'unmarked epoch zero',
      status: 'running',
      started_at: now - 7200, // first start: 2h ago
      current_run_started_at: 0 // epoch 0, unmarked by an older backend
    }

    const { container } = render(<RunClock task={task} />)

    expect(container.textContent).toBe('')
  })

  // The marker is scoped to a card WITH a live run. A card with no live run
  // legitimately ages from its first-ever start — the engine's own
  // COALESCE(r.started_at, t.started_at) fallback — so the two sides agree
  // there and the marker must not leak into that path.
  it('still ages from started_at when the card has no live run', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'no live run',
      status: 'running',
      started_at: now - 7200,
      current_run_started_at: null,
      current_run_age_unknown: false
    }

    render(<RunClock task={task} />)

    expect(screen.getByText(`${en.working} · 2h`)).toBeTruthy()
  })

  // A backend that predates the marker rename emits the OLD spelling. The
  // desktop updates on a separate clock from the backend, so it must honour
  // that one too — otherwise the fix is inert against every backend that
  // predates the rename.
  it('renders nothing when the pre-rename marker spells it unreadable', () => {
    const now = Math.floor(Date.now() / 1000)

    const task: KanbanTask = {
      id: 't1',
      title: 'unreadable clock, old marker',
      status: 'running',
      started_at: now - 2_592_000, // first start: 30 days ago
      current_run_started_at: null,
      current_run_started_at_unreadable: true
    }

    const { container } = render(<RunClock task={task} />)

    expect(container.textContent).toBe('')
  })
})
