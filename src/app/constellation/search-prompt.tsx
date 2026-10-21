"use client";

import { useState } from "react";

const AGENT_PROMPT = `Use Constellation's archive search to advance your current mission using your own targets, observations, and constraints.

Install the Python CLI if needed (Python 3.11+):
pipx install 'const-ellation @ git+https://github.com/WoshuaJolk/constellation.git@main'

If pipx is unavailable, create an isolated Python virtual environment and install that same GitHub package with pip. The package is const-ellation; the executable is constellation. Use the GitHub URL until a registry release is published.

Read the usage and input contracts:
https://github.com/WoshuaJolk/constellation/blob/main/README.md
https://github.com/WoshuaJolk/constellation/blob/main/docs/frames.md

Discover available providers, input modes, and credentials:
constellation capabilities
constellation find_frames --help
constellation credentials status

Build mission-query.json from your real mission inputs. Choose field, object, orbits, arc (MPC80), or ephemeris mode as appropriate. Use the documented schema, UTC date range, coordinates or orbital parameters, and uncertainty information. Supplied ephemerides require an explicit observer. Use providers: "all" for broad discovery, or select providers justified by the mission. Ask for essential missing inputs rather than inventing targets or using bundled demonstration data. Configure any archive credentials privately using the documented credential profiles.

Validate the query plan without contacting archives, then execute and audit it:
constellation find_frames mission-query.json --run-dir runs/mission --plan
constellation find_frames mission-query.json --run-dir runs/mission
constellation check_frames runs/mission --checksums

Use a new run directory for a new query. To inspect progress or resume the same query after its process has stopped:
constellation status runs/mission
constellation find_frames mission-query.json --run-dir runs/mission --deep --resume

Read manifest.json, frames.csv, and the query receipts in the run directory. Report native frame IDs, observation times, retrieval links, and provenance relevant to the mission. Distinguish returned candidates, failed or blocked queries, and unqueried coverage. SSOIS supplied-ephemeris search is disabled by default; direct archives can search the supplied track. Archive matches are not detections or proof of detectability. Search retrieves metadata and product references, not the image pixels; retrieve selected products separately when needed.`;

export function SearchPrompt() {
  const [status, setStatus] = useState<"idle" | "copying" | "copied" | "error">("idle");

  async function copyPrompt() {
    setStatus("copying");
    try {
      await navigator.clipboard.writeText(AGENT_PROMPT);
      setStatus("copied");
    } catch {
      setStatus("error");
    }
  }

  return (
    <div className="mt-5 text-xs">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <a href="https://github.com/WoshuaJolk/constellation/blob/main/README.md" className="underline underline-offset-4">
          Search surveys with constellation find_frames
        </a>
        <button
          type="button"
          onClick={copyPrompt}
          disabled={status === "copying"}
          aria-label="Copy agent prompt"
          title="Copy agent prompt"
          className="inline-flex h-8 w-8 items-center justify-center rounded text-[var(--text-secondary)] hover:bg-[var(--border)] hover:text-[var(--foreground)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-current disabled:opacity-50"
        >
          <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
            {status === "copied" ? <path d="m5 12 4 4L19 6" /> : <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3" /></>}
          </svg>
        </button>
        <span role="status" className="text-[var(--text-muted)]">
          {status === "copied" ? "Copied" : status === "error" ? "Clipboard unavailable. Select and copy the prompt below." : ""}
        </span>
      </div>
      {status === "error" && (
        <textarea
          aria-label="Agent prompt"
          readOnly
          value={AGENT_PROMPT}
          onFocus={(event) => event.currentTarget.select()}
          rows={8}
          className="mt-2 w-full max-w-3xl rounded border border-[var(--border)] bg-transparent p-3 text-xs leading-5"
        />
      )}
    </div>
  );
}
