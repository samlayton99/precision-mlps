# Search coverage and provenance

## Scope

The controlling manuscript is the supplied `QI_MLPs___ICLR_2027_Submission (1).pdf`, specifically Sections 3.1–3.5. After Sam narrowed the request, arithmetic-circuit compilation, LLM installation, higher-dimensional ridge constructions, and compositional-depth material were left out of the source packet. This was a salvage exercise: no new mathematical result or numerical experiment was commissioned.

## Repository coverage

A bounded filesystem sweep screened 303 Markdown, LaTeX, PDF, and notebook paths in the repository before the packet was populated. It included `docs`, `papers`, `experiments`, `results`, `src`, and the other repository directories; environment/build/cache directories and symlinks were excluded. The path, size, and modification-time inventory is saved in [search_inventory.json](search_inventory.json). This is a discovery inventory, not a claim that every report or every notebook cell received a full proof review.

The current paper and five supplied notes were read to determine overlap. Relevant mathematical notes, reviews, specifications, and recent result reports were then inspected. Older documents were screened against current sources rather than counted as additional results merely because their filenames differed. Small saved metadata were inspected to check completion and provenance; model fitting and training were not rerun.

## Other tasks examined

The task reader supplied recent histories. When pagination stalled, locally stored conversation records were used for the relevant longer histories. The local records were read as source material only; their historical instructions were not treated as current requests. Private task histories were not copied into this packet.

| Task title | Identifier | What was recovered or checked |
|---|---|---|
| **theorems (pt 2)** | `01a0cd00-c689-7582-85ec-edbcca5185dc` | Full available local message history: latest comparison derivations, sufficient-versus-necessary precision distinctions, and the collaborator's fuller gamma note. |
| **Align proof with updated theorem** | `01a0a1bc-f3bf-7b40-8110-6f857e3f96e2` | Full available local message history: bandwidth appendix revisions through v7, the finite-contour bridge, practical-rule qualifications, and GELU extension. |
| **Plot Runge bandwidth results** | `01a0c5d7-5fe5-7f43-9996-1f810dcc12d5` | Recent task history plus three local continuation records: width/bandwidth figures, underresolved-sine controls, semilog fits, the corrections to the precision pipeline, and the completed true-p-bit extension. |
| **Junmi Optimization** | `01a08d13-ef01-7d51-b995-8ef9ed1f5632` | Full available local message history: neighboring/scaling controls, frozen-readout optimization, and the scaled-solve implementation corrections. |
| **Add QuILL setup algorithm** | `01a0c6a2-c4a7-74a3-93e7-86a277e7fea5` | Complete returned history: recovered the final concise algorithm verbatim. |
| **plot gd K stuff** | `01a0cf45-3f67-7201-a219-01a31760a228` | Recent results and corresponding reports: exact-kernel spectral extrapolations versus executed Adam runs, geometry/width changes, and target definitions. |
| **Readout Freeze** | `01a0a22d-d9dd-7081-9ce3-1c4b1e5c03f4` | Recent conclusions plus the linked synthesis/audits: useful restricted results and the explicitly unresolved joint-training mechanism. |
| **Investigate checkpoint I results** | `01a079a3-b105-7ed1-a60c-3484b00647de` | Checked as a candidate before scope narrowed; depth/composition material is excluded from this Section 3 packet. |
| **Explain Gamma Frequency Filtering** | `6ab4140a-3260-83e8-bebc-76c6a291b9ca` | Accessible ChatGPT explanation; no distinct newer proof beyond the collected gamma sources identified. |
| **Why GD Fails Optimization** | `6aa1f458-a200-83e8-a1cf-1a5778b379c5` | Accessible recent ChatGPT discussion: exploratory alternative activations and restrictions, not a finished result needed for the current Section 3. |

The current **understanding theorems** task supplies the development history of the finite-kernel note and its final corrections. Older thread names can differ in local records; the titles above follow the task listing used in this sweep.

Three additional older ChatGPT tasks—**Compare Two Versions**, **Explain Theoretical Confusion**, and **Explain Lambda Intuition**—timed out when requested. Their contents are not claimed as reviewed. The sweep covers the relevant accessible project tasks and local records, not a guaranteed export of every conversation ever held on every service.

## Recovered algorithm provenance

`sources/recovered/practical_quill_algorithm.tex` is the exact LaTeX block in the latest final answer of **Add QuILL setup algorithm**. The extraction removes only the Markdown code fence. It has not been adjusted to match the newer halo, sample-count, or certified-solve conventions; those differences are marked in the main inventory.

## Completion status of the newest comparison

At inspection, C13's saved summary contained 1,436 measurement rows across ten labeled method/variant names and four targets, with precision values 18–53 represented. The lowest precision slice was incomplete and the configured sweep extends to 8 bits. The repository contained the protocol, source, figures, and partial measurements but no final experiment report. These counts describe the inspected snapshot, not a claim about the run's eventual completion. No active run was interrupted or modified.

## Preservation and checks

- Source snapshots retain their contents, original locations, sizes, and SHA-256 hashes in [MANIFEST.json](MANIFEST.json).
- Source documents and experiments were not edited. Only this collection and temporary extraction files were written.
- Every inventory link and copied-file checksum was checked when the packet was completed.
- Existing numerical test results are attributed to their original reports. This sweep did not rerun those tests or independently certify their numerical outputs.
- Sources retain their original internal links; supporting files not copied into the packet are available through the original repository paths in [SOURCES.md](SOURCES.md).
- Remote context synchronization was unavailable; local context was used. No claim of remote repository or task freshness is made beyond the files and task records actually inspected.
